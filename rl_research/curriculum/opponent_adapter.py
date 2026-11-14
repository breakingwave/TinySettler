"""Audited observation-world adapter around upstream ValueFunctionPlayer weights.

K=2 common sampled worlds, float64 scores, independent planning RNG. This is
NOT the unmodified upstream player and is not a search-agent repair. Resource
beliefs are conservative interval constraints, not an exact Bayesian posterior.
"""
import copy, itertools, random, time
from collections import Counter
import numpy as np
from catanatron.players.value import ValueFunctionPlayer, get_value_fn
from catanatron.models.actions import generate_playable_actions
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS
from catanatron.state_functions import player_key, player_num_resource_cards, player_num_dev_cards

K = 2

def hand(s,c):
    k=player_key(s,c)
    return [s.player_state[k+'_'+r+'_IN_HAND'] for r in RESOURCES]

def private_copy(game, rng):
    g=game.copy();g.random=random.Random();g.random.setstate(rng.getstate());g.state.random=g.random
    return g

class PublicKnowledge:
    """Per-observer resource bounds. Never stores the opponent's resource vector.

    Updated from own hand changes, opponent hand SIZE, and visible action types.
    Unknown opponent discards widen bounds. Public deterministic transfers update
    them. Production bounds use public board potential plus observed total gain;
    possible bank shortages are retained. Bounds intentionally lose correlations.
    """
    def __init__(self,color):
        self.color=color;self.low=[0]*5;self.high=[0]*5;self.updates=0
    def update(self,before,after,action,initial,free_road,potential):
        own0,total0=before;own1,total1=after;delta=[0]*5;t=action.action_type.value
        ours=action.color==self.color
        if t=='ROLL' and total1>total0:
            gain=total1-total0
            options=[p for p in itertools.product(*[(0,v) if v else (0,) for v in potential]) if sum(p)==gain]
            assert options,('production total inconsistent',gain,potential)
            lo=[min(p[i] for p in options) for i in range(5)];hi=[max(p[i] for p in options) for i in range(5)]
            self.low=[a+b for a,b in zip(self.low,lo)];self.high=[a+b for a,b in zip(self.high,hi)]
        elif not ours and t=='DISCARD_RESOURCE':
            self.low=[max(0,x-1) for x in self.low] # do not read discarded type
        else:
            costs={'BUILD_SETTLEMENT':[1,1,1,1,0],'BUILD_CITY':[0,0,0,2,3],'BUILD_ROAD':[1,1,0,0,0],'BUY_DEVELOPMENT_CARD':[0,0,1,1,1]}
            if t in ('MOVE_ROBBER','PLAY_MONOPOLY'):
                delta=[a-b for a,b in zip(own0,own1)] # two-player transfer known to observer
            elif not ours:
                if t in costs and not initial and not (t=='BUILD_ROAD' and free_road):delta=[-x for x in costs[t]]
                elif t=='BUILD_SETTLEMENT' and initial:delta=potential if total1>total0 else [0]*5
                elif t=='MARITIME_TRADE':
                    for r in action.value[:-1]:
                        if r is not None:delta[RESOURCES.index(r)]-=1
                    delta[RESOURCES.index(action.value[-1])]+=1
                elif t=='PLAY_YEAR_OF_PLENTY':
                    for r in action.value:delta[RESOURCES.index(r)]+=1
            self.low=[max(0,a+b) for a,b in zip(self.low,delta)];self.high=[max(0,a+b) for a,b in zip(self.high,delta)]
        # Enforce the publicly observed total and physical per-resource maximum.
        for _ in range(3):
            self.high=[min(x,19-own1[i],total1) for i,x in enumerate(self.high)]
            self.low=[max(x,total1-sum(self.high)+self.high[i]) for i,x in enumerate(self.low)]
            self.high=[min(x,total1-sum(self.low)+self.low[i]) for i,x in enumerate(self.high)]
        assert all(0<=l<=h for l,h in zip(self.low,self.high)) and sum(self.low)<=total1<=sum(self.high),(t,self.low,self.high,total1)
        self.updates+=1

def public_capture(game,color):
    s=game.state;other=next(c for c in s.colors if c!=color)
    return hand(s,color),player_num_resource_cards(s,other)

def potential_production(game,color,action):
    s=game.state;other=next(c for c in s.colors if c!=color);out=[0]*5
    if action.action_type.value=='BUILD_SETTLEMENT' and s.is_initial_build_phase and action.color==other:
        for tile in s.board.map.adjacent_tiles[action.value]:
            if tile.resource:out[RESOURCES.index(tile.resource)]+=1
    # For ROLL the result is only available AFTER execution; caller uses public dice
    # and the unchanged board to fill potential then.
    return out

def roll_potential(game,color):
    s=game.state;other=next(c for c in s.colors if c!=color);out=[0]*5
    record=s.action_records[-1];number=sum(record.result)
    if number==7:return out
    for node,(c,kind) in s.board.buildings.items():
        if c!=other:continue
        for tile in s.board.map.adjacent_tiles[node]:
            if tile.resource and tile.number==number and tile!=s.board.map.tiles[s.board.robber_coordinate]:
                out[RESOURCES.index(tile.resource)]+=2 if kind=='CITY' else 1
    return out

class AdaptedValuePlayer:
    label='Catanatron ValueFunctionPlayer, information/RNG adapter v1, K=2'
    def __init__(self,color,seed,value_fn='base'):
        assert value_fn in ('base','contender')
        self.color=color;self.value_fn=value_fn;self.rng=random.Random(seed)
        self.calls=0;self.seconds=0.;self.worlds=0;self.rejections=0;self.fallbacks=0
    def _world(self,game,knowledge):
        s=game.state;other=next(c for c in s.colors if c!=self.color)
        own=player_key(s,self.color);opp=player_key(s,other)
        ownhand=hand(s,self.color);total=player_num_resource_cards(s,other)
        low=list(knowledge.low);high=[min(h,19-ownhand[i]) for i,h in enumerate(knowledge.high)]
        observed=set(game.playable_actions)
        # Erase hidden hand, VP, age and ALL raw history before any scoring.
        g=private_copy(game,self.rng);g.seed=0;g.id='sampled-information-world';gs=g.state
        gs.action_records=[]
        for r in RESOURCES+DEVELOPMENT_CARDS:gs.player_state[opp+'_'+r+'_IN_HAND']=0
        for d in DEVELOPMENT_CARDS:
            if d!='VICTORY_POINT':gs.player_state[opp+'_'+d+'_OWNED_AT_START']=False
        remaining=Counter(starting_devcard_bank())
        for d in DEVELOPMENT_CARDS:
            remaining[d]-=s.player_state[own+'_'+d+'_IN_HAND']
            for key in (own,opp):remaining[d]-=s.player_state[key+'_PLAYED_'+d]
        assert min(remaining.values())>=0
        pool=list(remaining.elements());oppdev=player_num_dev_cards(s,other)
        # Public nonterminal fact rules out a sampled hidden already-winner.
        for _ in range(1000):
            self.rng.shuffle(pool);dev=pool[:oppdev]
            if s.player_state[opp+'_VICTORY_POINTS']+dev.count('VICTORY_POINT')<game.vps_to_win:break
        else:raise RuntimeError('no nonterminal development allocation')
        gs.development_listdeck=pool[oppdev:]
        for d in DEVELOPMENT_CARDS:
            gs.player_state[opp+'_'+d+'_IN_HAND']=dev.count(d)
            if d!='VICTORY_POINT':gs.player_state[opp+'_'+d+'_OWNED_AT_START']=bool(dev.count(d))
        # Opponent age flags do not affect this actor's one-action score. Sample
        # old/new assignment from public purchase count since its previous END_TURN.
        bought=0
        for record in reversed(s.action_records):
            if record.action.color==other:
                if record.action.action_type.value=='END_TURN':break
                if record.action.action_type.value=='BUY_DEVELOPMENT_CARD':bought+=1
        olddev=dev[:max(0,len(dev)-bought)]
        for d in DEVELOPMENT_CARDS:
            if d!='VICTORY_POINT':gs.player_state[opp+'_'+d+'_OWNED_AT_START']=d in olddev
        gs.player_state[opp+'_ACTUAL_VICTORY_POINTS']=s.player_state[opp+'_VICTORY_POINTS']+dev.count('VICTORY_POINT')
        def accept(counts):
            for i,r in enumerate(RESOURCES):gs.player_state[opp+'_'+r+'_IN_HAND']=counts[i]
            gs.resource_freqdeck=[19-ownhand[i]-counts[i] for i in range(5)]
            actions=generate_playable_actions(gs)
            if set(actions)!=observed:return False
            g.playable_actions=actions;return True
        for _ in range(128):
            counts=low.copy();available=[i for i in range(5) for _ in range(high[i]-low[i])]
            for i in self.rng.sample(available,total-sum(low)):counts[i]+=1
            if accept(counts):break
            self.rejections+=1
        else:
            # Public-only bounded enumeration, never revert to the real hidden state.
            self.fallbacks+=1;found=False
            for first in itertools.product(*[range(low[i],high[i]+1) for i in range(4)]):
                last=total-sum(first)
                if low[4]<=last<=high[4] and accept([*first,last]):found=True;break
            if not found:raise RuntimeError('no legal sampled world from public constraints')
        g.random=random.Random(self.rng.getrandbits(128));gs.random=g.random
        self.worlds+=1
        return g
    def decide(self,game,knowledge):
        tick=time.perf_counter();self.calls+=1
        if len(game.playable_actions)==1:
            self.seconds+=time.perf_counter()-tick;return game.playable_actions[0]
        worlds=[self._world(game,knowledge) for _ in range(K)]
        upstream=ValueFunctionPlayer(self.color,ValueFunctionPlayer.Params(value_fn=self.value_fn))
        fn=get_value_fn(upstream.value_fn_builder_name,upstream.params.weights)
        scores=[]
        for action in game.playable_actions:
            values=[]
            for world in worlds:
                g=private_copy(world,world.random);g.execute(action)
                values.append(np.float64(fn(g,self.color)))
            scores.append(np.mean(values,dtype=np.float64))
        self.seconds+=time.perf_counter()-tick
        assert np.isfinite(scores).all()
        return game.playable_actions[int(np.argmax(scores))]

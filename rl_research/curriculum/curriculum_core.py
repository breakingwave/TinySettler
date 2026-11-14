"""Starting-state distribution ablation; inherits immutable PPO implementation."""
import copy, json, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from env import Env, COLORS, opponent_action, action_index, decode, RESOURCES, DEVELOPMENT_CARDS, player_key
from train import Trainer, CONFIG, source_hashes, atomic_save, masked
from reliability_common import DiagnosticEnv, sha, write, read, rng_state, restore_rng
from opponent_adapter import AdaptedValuePlayer, PublicKnowledge, public_capture, potential_production, roll_potential

SEEDS=[104729,130363,155921]
MIX={'base':.5,'weighted':.25,'heuristic':.25}
RESET={'standard':{'fresh':1.},'curriculum':{'fresh':.5,'midgame':.25,'endgame':.25}}
OPPONENTS=['base','contender','random','weighted','heuristic']
RUNS=[(arm,s) for i,s in enumerate(SEEDS) for arm in (['standard','curriculum'] if i%2==0 else ['curriculum','standard'])]
CORPUS=None

def choose(rng,probs):
    assert all(v>=0 for v in probs.values()) and abs(sum(probs.values())-1)<1e-10
    return rng.choices(list(probs),weights=list(probs.values()))[0]

def load_corpus(path):
    global CORPUS
    CORPUS=torch.load(path,map_location='cpu',weights_only=False)
    return CORPUS

class ExperimentEnv(DiagnosticEnv):
    def __init__(self,seed,seat=0,opponent='base'):
        self.knowledge={c:PublicKnowledge(c) for c in COLORS}
        self.planners={c:AdaptedValuePlayer(c,seed+600000000000+i,opponent if opponent in ('base','contender') else 'base') for i,c in enumerate(COLORS)}
        self.category='fresh';self.snapshot_id=None;self.start_actions=0;self.start_turns=0
        super().__init__(seed,seat,opponent)
    def execute(self,a):
        before={c:public_capture(self.game,c) for c in COLORS}
        potential={c:potential_production(self.game,c,a) for c in COLORS}
        initial=self.game.state.is_initial_build_phase;free=self.game.state.is_road_building
        super().execute(a)
        for c in COLORS:
            if a.action_type.value=='ROLL':potential[c]=roll_potential(self.game,c)
            self.knowledge[c].update(before[c],public_capture(self.game,c),a,initial,free,potential[c])
    def advance(self):
        count=0
        while self.game.winning_color() is None and self.game.state.current_color()!=self.color:
            c=self.game.state.current_color()
            a=self.planners[c].decide(self.game,self.knowledge[c]) if self.opponent in ('base','contender') else opponent_action(self.game,self.opponent,self.rng)
            self.execute(a);count+=1
            if count>20000:raise RuntimeError('opponent advancement guard')
    @classmethod
    def reset_snapshot(cls,snapshot,seed,opponent):
        e=copy.deepcopy(snapshot['env']);assert isinstance(e,cls)
        e.opponent=opponent;e.category=snapshot['category'];e.snapshot_id=snapshot['id']
        e.start_actions=e.actions;e.start_turns=e.game.state.num_turns
        # Preserve shuffled deck order and all physical/history facts. Only future
        # dice/steal randomness changes. Planning and opponent action RNG independent.
        e.game.random=random.Random(seed);e.game.state.random=e.game.random;e.rng=random.Random(seed+981723)
        e.planners={c:AdaptedValuePlayer(c,seed+600000000000+i,opponent if opponent in ('base','contender') else 'base') for i,c in enumerate(COLORS)}
        e.action_counts={'learner':Counter(),'opponent':Counter()};e.public_states=Counter()
        assert e.game.winning_color() is None and e.game.state.current_color()==e.color and e.mask().any()
        assert e.actions<=15000 and e.game.state.num_turns<=750
        return e
    def diagnostic(self):
        return super().diagnostic()|dict(category=self.category,snapshot_id=self.snapshot_id,start_actions=self.start_actions,start_turns=self.start_turns,planning={c.value:dict(calls=p.calls,seconds=p.seconds,worlds=p.worlds,rejections=p.rejections,fallbacks=p.fallbacks) for c,p in self.planners.items()})

class ContinuationTrainer(Trainer):
    @classmethod
    def clone(cls,path,arm,seed,target,device):
        # Do not use Trainer.load on CPU for historical CUDA tensors.
        state=torch.load(path,map_location=device,weights_only=False)
        assert state['source_hashes']==source_hashes() and state['config']==CONFIG
        from env import MAPPING_HASH
        assert state['mapping_hash']==MAPPING_HASH
        t=state['trainer'];assert t.width==128 and t.n==434176 and t.seed==seed
        t.__class__=cls;t.device=device;t.arm=arm;t.origin_sha256=sha(path);t.origin_n=t.n
        t.target=target;t.n=t.updates=t.engine_actions=0;t.elapsed=t.update_seconds=0.
        t.metrics=[];t.episodes=[];t.rollout=[];t.replay.clear();t.truncation_diagnostics=[]
        t.reset_counts=Counter();t.category_transitions=Counter();t.planning_totals=Counter()
        # Same fresh RNG namespace for both arms within a starting-policy pair.
        namespace=710000000000+SEEDS.index(seed)*10000000
        t.namespace=namespace;t.episode=0;t.rng=random.Random(namespace)
        random.seed(namespace);np.random.seed(namespace%(2**32));torch.manual_seed(namespace)
        t.env=t.new_env()
        return t
    def new_env(self):
        assert CORPUS is not None
        self.episode+=1;category=choose(self.rng,RESET[self.arm]);opponent=choose(self.rng,MIX)
        seed=self.namespace+self.episode;seat=self.rng.randrange(2)
        if category=='fresh':e=ExperimentEnv(seed,seat,opponent)
        else:
            candidates=[s for s in CORPUS['snapshots'] if s['split']=='train' and s['category']==category]
            e=ExperimentEnv.reset_snapshot(self.rng.choice(candidates),seed,opponent)
        self.reset_counts[category]+=1
        return e
    def collect(self):
        old=self.env;self.category_transitions[old.category]+=1
        super().collect()
        if old is not self.env:
            self.engine_actions-=self.env.start_actions
            self.episodes[-1].update(category=old.category,snapshot_id=old.snapshot_id,start_actions=old.start_actions,start_turns=old.start_turns,continuation_actions=old.actions-old.start_actions)
            for p in old.planners.values():
                for key in ['calls','seconds','worlds','rejections','fallbacks']:self.planning_totals[key]+=getattr(p,key)
            if self.episodes[-1]['truncated']:self.truncation_diagnostics.append(dict(n=self.n,**old.diagnostic()))
    def save(self,path):
        super().save(path)
        write(Path(path).parent/'progress.json',dict(n=self.n,target=self.target,elapsed=self.elapsed,arm=self.arm,seed=self.seed,category_transitions=dict(self.category_transitions)))

def task_flags(e):
    """Legal, own/public-information exercise labels. No hidden opponent inputs."""
    actions=e.game.playable_actions;k=player_key(e.game.state,e.color);ps=e.game.state.player_state
    expansion=[a for a in actions if a.action_type.value=='BUILD_SETTLEMENT' and not e.game.state.is_initial_build_phase]
    knights=[a for a in actions if a.action_type.value=='PLAY_KNIGHT_CARD']
    productive=[];immediate=[]
    current_purchases={a for a in actions if a.action_type.value in ('BUILD_SETTLEMENT','BUILD_CITY','BUY_DEVELOPMENT_CARD')}
    for a in actions:
        kind=a.action_type.value
        if kind=='MARITIME_TRADE':
            g=e.game.copy();g.random=random.Random(0);g.state.random=g.random;g.execute(a)
            new=[b for b in g.playable_actions if b.action_type.value in ('BUILD_SETTLEMENT','BUILD_CITY','BUY_DEVELOPMENT_CARD') and b not in current_purchases]
            if new:productive.append((a,new))
        if kind in ('BUILD_SETTLEMENT','BUILD_CITY','BUILD_ROAD','PLAY_KNIGHT_CARD'):
            g=e.game.copy();g.random=random.Random(0);g.state.random=g.random;g.execute(a)
            if g.winning_color()==e.color:immediate.append(a)
    army=bool(knights and not ps[k+'_HAS_ARMY'] and ps[k+'_PLAYED_KNIGHT']>=2)
    return dict(expansion=expansion,knight=knights,largest_army=knights if army else [],productive_trade=productive,immediate_win=immediate)

def policy_action(t,e,rng=None):
    with torch.no_grad():
        q,_=t.net(t.tensor(e.obs()));probs=masked(q,t.tensor(e.mask(),torch.bool)).softmax(-1).cpu().numpy()
    a=int(np.argmax(probs)) if rng is None else int(rng.choice(len(probs),p=probs/probs.sum()))
    return a,probs

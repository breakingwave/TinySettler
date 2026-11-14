import copy,random,pickle
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from curriculum_core import *
from opponent_adapter import hand

def fast_game(seed=123):
    e=ExperimentEnv(seed,0,'weighted')
    for _ in range(80):
        _,r,term,trunc,_=e.step(action_index(opponent_action(e.game,'heuristic',e.rng),e.color))
        if term or trunc:break
    return e

def test_mapping_rng_and_conservation():
    e=fast_game();c=e.color;p=e.planners[c]
    for i,a in enumerate(e.game.playable_actions):assert decode(action_index(a,c),c)==a
    original=e.game.random.getstate();state=copy.deepcopy(p.rng.getstate())
    a=p.decide(e.game,e.knowledge[c]);assert a in e.game.playable_actions
    assert e.game.random.getstate()==original and e.game.state.random.getstate()==original
    p.rng.setstate(state)
    for _ in range(4):
        w=p._world(e.game,e.knowledge[c]);s=w.state
        assert all(s.resource_freqdeck[i]+sum(hand(s,c)[i] for c in s.colors)==19 for i in range(5))
        total=Counter(s.development_listdeck)
        for c in s.colors:
            k=player_key(s,c)
            for d in DEVELOPMENT_CARDS:total[d]+=s.player_state[k+'_'+d+'_IN_HAND']+s.player_state[k+'_PLAYED_'+d]
        from catanatron.models.decks import starting_devcard_bank
        assert total==Counter(starting_devcard_bank())
        assert set(w.playable_actions)==set(e.game.playable_actions)
        assert not s.action_records

def test_hidden_invariance():
    e=fast_game(29);g=e.game;c=e.color;other=next(c1 for c1 in g.state.colors if c1!=c);k=player_key(g.state,other)
    altered=copy.deepcopy(g)
    # Deliberately perturb every forbidden input, preserving allowed totals/actions.
    old=[altered.state.player_state[k+'_'+r+'_IN_HAND'] for r in RESOURCES]
    for r,n in zip(RESOURCES,old[::-1]):altered.state.player_state[k+'_'+r+'_IN_HAND']=n
    dev=[altered.state.player_state[k+'_'+d+'_IN_HAND'] for d in DEVELOPMENT_CARDS]
    for d,n in zip(DEVELOPMENT_CARDS,dev[::-1]):altered.state.player_state[k+'_'+d+'_IN_HAND']=n
    altered.state.player_state[k+'_ACTUAL_VICTORY_POINTS']=99
    altered.state.development_listdeck.reverse();altered.state.resource_freqdeck=[0]*5
    altered.random.seed(773);altered.state.random=altered.random
    for d in DEVELOPMENT_CARDS:
        if d!='VICTORY_POINT':altered.state.player_state[k+'_'+d+'_OWNED_AT_START']=True
    a=copy.deepcopy(e.planners[c]);b=copy.deepcopy(e.planners[c])
    wa=a._world(g,e.knowledge[c]);wb=b._world(altered,e.knowledge[c])
    assert wa.state.player_state==wb.state.player_state
    assert wa.state.resource_freqdeck==wb.state.resource_freqdeck and wa.state.development_listdeck==wb.state.development_listdeck
    a=copy.deepcopy(e.planners[c]);b=copy.deepcopy(e.planners[c])
    assert a.decide(g,e.knowledge[c])==b.decide(altered,e.knowledge[c])
    assert np.array_equal(e.obs(),observe_for(altered,c))

def observe_for(game,c):
    from env import observe
    return observe(game,c)

def test_mixture_is_wired():
    rng=random.Random(1);counts=Counter(choose(rng,MIX) for _ in range(20000))
    assert all(abs(counts[k]/20000-v)<.02 for k,v in MIX.items())
    assert {choose(rng,{'base':0.,'weighted':1.}) for _ in range(100)}=={'weighted'}
    assert {choose(rng,RESET['standard']) for _ in range(100)}=={'fresh'}

def test_public_belief_contains_true_world():
    for seed in range(3):
        e=ExperimentEnv(1000+seed,seed%2,'base')
        for n in range(300):
            for c in COLORS:
                other=next(o for o in COLORS if o!=c);h=hand(e.game.state,other);b=e.knowledge[c]
                assert all(l<=x<=u for l,x,u in zip(b.low,h,b.high))
            _,_,term,trunc,_=e.step(action_index(opponent_action(e.game,'weighted',e.rng),e.color))
            if term or trunc:break

def test_snapshot_facts_and_reward():
    e=fast_game(17);assert e.game.winning_color() is None
    s=dict(env=e,id='test',category='midgame')
    a=ExperimentEnv.reset_snapshot(s,111,'base');b=ExperimentEnv.reset_snapshot(s,222,'base')
    assert a.actions==e.actions and a.game.state.num_turns==e.game.state.num_turns
    assert a.game.state.player_state==e.game.state.player_state and a.game.state.action_records==e.game.state.action_records
    assert a.game.state.development_listdeck==e.game.state.development_listdeck
    assert np.array_equal(a.obs(),b.obs()) and np.array_equal(a.mask(),b.mask())
    assert a.game.random.getstate()!=b.game.random.getstate()
    act=next((x for x in a.game.playable_actions if x.action_type.value=='END_TURN'),a.game.playable_actions[0])
    _,reward,term,_,_=a.step(action_index(act,a.color));assert reward==0 if not term else abs(reward)==1

def test_real_reset_wiring_clone_and_recovery(tmp_path):
    import os
    import curriculum_core as core
    base=Path(os.environ.get('CURRICULUM_INPUTS','/tmp/tinysettler-curriculum-data'))
    cp=base/'104729.pt'
    if not cp.exists():
        cp=Path('/root/tinysettler-artifacts/tinysettler-ablation-20261111-b/retrieved/ppo-128-104729/434176.pt')
    corpus=load_corpus(base/'corpus.pt')
    a=ContinuationTrainer.clone(cp,'standard',104729,4096,'cpu')
    b=ContinuationTrainer.clone(cp,'curriculum',104729,4096,'cpu')
    assert a.n==b.n==0 and not a.rollout and not b.rollout
    assert all(torch.equal(x,y) for x,y in zip(a.net.parameters(),b.net.parameters()))
    for key in a.optimizer.state_dict()['state']:
        for field,x in a.optimizer.state_dict()['state'][key].items():
            y=b.optimizer.state_dict()['state'][key][field]
            assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
    previous=core.MIX;core.MIX={'weighted':1.}
    try:
        for _ in range(8):
            env=b.new_env();assert env.opponent=='weighted'
    finally:core.MIX=previous
    for _ in range(32):b.collect()
    b.save(tmp_path/'recovery.pt')
    for _ in range(16):b.collect()
    b.update();weights=[x.detach().clone() for x in b.net.parameters()]
    restored=Trainer.load(tmp_path/'recovery.pt')
    for _ in range(16):restored.collect()
    restored.update()
    assert all(torch.equal(x,y) for x,y in zip(weights,restored.net.parameters()))
    assert np.array_equal(b.env.obs(),restored.env.obs())
    assert b.category_transitions==restored.category_transitions
    assert all(b.env.planners[c].rng.getstate()==restored.env.planners[c].rng.getstate() for c in COLORS)

def test_corpus_splits_and_task_legality():
    import os
    corpus=load_corpus(Path(os.environ.get('CURRICULUM_INPUTS','/tmp/tinysettler-curriculum-data'))/'corpus.pt')
    groups={split:{s['source_game'] for s in corpus['snapshots'] if s['split']==split} for split in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    for s in corpus['snapshots']:
        e=s['env'];assert e.game.winning_color() is None and e.mask().any() and e.game.state.current_color()==e.color
        assert e.actions<=15000 and e.game.state.num_turns<=750
        assert (4<=e.points()<=6) if s['category']=='midgame' else (7<=e.points()<=9)
        flags=task_flags(e)
        assert all(flags[tag] for tag in s['tags'])

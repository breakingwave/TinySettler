import copy,random
import numpy as np
import torch
from env import Env, observe, TABLE, COLORS, INDEX, action_index, decode
from train import Trainer, double_target
from catanatron.state_functions import player_key

def test_privacy_and_dice():
    e=Env(42); g=copy.deepcopy(e.game); original=observe(g,e.color)
    other=next(c for c in COLORS if c!=e.color); key=player_key(g.state,other)
    g.state.player_state[key+'_WOOD_IN_HAND']+=1
    g.state.player_state[key+'_BRICK_IN_HAND']-=1
    g.state.resource_freqdeck[0]-=1; g.state.resource_freqdeck[1]+=1
    g.state.development_listdeck.reverse(); g.state.random.random()
    g.state.player_state[key+'_ACTUAL_VICTORY_POINTS']+=1
    assert np.array_equal(original,observe(g,e.color))
    tile=next(t for t in g.state.board.map.tiles_by_id.values() if t.number in (6,8))
    tile.number=14-tile.number
    assert not np.array_equal(original,observe(g,e.color))

def test_transitions_mapping_rewards():
    seen=set()
    for seed in range(3):
        e=Env(seed,seat=seed%2,opponent='weighted',max_actions=1000)
        for _ in range(1000):
            assert e.game.state.current_color()==e.color
            legal=np.flatnonzero(e.mask()); assert len(legal)
            for a in e.game.playable_actions:
                assert action_index(a,e.color) in legal
                assert decode(action_index(a,e.color),e.color)==a
                seen.add(a.action_type.value)
            _,r,term,trunc,_=e.step(random.choice(legal))
            assert r in (-1,0,1) and (r!=0)==term
            if term or trunc: break
        assert term or trunc
    assert {'BUILD_SETTLEMENT','BUILD_ROAD','ROLL','END_TURN'}<=seen

def test_truncation_bootstrap_position():
    e=Env(12,max_actions=1)
    _,r,term,trunc,_=e.step(np.flatnonzero(e.mask())[0])
    assert trunc and not term and r==0 and e.mask().any()
    assert e.game.state.current_color()==e.color

def test_double_selection():
    class Fixed:
        def __init__(self,x): self.x=torch.tensor([x],dtype=torch.float32)
        def __call__(self,obs): return self.x.expand(len(obs),-1),None
    y=double_target(Fixed([1,5,99]),Fixed([10,2,500]),torch.zeros(2,1),torch.tensor([[True,True,False]]*2),torch.tensor([0.,-1.]),torch.tensor([False,True]))
    assert y.tolist()==[2.,-1.]

def test_checkpoint_exact_recovery(tmp_path):
    for algo in ['ppo','ddqn']:
        t=Trainer(algo,128,41,5000)
        for _ in range(32): t.collect()
        path=tmp_path/(algo+'.pt'); t.save(path)
        for _ in range(5): t.collect()
        expected=(t.n,t.env.obs().copy(),list(t.replay),list(t.rollout))
        restored=Trainer.load(path)
        for _ in range(5): restored.collect()
        assert restored.n==expected[0]
        assert np.array_equal(restored.env.obs(),expected[1])
        assert len(restored.replay)==len(expected[2]) and len(restored.rollout)==len(expected[3])
        if algo=='ppo': restored.update()
        else:
            for _ in range(256): restored.collect()
            restored.update()
        assert restored.updates>0

def test_terminal_reward_sign():
    for winner_self in [True,False]:
        e=Env(56)
        action=np.flatnonzero(e.mask())[0]
        original=e.execute
        def execute(a):
            original(a)
            winner=e.color if winner_self else next(c for c in COLORS if c!=e.color)
            e.game.winning_color=lambda: winner
        e.execute=execute
        _,r,term,trunc,_=e.step(action)
        assert term and not trunc and r==(1 if winner_self else -1)


def test_actor_relative_mapping():
    for color in COLORS:
        for i in range(len(TABLE)):
            assert action_index(decode(i,color),color)==i

def test_evaluation_rng_isolation():
    from evaluate import evaluate
    import time
    t=Trainer('ppo',128,71,100)
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),t.n,t.env.obs().copy())
    assert len(evaluate(t,1900000001,1,time.time()+60))==6
    assert random.getstate()==state[0]
    assert np.array_equal(np.random.get_state()[1],state[1][1])
    assert torch.equal(torch.get_rng_state(),state[2])
    assert t.n==state[3] and np.array_equal(t.env.obs(),state[4])

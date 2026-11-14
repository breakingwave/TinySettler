import copy,random,time
import numpy as np
import pytest
import torch
from env import Env,RESOURCES,player_key
from train import Trainer
from reliability_common import DiagnosticTrainer,DiagnosticEnv,public_key,selection,fixed_deadline,write,read,rng_state,restore_rng
from reliability_evaluate import evaluate

def test_instrumentation_preserves_training_and_updates(tmp_path):
    a=Trainer('ppo',128,31,4096)
    b=DiagnosticTrainer('ppo',128,31,4096)
    initial=rng_state()
    for t in [a,b]:
        restore_rng(initial)
        for _ in range(256): t.collect()
        t.update()
    assert a.episodes==b.episodes and a.engine_actions==b.engine_actions
    assert np.array_equal(a.env.obs(),b.env.obs())
    for x,y in zip(a.net.parameters(),b.net.parameters()): assert torch.equal(x,y)
    b.save(tmp_path/'checkpoint.pt'); expected=rng_state(); b.collect()
    c=Trainer.load(tmp_path/'checkpoint.pt');c.collect()
    assert c.n==b.n and np.array_equal(c.env.obs(),b.env.obs())
    assert c.env.action_counts==b.env.action_counts and c.env.public_states==b.env.public_states

def test_public_diagnostic_excludes_hidden_state():
    e=DiagnosticEnv(17); before=public_key(e); s=e.game.state
    other=next(c for c in s.colors if c!=e.color);k=player_key(s,other)
    s.player_state[k+'_WOOD_IN_HAND']+=2
    s.player_state[k+'_VICTORY_POINT_IN_HAND']+=1
    s.resource_freqdeck[0]+=2
    assert public_key(e)==before
    own=player_key(s,e.color);s.player_state[own+'_WOOD_IN_HAND']+=1
    assert public_key(e)!=before

def test_truncation_counts_and_selection():
    e=DiagnosticEnv(4,max_turns=1)
    while True:
        _,r,term,trunc,_=e.step(np.flatnonzero(e.mask())[0])
        if term or trunc:break
    assert trunc and r==0
    d=e.diagnostic();assert d['turn_limit_reached'];assert sum(sum(c.values()) for c in d['action_type_counts'].values())==e.actions
    def candidate(step,w,tr):return dict(step=step,rows=[dict(win=i<w,truncated=w<=i<w+tr) for i in range(6)])
    assert selection([candidate(10,4,1),candidate(20,4,0)])['step']==20
    assert selection([candidate(10,4,0),candidate(20,4,0)])['step']==10
    assert selection([candidate(10,5,1),candidate(20,4,0)])['step']==10

def test_resume_deadline_and_evaluation_rng(tmp_path):
    p=tmp_path/'deadline.json';fixed_deadline(p,1234);fixed_deadline(p,1234)
    with pytest.raises(AssertionError):fixed_deadline(p,2234)
    t=DiagnosticTrainer('ppo',128,5,4096);before=rng_state()
    rows=evaluate(t,2100000000,1,time.time()-1,tmp_path/'eval.json','hash')
    after=rng_state();assert rows==[] and before[0]==after[0] and torch.equal(before[2],after[2])
    assert np.array_equal(before[1][1],after[1][1])

def test_evaluation_matches_original_and_resumes_complete_blocks(tmp_path):
    from evaluate import evaluate as original_evaluate
    t=DiagnosticTrainer('ppo',128,7,4096)
    before=rng_state();old=original_evaluate(t,1800000000,1,time.time()+120);restore_rng(before)
    new=evaluate(t,1800000000,1,time.time()+120,tmp_path/'evaluation.json','unchanged')
    fields=['board_seed','seat','opponent','reward','win','loss','truncated','points','engine_actions','decisions']
    assert len(old)==len(new)==6
    assert [{k:r[k] for k in fields} for r in old]==[{k:r[k] for k in fields} for r in new]
    assert evaluate(t,1800000000,1,time.time()-1,tmp_path/'evaluation.json','unchanged')==new
    assert before[0]==rng_state()[0] and torch.equal(before[2],rng_state()[2])

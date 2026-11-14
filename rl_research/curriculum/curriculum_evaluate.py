"""Independent-RNG full-game and source-game-block exercise evaluation."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from curriculum_core import *

def play(t,e,deadline,task=None,policy_seed=None):
    start=time.perf_counter();initial_turn=e.game.state.num_turns;initial_actions=e.actions
    initial_points=e.points();k=player_key(e.game.state,e.color);initial_army=e.game.state.player_state[k+'_HAS_ARMY']
    calls=0;latency=0.;success=False;completion=None;productive_pending=False
    rng=np.random.default_rng(policy_seed) if policy_seed is not None else None
    while True:
        if time.time()>=deadline:raise TimeoutError('evaluation deadline')
        obs=t.tensor(e.obs());mask=t.tensor(e.mask(),torch.bool)
        with torch.no_grad():
            if t.device=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter();q,_=t.net(obs);probs=masked(q,mask).softmax(-1)
            a=int(probs.argmax().item()) if rng is None else int(rng.choice(len(probs),p=probs.cpu().numpy()))
            if t.device=='cuda':torch.cuda.synchronize()
            latency+=time.perf_counter()-tick;calls+=1
        kind=decode(a,e.color).action_type.value
        if task=='productive_trade' and kind=='MARITIME_TRADE':
            productive_pending=any(action_index(x[0],e.color)==a for x in task_flags(e)['productive_trade']) or productive_pending
        _,r,terminal,truncated,_=e.step(a)
        hit=(task=='expansion' and kind=='BUILD_SETTLEMENT') or (task=='knight' and kind=='PLAY_KNIGHT_CARD') or (task=='largest_army' and not initial_army and e.game.state.player_state[k+'_HAS_ARMY']) or (task=='productive_trade' and productive_pending and kind in ('BUILD_CITY','BUILD_SETTLEMENT','BUY_DEVELOPMENT_CARD')) or (task=='immediate_win' and r==1 and calls==1)
        if hit and not success:success=True;completion=dict(learner_decisions=calls,turns=e.game.state.num_turns-initial_turn,engine_actions=e.actions-initial_actions)
        if kind=='END_TURN':productive_pending=False
        if terminal or truncated:break
    return dict(win=r==1,loss=r==-1,truncated=truncated,terminal=terminal,reward=r,points=e.points(),initial_points=initial_points,decisions=calls,engine_actions=e.actions-initial_actions,seconds=time.perf_counter()-start,inference_seconds=latency,illegal_actions=0,task=task,task_success=success if task else None,completion=completion,diagnostic=e.diagnostic())

def evaluate_full(t,start,boards,deadline,path,checkpoint_hash):
    path=Path(path);saved=read(path) if path.exists() else dict(seed_start=start,boards=boards,checkpoint_sha256=checkpoint_hash,rows=[],wall_seconds=0)
    assert (saved['seed_start'],saved['boards'],saved['checkpoint_sha256'])==(start,boards,checkpoint_hash)
    assert len(saved['rows'])%10==0
    state=rng_state();tick=time.time()
    try:
        for board in range(len(saved['rows'])//10,boards):
            block=[]
            for opponent in OPPONENTS:
                for seat in [0,1]:
                    e=ExperimentEnv(start+board,seat,opponent)
                    row=play(t,e,deadline)|dict(board_seed=start+board,seat=seat,opponent=opponent)
                    block.append(row)
            saved['rows'].extend(block);write(path,saved|dict(complete=len(saved['rows'])==boards*10,wall_seconds=saved['wall_seconds']+time.time()-tick))
        return saved['rows']
    finally:restore_rng(state);write(path,saved|dict(complete=len(saved['rows'])==boards*10,wall_seconds=saved['wall_seconds']+time.time()-tick))

def exercise_manifest(corpus,per_task=3,split='test'):
    rows=[]
    for task in ['largest_army','expansion','productive_trade','immediate_win']:
        seen=set()
        for s in corpus['snapshots']:
            if s['split']!=split or task not in s['tags'] or s['source_game'] in seen:continue
            rows.append(dict(snapshot_id=s['id'],source_game=s['source_game'],task=task,category=s['category']))
            seen.add(s['source_game'])
            if len(seen)>=per_task:break
        assert seen,task
    return rows

def evaluate_tasks(t,corpus,cases,deadline,path,checkpoint_hash):
    path=Path(path);saved=read(path) if path.exists() else dict(cases=cases,checkpoint_sha256=checkpoint_hash,rows=[])
    assert saved['checkpoint_sha256']==checkpoint_hash and saved['cases']==cases
    state=rng_state();lookup={s['id']:s for s in corpus['snapshots']}
    try:
        for case in cases:
            for repeat in range(2):
                if any(r['snapshot_id']==case['snapshot_id'] and r['task']==case['task'] and r['repeat']==repeat for r in saved['rows']):continue
                e=ExperimentEnv.reset_snapshot(lookup[case['snapshot_id']],860000000000+case['source_game']*10+repeat,'base')
                row=play(t,e,deadline,case['task'])|case|dict(repeat=repeat,opponent='base')
                saved['rows'].append(row);write(path,saved)
        saved['complete']=True;return saved['rows']
    finally:restore_rng(state);write(path,saved)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-start',type=int,required=True);p.add_argument('--boards',type=int,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    t=Trainer.load(a.checkpoint);evaluate_full(t,a.seed_start,a.boards,a.deadline,a.output,sha(a.checkpoint))

"""Resumable complete-board evaluation with isolated RNG and public stall diagnostics."""
import argparse,time
from pathlib import Path
import torch
from train import Trainer,masked
from reliability_common import DiagnosticEnv,rng_state,restore_rng,read,write,sha

def evaluate(t,start,boards,deadline,path,checkpoint_hash=None):
    path=Path(path); state=rng_state(); started=time.time()
    saved=read(path) if path.exists() else dict(rows=[],seed_start=start,boards_target=boards,checkpoint_sha256=checkpoint_hash,wall_seconds=0.)
    assert saved['seed_start']==start and saved['boards_target']==boards
    assert saved['checkpoint_sha256']==checkpoint_hash
    rows=saved['rows']; assert len(rows)%6==0
    try:
        for board in range(len(rows)//6,boards):
            block=[]
            for opponent in ['random','weighted','heuristic']:
                for seat in [0,1]:
                    if time.time()>=deadline: return rows
                    e=DiagnosticEnv(start+board,seat,opponent); calls=0; latency=0.; ts=time.perf_counter()
                    while True:
                        if time.time()>=deadline: return rows
                        with torch.no_grad():
                            x=t.tensor(e.obs()); m=t.tensor(e.mask(),torch.bool)
                            if t.device=='cuda': torch.cuda.synchronize()
                            tick=time.perf_counter(); q,_=t.net(x); a=int(masked(q,m).argmax().item())
                            if t.device=='cuda': torch.cuda.synchronize()
                            latency+=time.perf_counter()-tick; calls+=1
                        _,r,terminal,truncated,_=e.step(a)
                        if terminal or truncated: break
                    row=dict(board_seed=start+board,seat=seat,opponent=opponent,reward=r,win=r==1,loss=r==-1,terminal=terminal,truncated=truncated,points=e.points(),engine_actions=e.actions,decisions=calls,seconds=time.perf_counter()-ts,inference_seconds=latency,illegal_actions=0)
                    if truncated: row['diagnostic']=e.diagnostic()
                    block.append(row)
            rows.extend(block)
            write(path,saved|dict(rows=rows,wall_seconds=saved['wall_seconds']+time.time()-started,complete=len(rows)==boards*6))
        return rows
    finally:
        restore_rng(state)
        write(path,saved|dict(rows=rows,wall_seconds=saved['wall_seconds']+time.time()-started,complete=len(rows)==boards*6))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--seed-start',type=int,required=True);p.add_argument('--boards',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evaluate(Trainer.load(a.checkpoint),a.seed_start,a.boards,a.deadline,a.output,sha(a.checkpoint))

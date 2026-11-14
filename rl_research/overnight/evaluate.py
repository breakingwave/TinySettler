"""Paired-board evaluation with independent RNG and clustered intervals."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from env import Env
from train import Trainer, masked, CONFIG

def evaluate(t,seed_start,boards,deadline):
    # Preserve all training RNG streams, including CUDA policy sampling.
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    rows=[]
    try:
        for board in range(boards):
            group=[]
            for opponent in ['random','weighted','heuristic']:
                for seat in [0,1]:
                    if time.time()>=deadline: return rows
                    env=Env(seed_start+board,seat,opponent); done=False; calls=0; latency=0.; start=time.perf_counter()
                    while not done:
                        if time.time()>=deadline: return rows # discard partial six-game board block uniformly
                        with torch.no_grad():
                            x=t.tensor(env.obs()); m=t.tensor(env.mask(),torch.bool)
                            if t.device=='cuda': torch.cuda.synchronize()
                            ts=time.perf_counter(); q,_=t.net(x); a=int(masked(q,m).argmax().item())
                            if t.device=='cuda': torch.cuda.synchronize()
                            latency+=time.perf_counter()-ts; calls+=1
                        _,r,terminal,truncated,_=env.step(a); done=terminal or truncated
                    group.append(dict(board_seed=seed_start+board,seat=seat,opponent=opponent,reward=r,win=r==1,loss=r==-1,truncated=truncated,points=env.points(),engine_actions=env.actions,decisions=calls,seconds=time.perf_counter()-start,inference_seconds=latency,illegal_actions=0))
            rows.extend(group)
        return rows
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3]: torch.cuda.set_rng_state_all(state[3])

def summarize(rows):
    out={}; rng=np.random.default_rng(718)
    for opp in ['random','weighted','heuristic']:
        group=[r for r in rows if r['opponent']==opp]
        seeds=sorted(set(r['board_seed'] for r in group))
        pairs=np.asarray([np.mean([r['win'] for r in group if r['board_seed']==s]) for s in seeds])
        ci=np.quantile(rng.choice(pairs,(2000,len(pairs)),replace=True).mean(1),[.025,.975]).tolist() if len(pairs) else [None,None]
        out[opp]=dict(games=len(group),wins=sum(r['win'] for r in group),losses=sum(r['loss'] for r in group),truncations=sum(r['truncated'] for r in group),completed=sum(not r['truncated'] for r in group),win_rate_ci95_clustered=ci,points=float(np.mean([r['points'] for r in group])) if group else None,inference_ms=1000*sum(r['inference_seconds'] for r in group)/max(1,sum(r['decisions'] for r in group)),inference_location='Colab when executed remotely; excludes observation encoding and host-to-device transfer')
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--split',choices=['validation','final'],required=True); p.add_argument('--boards',type=int,default=100); p.add_argument('--seconds',type=int,default=3600); p.add_argument('--output',required=True); a=p.parse_args()
    t=Trainer.load(a.checkpoint); rows=evaluate(t,CONFIG['validation_seed_start' if a.split=='validation' else 'final_seed_start'],a.boards,time.time()+a.seconds)
    Path(a.output).write_text(json.dumps(dict(rows=rows,summary=summarize(rows)),indent=2))

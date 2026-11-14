"""Independent sampled/greedy panels; atomic per-game rows and matched board blocks."""
from sc_train import *
MAIN=['base','contender','heldout']
WEAK=['random','weighted','heuristic']

def get_policy(path,device='cuda'):
    t=ScalingTrainer.load(path,device);t.net.eval();return t.net,SCHEMA,t.kind

def play(net,schema,e,mode,policy_seed,deadline):
    rng=np.random.default_rng(policy_seed);start=time.perf_counter();inference=0.;e2e=0.;decisions=0
    while True:
        if time.time()>=deadline:raise TimeoutError('evaluation deadline')
        tick=time.perf_counter();x=observe(e.game,e.color);x=x if schema=='legacy-2049' else compact(x);m=e.mask();device=next(net.parameters()).device
        xx=torch.as_tensor(x,device=device);mm=torch.as_tensor(m,device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        forward=time.perf_counter()
        with torch.no_grad():q,_=net(xx);p=masked(q,mm).softmax(-1).cpu().numpy().astype(np.float64)
        if device.type=='cuda':torch.cuda.synchronize()
        inference+=time.perf_counter()-forward;p/=p.sum()
        a=int(p.argmax()) if mode=='greedy' else int(rng.choice(332,p=p));e2e+=time.perf_counter()-tick
        e.record_decision(a,m,p,check_win=True,observation=compact(x) if schema=='legacy-2049' else x);_,reward,terminal,truncated,_=e.step(a);decisions+=1
        if terminal or truncated:break
    return dict(win=reward==1,loss=reward==-1,truncated=truncated,terminal=terminal,reward=reward,points=e.points(),decisions=decisions,engine_actions=e.actions,seconds=time.perf_counter()-start,inference_seconds=inference,end_to_end_decision_seconds=e2e,illegal_actions=0,infrastructure_failure=False,diagnostic=e.diagnostic(),failure_trace=e.trace if truncated else [])

def panel(net,schema,identifier,start,boards,opponents,modes,path,deadline,condition=None,seed=None,n=None):
    path=Path(path);spec=dict(policy=identifier,seed_start=start,boards=boards,opponents=opponents,modes=modes,condition=condition,training_seed=seed,n=n)
    saved=read(path) if path.exists() else dict(spec=spec,rows=[],complete_boards=[],complete=False,wall_seconds=0)
    assert saved['spec']==spec,'evaluation resume mismatch'
    done={(r['board_seed'],r['opponent'],r['mode'],r['seat']) for r in saved['rows']};archive=initial_archive();heldout=archive[-1]
    tick=time.perf_counter()
    try:
        for board in range(start,start+boards):
            for opponent in opponents:
                for mode in modes:
                    for seat in [0,1]:
                        key=(board,opponent,mode,seat)
                        if key in done:continue
                        e=ScalingEnv(board,seat,'neural' if opponent=='heldout' else opponent,board*17+seat,heldout if opponent=='heldout' else None)
                        row=play(net,schema,e,mode,board*23+seat*101+(['base','contender','heldout',*WEAK].index(opponent))*1009,deadline)
                        saved['rows'].append(row|dict(board_seed=board,opponent=opponent,mode=mode,seat=seat));done.add(key)
                        write(path,saved)
            if board not in saved['complete_boards']:saved['complete_boards'].append(board)
            write(path,saved)
        saved['complete']=True
    finally:
        saved['wall_seconds']+=time.perf_counter()-tick;write(path,saved)
    return saved

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--boards',type=int,required=True);p.add_argument('--opponents',default='base,contender,heldout');p.add_argument('--modes',default='sampled');p.add_argument('--deadline',type=float,required=True);p.add_argument('--device',default='cuda');p.add_argument('--condition');p.add_argument('--seed',type=int);p.add_argument('--n',type=int);a=p.parse_args()
    if a.checkpoint.startswith('historical:'):
        spec=next(s for s in initial_archive() if s['id']==a.checkpoint.split(':')[1]);net=FrozenPolicy(spec).net.to(a.device);schema=spec['schema'];identifier=spec['source_sha256']
    else:net,schema,_=get_policy(a.checkpoint,a.device);identifier=sha(a.checkpoint)
    panel(net,schema,identifier,a.start,a.boards,a.opponents.split(','),a.modes.split(','),a.output,a.deadline,a.condition,a.seed,a.n)

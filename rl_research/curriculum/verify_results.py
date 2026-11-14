"""Verify retrieved bytes, provenance, fixed-budget coverage, recovery, shutdown."""
import argparse,json,time,os,tarfile,subprocess
from pathlib import Path
import numpy as np
import torch
from curriculum_core import *

def main(root,out):
    remote=root/'retrieved';frozen=read(remote/'frozen.json');controller=read(root/'controller.json')
    assert controller['termination_verified'] and controller['total_allocated_upper_bound_seconds']<=14400
    assert controller['deadline']-controller['started']==14400
    assert read(remote/'external-recovery.json')['cuda_continuation_bitwise_equal']
    index=read(remote/'artifact-index.json');checks=[]
    load_corpus(root/'inputs/corpus.pt')
    for key,info in index.items():
        path=remote/key;assert sha(path)==info['sha256'] and path.stat().st_size==info['bytes']
        state=torch.load(path,map_location='cpu',weights_only=False);t=state['trainer']
        assert state['source_hashes']==source_hashes() and state['config']==CONFIG
        assert t.experiment_hashes=={n:sha(root/'source'/n) for n in t.experiment_hashes}
        assert all(torch.isfinite(p).all() for p in t.net.parameters())
        assert not t.rollout or len(t.rollout)<CONFIG['ppo']['rollout']
        for v in t.optimizer.state_dict()['state'].values():
            for value in v.values():
                if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        checks.append(dict(path=str(path),sha256=info['sha256'],n=t.n,arm=t.arm,seed=t.seed))
    # CPU recovery from external final checkpoints; bytes on disk remain unchanged.
    recoveries=[];primary=[];tasks=[];incomplete=[]
    for arm,seed in RUNS:
        folder=remote/f'{arm}-{seed}';result=read(folder/'result.json')
        if not result['complete']:incomplete.append(dict(arm=arm,seed=seed,**result));continue
        path=folder/f'{frozen["target"]}.pt';digest=sha(path);state=torch.load(path,map_location='cpu',weights_only=False);t=state['trainer'];t.device='cpu'
        assert t.n==frozen['target'] and sum(t.category_transitions.values())==t.n
        assert t.origin_sha256==frozen['starting_checkpoints'][str(seed)]['sha256']
        for _ in range(16):t.collect()
        t.update();assert all(torch.isfinite(p).all() for p in t.net.parameters()) and sha(path)==digest
        recoveries.append(dict(arm=arm,seed=seed,sha256=digest,cpu_collect_update=True))
        ep=remote/'evaluations'/f'{arm}-{seed}.json'
        if not ep.exists():continue
        e=read(ep);assert e['checkpoint_sha256']==digest
        rows=e['rows'];assert len(rows)%10==0
        keys={(r['board_seed'],r['opponent'],r['seat']) for r in rows};assert len(keys)==len(rows)
        expected={(b,o,s) for b in range(880000000000,880000000000+len(rows)//10) for o in OPPONENTS for s in [0,1]}
        assert keys==expected
        for r in rows:
            assert int(r['win'])+int(r['loss'])+int(r['truncated'])==1 and r['illegal_actions']==0
            assert r['reward']==(1 if r['win'] else -1 if r['loss'] else 0)
            if r['truncated']:assert r['diagnostic']['turn_limit_reached'] or r['diagnostic']['action_limit_reached']
        primary.append(dict(arm=arm,seed=seed,games=len(rows),complete=len(rows)==frozen['full_game_boards']*10))
        taskp=remote/'tasks'/f'{arm}-{seed}.json'
        if taskp.exists():tasks.append(dict(arm=arm,seed=seed,games=len(read(taskp)['rows']),complete=read(taskp).get('complete',False)))
    # Confirm identical initial model AND optimizer bytes/tensors in each pair.
    paired=[]
    for seed in SEEDS:
        aa=torch.load(remote/f'standard-{seed}/0.pt',map_location='cpu',weights_only=False)['trainer']
        bb=torch.load(remote/f'curriculum-{seed}/0.pt',map_location='cpu',weights_only=False)['trainer']
        assert all(torch.equal(x,y) for x,y in zip(aa.net.parameters(),bb.net.parameters()))
        for k,v in aa.optimizer.state_dict()['state'].items():
            for field,x in v.items():
                y=bb.optimizer.state_dict()['state'][k][field];assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
        assert not aa.rollout and not bb.rollout and aa.n==bb.n==0
        paired.append(dict(seed=seed,weights_equal=True,optimizer_equal=True,pending_rollout_empty=True))
    audit=dict(checkpoints=checks,recoveries=recoveries,initial_pair_checks=paired,primary_coverage=primary,task_coverage=tasks,incomplete_runs=incomplete,allocation_seconds=controller['total_allocated_upper_bound_seconds'],termination_verified=True,all_primary_complete=len(primary)==6 and all(x['complete'] for x in primary),all_tasks_complete=len(tasks)==6 and all(x['complete'] for x in tasks))
    write(out/'completion-verification.json',audit)
    # Self-contained six-policy archive with exact source, corpus, and reports.
    package=root/'final-policies-and-corpus.tar.gz'
    with tarfile.open(package,'w:gz',compresslevel=1) as tar:
        for arm,seed in RUNS:
            path=remote/f'{arm}-{seed}'/f'{frozen["target"]}.pt'
            if path.exists():tar.add(path,arcname=f'checkpoints/{arm}-{seed}.pt')
        for p in [root/'source.tar.gz',root/'inputs/corpus.pt',root/'corpus-manifest.json',root/'input-manifest.json',remote/'frozen.json']:
            tar.add(p,arcname=p.name)
    with tarfile.open(package) as tar:
        for arm,seed in RUNS:
            path=remote/f'{arm}-{seed}'/f'{frozen["target"]}.pt'
            if path.exists():
                import hashlib
                f=tar.extractfile(f'checkpoints/{arm}-{seed}.pt');assert hashlib.file_digest(f,'sha256').hexdigest()==sha(path)
    write(out/'checkpoint-package.json',dict(path=str(package),bytes=package.stat().st_size,sha256=sha(package),models=6-len(incomplete),includes='final policies, complete corpus, executed source and manifests'))
    print(json.dumps(dict(checkpoints=len(checks),recovered=len(recoveries),incomplete=len(incomplete),primary_games=sum(x['games'] for x in primary),task_games=sum(x['games'] for x in tasks),package_sha256=sha(package))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.output)

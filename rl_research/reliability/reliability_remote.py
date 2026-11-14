"""Five-hour reliability experiment. All core training methods are inherited unchanged."""
import argparse,os,platform,subprocess,time,traceback
from pathlib import Path
import torch
from train import Trainer,source_hashes
from env import TABLE,MAPPING_HASH
from reliability_common import *
from reliability_evaluate import evaluate

def status(out,phase,**kwargs):write(out/'status.json',dict(phase=phase,time=time.time(),**kwargs))
def wait_for(path,deadline):
    while not path.exists():
        if time.time()>=deadline:raise TimeoutError(f'waiting for {path.name}')
        time.sleep(2)
def checkpoint_path(out,historical,w,s,n):return (historical if s in OLD_SEEDS else out)/name(w,s)/f'{n}.pt'
def evaluation(out,historical,w,s,n,split,manifest,deadline):
    p=checkpoint_path(out,historical,w,s,n); dest=out/'evaluations'/split/name(w,s)/f'{n}.json'
    spec=manifest['splits'][split]; status(out,split,run=name(w,s),step=n,boards=spec['boards'])
    rows=evaluate(Trainer.load(p),spec['seed_start'],spec['boards'],deadline,dest,sha(p))
    if len(rows)!=6*spec['boards']:raise TimeoutError(f'incomplete {split} {name(w,s)} {n}')
    return rows

def main(out,historical,deadline):
    out.mkdir(parents=True,exist_ok=True);fixed_deadline(out/'budget.json',deadline)
    work_deadline=deadline-1200
    original=read(Path(__file__).with_name('input-manifest.json'))
    assert source_hashes()==original['core_hashes']
    for key,info in original['historical_checkpoint_checks'].items():assert sha(historical/key.removeprefix('retrieved/'))==info['sha256']
    if not (out/'preflight.json').exists():
        gpu=subprocess.check_output(['nvidia-smi'],text=True);assert 'T4' in gpu and torch.cuda.is_available()
        m=torch.nn.Linear(8,1).cuda();opt=torch.optim.Adam(m.parameters());before=m.weight.detach().clone()
        m(torch.ones(4,8,device='cuda')).square().mean().backward();opt.step();torch.cuda.synchronize();assert not torch.equal(before,m.weight)
        write(out/'preflight.json',dict(gpu=gpu,torch=torch.__version__,python=platform.python_version(),cpu_count=os.cpu_count(),cuda_update=True,actions=len(TABLE),mapping_hash=MAPPING_HASH,source_hashes=source_hashes()))
        subprocess.run(['python','-m','pytest','-q','test_contract.py','test_reliability.py'],check=True,stdout=(out/'tests.log').open('w'),stderr=subprocess.STDOUT)
        subprocess.Popen(['python','monitor.py',str(out/'telemetry.csv'),str(deadline)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.Popen(['python','cpu_monitor.py','--pid',str(os.getpid()),'--output',str(out/'cpu-telemetry.csv'),'--deadline',str(deadline)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if not (out/'frozen.json').exists():
        status(out,'calibration')
        t=DiagnosticTrainer('ppo',256,23,8192,'cuda')
        pilot=t.run(out/'pilot.pt',4096,min(work_deadline,time.time()+180));assert pilot['complete'] and t.updates>0
        evaluations=[]
        for n in [106496,434176]:
            p=checkpoint_path(out,historical,128,130363,n)
            start=time.time();rows=evaluate(Trainer.load(p),1800000000,2,min(work_deadline,time.time()+240),out/f'calibration-{n}-{time.time_ns()}.json',sha(p))
            assert len(rows)==12
            evaluations.append(dict(step=n,seconds=time.time()-start,boards=2,truncations=sum(r['truncated'] for r in rows)))
        write(out/'calibration.json',dict(pilot=pilot,evaluations=evaluations))
        status(out,'awaiting_external_recovery');wait_for(out/'recovery_verified',min(work_deadline,time.time()+600))
        # Conservatively budget all 40 nonzero validations and all 10 primary policies.
        train_estimate=6*434176/pilot['learner_decisions_s']*1.20
        board_seconds=max(e['seconds']/e['boards'] for e in evaluations)*1.25
        val,test=20,50
        while train_estimate+(40*val+10*test)*board_seconds+600>work_deadline-time.time() and val>4:
            val-=2;test-=5
        projected=train_estimate+(40*val+10*test)*board_seconds+600
        if projected>work_deadline-time.time():raise TimeoutError('full six-run primary workload cannot fit conservatively')
        frozen=original|dict(deadline=deadline,calibration=dict(training_seconds=train_estimate,evaluation_board_seconds=board_seconds,projected_primary_seconds=projected),split_reduction=val!=20)
        frozen['splits']['validation']['boards']=val;frozen['splits']['final']['boards']=test
        for split in ['validation','final']:
            spec=frozen['splits'][split]
            spec['board_seeds']=list(range(spec['seed_start'],spec['seed_start']+spec['boards']))
            spec['games_per_policy']=spec['boards']*6
        write(out/'frozen.json',frozen)
        status(out,'awaiting_manifest_copy');wait_for(out/'manifest_verified',min(work_deadline,time.time()+300))
    frozen=read(out/'frozen.json');assert frozen['deadline']==deadline
    # Expanded validation of historical nonzero checkpoints, never historical test selection.
    for s in OLD_SEEDS:
        for w in [128,256]:
            for n in STEPS:evaluation(out,historical,w,s,n,'validation',frozen,work_deadline-3600)
    for w,s in RUNS:
        folder=out/name(w,s);folder.mkdir(exist_ok=True)
        result=folder/'result.json'
        if result.exists():continue # completed OR explicitly incomplete runs are never silently retrained
        latest=sorted(folder.glob('*.pt'),key=lambda p:int(p.stem))
        t=Trainer.load(latest[-1]) if latest else DiagnosticTrainer('ppo',w,s,434176,'cuda')
        assert (t.width,t.seed,t.target)==(w,s,434176)
        if not latest:t.save(folder/'0.pt')
        for n in STEPS:
            meta=folder/f'{n}.json'
            if meta.exists() and read(meta)['stats']['complete']:
                stats=read(meta)['stats'];continue
            if n<t.n:continue
            status(out,'training',run=name(w,s),step=t.n,next_checkpoint=n)
            stop=min(work_deadline-2700,time.time()+max(0,original['run_training_limit_seconds']-t.elapsed))
            if stop<=time.time():
                write(result,dict(complete=False,n=t.n,reason='training or evaluation reserve reached'));break
            stats=t.run(folder/f'{n}.pt',n,stop)
            write(meta,dict(stats=stats,metrics=t.metrics,episodes=t.episodes,diagnostics=getattr(t,'truncation_diagnostics',[])))
            if not stats['complete']:
                write(result,dict(complete=False,n=t.n,reason='run limit reached',stats=stats));break
            # Evaluation preserves all sampling RNG streams; training elapsed excludes evaluation.
            spec=frozen['splits']['validation'];dest=out/'evaluations/validation'/name(w,s)/f'{n}.json'
            status(out,'validation',run=name(w,s),step=n,boards=spec['boards'])
            rows=evaluate(t,spec['seed_start'],spec['boards'],work_deadline-2700,dest,sha(folder/f'{n}.pt'))
            if len(rows)!=6*spec['boards']:raise TimeoutError('new validation incomplete')
        if not result.exists():write(result,dict(complete=t.n==434176,n=t.n,stats=stats))
    # On recovery fill missing validation files before selection (no retraining).
    completed=[]
    for w,s in ALL_RUNS:
        if s in NEW_SEEDS and not read(out/name(w,s)/'result.json')['complete']:continue
        completed.append((w,s))
        for n in STEPS:evaluation(out,historical,w,s,n,'validation',frozen,work_deadline-1800)
    if not (out/'selections.json').exists():
        selections=[]
        for w,s in completed:
            candidates=[dict(step=n,rows=read(out/'evaluations/validation'/name(w,s)/f'{n}.json')['rows']) for n in STEPS]
            selected=selection(candidates);n=selected['step']
            selections.append(dict(run=name(w,s),width=w,seed=s,step=n,checkpoint_sha256=sha(checkpoint_path(out,historical,w,s,n)),wins=sum(r['win'] for r in selected['rows']),truncations=sum(r['truncated'] for r in selected['rows']),games=len(selected['rows'])))
        write(out/'selections.json',dict(rule=frozen['selection_rule'],committed_at=time.time(),selections=selections))
    status(out,'awaiting_selection_copy');wait_for(out/'selection_verified',min(work_deadline,time.time()+300))
    assert (out/'selection_verified').read_text()==sha(out/'selections.json')
    # Untouched final seeds are first used only after the selection commit is verified externally.
    for w,s in completed:evaluation(out,historical,w,s,434176,'final',frozen,work_deadline)
    selections=read(out/'selections.json')['selections'];distinct=[x for x in selections if x['step']!=434176]
    planpath=out/'secondary-plan.json'
    if not planpath.exists():
        remaining=work_deadline-time.time()-180
        full=frozen['splits']['final']['boards'];board_seconds=read(out/'frozen.json')['calibration']['evaluation_board_seconds']
        boards=min(full,max(0,int(remaining/max(1,len(distinct))/board_seconds)))
        write(planpath,dict(boards=boards,policies=len(distinct),seed_start=frozen['splits']['final']['seed_start']))
    boards=read(planpath)['boards']
    if boards and distinct:
        secondary=dict(frozen);secondary['splits']=dict(frozen['splits']);secondary['splits']['secondary']=dict(seed_start=frozen['splits']['final']['seed_start'],boards=boards)
        # Keep every complete board block; report uses uniform minimum across secondary policies.
        for sel in distinct:
            if time.time()>=work_deadline:break
            try:evaluation(out,historical,sel['width'],sel['seed'],sel['step'],'secondary',secondary,work_deadline)
            except TimeoutError:break
    write(out/'artifact-index.json',{str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*.pt')})
    status(out,'complete',completed_runs=len(completed),new_completed=sum(s in NEW_SEEDS for _,s in completed),primary_boards=frozen['splits']['final']['boards'],secondary_boards_planned=boards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    try:main(a.out,a.historical,a.deadline)
    except BaseException:
        status(a.out,'failed',traceback=traceback.format_exc());raise

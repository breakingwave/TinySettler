"""Four-hour sequential continuation experiment; frozen budgets and no selection."""
import argparse,os,platform,subprocess,time,traceback,copy
from pathlib import Path
import torch
from curriculum_core import *
from curriculum_evaluate import evaluate_full,evaluate_tasks,exercise_manifest

def status(out,phase,**kw):write(out/'status.json',dict(phase=phase,time=time.time(),**kw))
def wait(path,deadline):
    while not path.exists():
        if time.time()>=deadline:raise TimeoutError('external checkpoint/manifest gate')
        time.sleep(2)
def load(path):
    t=Trainer.load(path)
    if isinstance(t,ContinuationTrainer):assert t.experiment_hashes==experiment_hashes()
    return t

def experiment_hashes():
    here=Path(__file__).parent
    return {p.name:sha(p) for p in here.glob('*.py') if p.name.startswith(('curriculum_','opponent_'))}

def main(out,historical,deadline):
    out.mkdir(exist_ok=True,parents=True);work_deadline=deadline-1200
    budget=out/'budget.json'
    if budget.exists():assert read(budget)['deadline']==deadline
    else:write(budget,dict(deadline=deadline,cap_seconds=14400,retrieval_reserve=1200,training_seconds=8100,evaluation_seconds=3000,recovery_seconds=600))
    manifest=read(Path(__file__).with_name('input-manifest.json'));assert source_hashes()==manifest['core_hashes']
    for seed,info in manifest['starting_checkpoints'].items():assert sha(historical/f'{seed}.pt')==info['sha256']
    corpus=load_corpus(historical/'corpus.pt');assert sha(historical/'corpus.pt')==manifest['corpus_sha256']
    if not (out/'preflight.json').exists():
        gpu=subprocess.check_output(['nvidia-smi'],text=True);assert 'T4' in gpu and torch.cuda.is_available()
        m=torch.nn.Linear(8,1).cuda();opt=torch.optim.Adam(m.parameters());before=m.weight.detach().clone()
        m(torch.ones(4,8,device='cuda')).square().mean().backward();opt.step();torch.cuda.synchronize();assert not torch.equal(before,m.weight)
        write(out/'preflight.json',dict(gpu=gpu,torch=torch.__version__,python=platform.python_version(),cpu_count=os.cpu_count(),cuda_update=True,source_hashes=source_hashes(),experiment_hashes=experiment_hashes(),pythonhashseed=os.environ.get('PYTHONHASHSEED')))
        subprocess.run(['python','-m','pytest','-q','test_contract.py','test_curriculum.py'],check=True,stdout=(out/'tests.log').open('w'),stderr=subprocess.STDOUT)
        subprocess.Popen(['python','monitor.py',str(out/'telemetry.csv'),str(deadline)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.Popen(['python','cpu_monitor.py','--pid',str(os.getpid()),'--output',str(out/'cpu-telemetry.csv'),'--deadline',str(deadline)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if not (out/'frozen.json').exists():
        status(out,'calibration');pilot_results=[]
        for arm in ['standard','curriculum']:
            t=ContinuationTrainer.clone(historical/f'{SEEDS[0]}.pt',arm,SEEDS[0],8192,'cuda');t.experiment_hashes=experiment_hashes()
            stats=t.run(out/f'pilot-{arm}.pt',8192,min(work_deadline,time.time()+420));assert stats['complete']
            pilot_results.append(dict(arm=arm,**stats,reset_counts=dict(t.reset_counts),category_transitions=dict(t.category_transitions),planning=dict(t.planning_totals)))
        t.save(out/'pilot.pt');write(out/'calibration-pilots.json',pilot_results)
        # Each timing uses fresh files, fresh boards, actual configured opponents.
        timing=[]
        for weak in [False,True]:
            t=ContinuationTrainer.clone(historical/f'{SEEDS[1]}.pt','standard',SEEDS[1],8192,'cuda')
            if weak:
                from train import Net
                torch.manual_seed(443);t.net=Net(len(t.env.obs()),128).cuda()
            path=out/f'calibration-evaluation-{weak}-{time.time_ns()}.json'
            tick=time.time();rows=evaluate_full(t,870000000000+int(weak)*100,1,min(work_deadline,time.time()+420),path,'calibration-'+str(weak))
            assert len(rows)==10;timing.append(dict(weak=weak,seconds=time.time()-tick,games=10,truncations=sum(r['truncated'] for r in rows)))
        # Held-out validation corpus only for timing; untouched test cases frozen later.
        t=ContinuationTrainer.clone(historical/f'{SEEDS[0]}.pt','standard',SEEDS[0],8192,'cuda')
        tick=time.time();taskrows=evaluate_tasks(t,corpus,exercise_manifest(corpus,1,'validation'),min(work_deadline,time.time()+300),out/f'calibration-tasks-{time.time_ns()}.json','calibration')
        task_seconds=time.time()-tick
        status(out,'awaiting_external_recovery');wait(out/'recovery_verified',min(work_deadline,time.time()+600))
        # Slowest measured arm, 25% throughput margin; equal rollout-multiple target.
        slow=min(x['learner_decisions_s'] for x in pilot_results)
        target=min(262144,int(slow*1350*.75)//4096*4096);assert target>=4096
        train_estimate=6*target/slow/0.75
        perboard=max(x['seconds'] for x in timing)*1.25
        cases=exercise_manifest(corpus,3,'test')
        diagnostic_estimate=task_seconds/max(1,len(taskrows))*len(cases)*2*6*1.4
        eval_budget=min(3000,work_deadline-time.time()-train_estimate-600)
        boards=min(20,int((eval_budget-diagnostic_estimate)/(6*perboard)))
        assert boards>=2,('insufficient evaluation budget',eval_budget,diagnostic_estimate,perboard)
        frozen=manifest|dict(deadline=deadline,target=target,full_game_boards=boards,full_game_seeds=list(range(880000000000,880000000000+boards)),task_cases=cases,pilot_results=pilot_results,evaluation_timing=timing,task_timing_seconds=task_seconds,projected_training_seconds=train_estimate,projected_evaluation_seconds=6*boards*perboard+diagnostic_estimate,run_limit_seconds=1350,calibration_slow_rate=slow,calibration_margin=.75)
        write(out/'frozen.json',frozen);status(out,'awaiting_manifest_copy');wait(out/'manifest_verified',min(work_deadline,time.time()+300))
    frozen=read(out/'frozen.json');assert frozen['deadline']==deadline
    training_deadline=work_deadline-max(3000,frozen['projected_evaluation_seconds'])
    for arm,seed in RUNS:
        folder=out/f'{arm}-{seed}';folder.mkdir(exist_ok=True);result=folder/'result.json'
        if result.exists():continue
        target=frozen['target'];latest=folder/'latest.pt'
        if latest.exists():t=load(latest)
        else:
            t=ContinuationTrainer.clone(historical/f'{seed}.pt',arm,seed,target,'cuda');t.experiment_hashes=experiment_hashes();t.save(folder/'0.pt')
        status(out,'training',run=f'{arm}-{seed}',n=t.n,target=target)
        stop=min(training_deadline,time.time()+max(0,1350-t.elapsed))
        # Two common intermediate positions for curves, no checkpoint selection.
        steps=sorted(set([target//3//4096*4096,2*target//3//4096*4096,target]))
        stats=None
        for n in steps:
            if n<=t.n:continue
            stats=t.run(latest,n,stop);t.save(folder/f'{t.n}.pt')
            write(folder/'training.json',dict(stats=stats,metrics=t.metrics,episodes=t.episodes,reset_counts=dict(t.reset_counts),category_transitions=dict(t.category_transitions),planning_totals=dict(t.planning_totals),truncation_diagnostics=t.truncation_diagnostics))
            if not stats['complete']:break
        if stats is None:stats=dict(n=t.n,complete=t.n==target,elapsed=t.elapsed)
        write(result,dict(complete=t.n==target,n=t.n,target=target,stats=stats,reason=None if t.n==target else 'training deadline or per-run cap'))
    complete=[]
    for arm,seed in RUNS:
        folder=out/f'{arm}-{seed}'
        if not read(folder/'result.json')['complete']:continue
        complete.append((arm,seed));path=folder/f'{frozen["target"]}.pt';t=load(path)
        status(out,'primary_evaluation',run=f'{arm}-{seed}',boards=frozen['full_game_boards'])
        try:evaluate_full(t,880000000000,frozen['full_game_boards'],work_deadline,out/'evaluations'/f'{arm}-{seed}.json',sha(path))
        except TimeoutError:break
    # Diagnostics stop first if primary evaluation is running long.
    for arm,seed in complete:
        if time.time()>=work_deadline:break
        path=out/f'{arm}-{seed}'/f'{frozen["target"]}.pt';t=load(path)
        status(out,'task_evaluation',run=f'{arm}-{seed}')
        try:evaluate_tasks(t,corpus,frozen['task_cases'],work_deadline,out/'tasks'/f'{arm}-{seed}.json',sha(path))
        except TimeoutError:break
    write(out/'artifact-index.json',{str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*.pt')})
    status(out,'complete',completed_training_runs=len(complete),target=frozen['target'],primary_boards=frozen['full_game_boards'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    try:main(a.out,a.historical,a.deadline)
    except BaseException:status(a.out,'failed',traceback=traceback.format_exc());raise

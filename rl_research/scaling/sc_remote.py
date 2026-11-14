"""Deterministic remote stages; no performance-based changes after scope freeze."""
from sc_plan import *
from sc_train import *
from sc_eval import panel,get_policy,MAIN,WEAK
from sc_checks import recovery_probe
import subprocess,traceback,signal

def status(out,phase,**kw):write(out/'status.json',dict(phase=phase,time=time.time(),**kw))
def gate(out,name,deadline):
    while not (out/name).exists():
        if time.time()>=deadline:raise TimeoutError('external verification gate')
        time.sleep(2)

def launch(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(command,stdout=log.open('a'),stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))

def wait_batch(out,processes,deadline):
    write(out/'active-pids.json',[p.pid for p in processes]);last=0;last_activity=time.time()
    while any(p.poll() is None for p in processes):
        if time.time()>deadline+120:
            for p in processes:
                if p.poll() is None:p.terminate()
            raise RuntimeError('worker exceeded hard stage deadline')
        if time.time()-last>=30:
            activity=[p.stat().st_mtime for p in out.rglob('progress.json')]
            for directory in ['validation','final','regression','context','pilot-evaluation']:
                activity.extend(p.stat().st_mtime for p in (out/directory).glob('*.json'))
            if activity:last_activity=max(last_activity,max(activity))
            if time.time()-last_activity>900:
                for p in processes:
                    if p.poll() is None:p.terminate()
                raise RuntimeError('RECOVERABLE: worker made no progress for 15 minutes')
            write(out/'heartbeat.json',dict(time=time.time(),pids=[p.pid for p in processes if p.poll() is None]));last=time.time()
        time.sleep(2)
    write(out/'active-pids.json',[])
    if any(p.returncode!=0 for p in processes):raise RuntimeError('INTEGRITY: worker failure; inspect bounded worker logs')

def train_command(c,s,folder,target,deadline,pilot=False):
    return ['python','sc_train.py','--condition',c,'--seed',str(s),'--folder',str(folder),'--target',str(target),'--deadline',str(deadline)]+(['--pilot'] if pilot else [])

def calibration(out,deadline):
    repair_allowance=read(HERE/'input-manifest.json').get('preflight_repair_reserve_seconds',0)
    assert repair_allowance in (0,1800)
    manifest=read(HERE/'input-manifest.json')
    limit=min(manifest.get('authorized_preflight_deadline',deadline-50400+2700+repair_allowance),deadline-10800);pilots={}
    for c in manifest.get('calibration_order',list(CONDITIONS)):
        pilots[c]={}
        for mode,workers in [('single',1),('dual',2)]:
            folder=out/'pilots'/f'{c}-{mode}'
            summary=folder/'timing.json'
            if summary.exists():pilots[c][mode]=read(summary);continue
            if time.time()>=limit:raise TimeoutError('preflight pilot allowance')
            tick=time.perf_counter();children=[]
            for j in range(workers):
                f=folder/str(j);children.append(launch(train_command(c,SEEDS[j],f,8192,limit,pilot=True),f/'worker.log'))
            wait_batch(out,children,limit);elapsed=time.perf_counter()-tick
            ts=[read(folder/str(j)/'training.json') for j in range(workers)]
            assert all(t['n']==8192 for t in ts),'incomplete pilot'
            # Independent processes perform exact saved-RNG collection/update recovery.
            probes=[]
            for j in range(workers):
                f=folder/str(j);probes.append(launch(['python','sc_checks.py','--probe',str(f/'8192.pt'),'--out',str(f/'recovery.json'),'--device','cuda'],f/'recovery.log'))
            wait_batch(out,probes,limit)
            value=dict(seconds=elapsed,aggregate_rate=workers*8192/elapsed,workers=workers,teacher_seconds=max(t['teacher_seconds'] for t in ts),recovery_passed=all(read(folder/str(j)/'recovery.json')['bitwise_weights_rng_observation_equal'] for j in range(workers)))
            write(summary,value);pilots[c][mode]=value
        write(out/'pilot-timings.json',pilots)
    timing={}
    for ki,kind in enumerate(['mlp128','mlp512','graph']):
        c={'mlp128':'A','mlp512':'C','graph':'E'}[kind];times=[]
        # Fresh untrained policy and fresh pilot policy, with distinct boards/files.
        for weak in [True,False]:
            net=model(kind,SEEDS[0]).cuda().eval() if weak else get_policy(out/'pilots'/f'{c}-single'/'0/8192.pt')[0]
            tick=time.perf_counter();p=out/'pilot-evaluation'/f'{kind}-{weak}-{time.time_ns()}.json'
            result=panel(net,SCHEMA,'timing-only',NAMESPACES['pilot']+90000000+ki*100+int(weak)*10,1,MAIN,['sampled','greedy'],p,limit)
            assert len(result['rows'])==12;times.append((time.perf_counter()-tick)/12)
        timing[kind]=dict(seconds_per_game=max(times),fresh_weak_and_pilot_seconds_per_game=times)
    write(out/'evaluation-timing.json',timing)
    # Force a frozen own-policy opponent into the copy/upload/resume checkpoint.
    t=ScalingTrainer.load(out/'pilots/F-single/0/8192.pt','cuda');spec=t.archive[-1]
    t.env=ScalingEnv(NAMESPACES['pilot']+99000000,0,'neural',policy=spec);t.opponent_id=spec['id'];t.opponent_step=spec['step'];t.save(out/'external-probe.pt')
    # Same fresh-process probe on both original and uploaded copy.
    p=launch(['python','sc_checks.py','--probe',str(out/'external-probe.pt'),'--out',str(out/'external-probe-expected.json'),'--device','cuda'],out/'external-probe.log');wait_batch(out,[p],limit)
    status(out,'await_external_recovery');gate(out,'external_recovery_verified',min(limit,time.time()+600))
    return pilots,timing

def evaluate(out,path,folder,start,boards,opponents,modes,deadline,c=None,s=None,n=None):
    if folder.exists() and read(folder).get('complete'):return 0.
    command=['python','sc_eval.py','--checkpoint',str(path),'--output',str(folder),'--start',str(start),'--boards',str(boards),'--opponents',','.join(opponents),'--modes',','.join(modes),'--deadline',str(deadline)]
    if c is not None:command+=['--condition',c,'--seed',str(s),'--n',str(n)]
    tick=time.perf_counter();p=launch(command,folder.with_suffix('.log'));wait_batch(out,[p],deadline);return time.perf_counter()-tick

def choose_candidate(out,frozen,common):
    options=[]
    for c,s in frozen['runs']:
        path=out/'validation'/f'{c}-{s}-{common}-both.json'
        if not path.exists() or not read(path)['complete']:continue
        rows=read(path)['rows']
        for mode in ['sampled','greedy']:
            rr=[r for r in rows if r['opponent'] in ['base','contender'] and r['mode']==mode]
            options.append(dict(condition=c,seed=s,step=common,mode=mode,wins=sum(r['win'] for r in rr),games=len(rr),truncations=sum(r['truncated'] for r in rr),parameters=read(out/'runs'/f'{c}-{s}'/'training.json')['parameters']))
    assert options,'no complete candidate validation panel'
    chosen=max(options,key=lambda x:(x['wins']/x['games'],-x['truncations'],-x['parameters'],-list(CONDITIONS).index(x['condition']),-SEEDS.index(x['seed']),x['mode']=='sampled'))
    checkpoint=out/'runs'/f'{chosen["condition"]}-{chosen["seed"]}'/f'{common}.pt'
    return dict(selected=chosen,all_candidates=options,checkpoint_sha256=sha(checkpoint),rule=frozen['selection_rule'],time=time.time())

def main(out,deadline):
    out.mkdir(parents=True,exist_ok=True);assert time.time()<deadline-RESERVE
    manifest=read(HERE/'input-manifest.json');assert manifest['source_hashes']==hashes(),'INTEGRITY: executed source'
    assert sha(ROOT/'archive.pt')==manifest['archive_sha256'] and sha(ROOT/'teacher.pt')==manifest['teacher_sha256']
    statepath=out/'experiment-state.json';state=read(statepath) if statepath.exists() else dict(deadline=deadline,training_seconds=0.,validation_seconds=0.,final_seconds=0.)
    assert state['deadline']==deadline
    if not (out/'preflight.json').exists():
        gpu=subprocess.check_output(['nvidia-smi'],text=True);assert 'T4' in gpu and torch.cuda.is_available()
        m=torch.nn.Linear(3,1).cuda();before=m.weight.detach().clone();opt=torch.optim.Adam(m.parameters());m(torch.ones(2,3,device='cuda')).square().mean().backward();opt.step();torch.cuda.synchronize();assert not torch.equal(before,m.weight)
        rng_before=torch.cuda.get_rng_state_all()
        FrozenPolicy(initial_archive()[0]);model('graph',SEEDS[0])
        assert all(torch.equal(a,b) for a,b in zip(rng_before,torch.cuda.get_rng_state_all())),'INTEGRITY: CPU model initialization changed CUDA RNG'
        write(out/'preflight.json',dict(time=time.time(),gpu=gpu,torch=torch.__version__,cpu_count=os.cpu_count(),cuda_update=True,source_hashes=hashes()))
    monitor=launch(['python','sc_monitor.py',str(out),str(deadline)],out/'monitor.log')
    if not (out/'frozen.json').exists():
        status(out,'calibration');pilots,timing=calibration(out,deadline)
        frozen=manifest|freeze(pilots,timing,time.time(),deadline,os.cpu_count());write(out/'frozen.json',frozen)
        status(out,'await_frozen_copy');gate(out,'frozen_verified',min(manifest.get('authorized_preflight_deadline',deadline-50400+2700+manifest.get('preflight_repair_reserve_seconds',0)),time.time()+600))
    frozen=read(out/'frozen.json');assert frozen['deadline']==deadline
    for p in (out/'pilots').rglob('*.pt'):p.unlink()
    work_deadline=deadline-RESERVE
    if (out/'pause_requested').exists():status(out,'paused');return
    # Initialize every retained policy, then evaluate N=0 before any PPO chunk.
    for c,s in frozen['runs']:
        folder=out/'runs'/f'{c}-{s}'
        if not (folder/'0.pt').exists():
            status(out,'initializing',run=f'{c}-{s}')
            tick=time.perf_counter();p=launch(train_command(c,s,folder,0,work_deadline),folder/'worker.log');wait_batch(out,[p],work_deadline);state['training_seconds']+=time.perf_counter()-tick;write(statepath,state)
        vp=out/'validation'/f'{c}-{s}-0-sampled.json'
        state['validation_seconds']+=evaluate(out,folder/'0.pt',vp,NAMESPACES['validation'],frozen['validation_intermediate_boards'],MAIN,['sampled'],min(work_deadline,time.time()+max(0,2700-state['validation_seconds'])),c,s,0);write(statepath,state)
    interrupted=False
    for ri,step in enumerate(frozen['milestones']):
        tasks=ordered_runs(frozen['conditions'],ri)
        for offset in range(0,len(tasks),frozen['workers']):
            batch=tasks[offset:offset+frozen['workers']];children=[];tick=time.perf_counter()
            training_deadline=min(work_deadline-5400-max(0,2700-state['validation_seconds']),time.time()+max(0,36000-state['training_seconds']))
            for c,s in batch:
                folder=out/'runs'/f'{c}-{s}'
                if (folder/f'{step}.pt').exists():continue
                children.append(launch(train_command(c,s,folder,step,training_deadline),folder/'worker.log'))
            if children:
                status(out,'training',milestone=step,batch=[f'{c}-{s}' for c,s in batch]);wait_batch(out,children,training_deadline)
                state['training_seconds']+=time.perf_counter()-tick;write(statepath,state)
            if (out/'pause_requested').exists():status(out,'paused');return
            if any(not (out/'runs'/f'{c}-{s}'/f'{step}.pt').exists() for c,s in batch):interrupted=True;break
            for c,s in batch:
                modes=['sampled','greedy'] if step==frozen['target'] else ['sampled'];boards=frozen['validation_final_boards'] if step==frozen['target'] else frozen['validation_intermediate_boards'];suffix='both' if len(modes)==2 else 'sampled'
                status(out,'validation',run=f'{c}-{s}',milestone=step)
                state['validation_seconds']+=evaluate(out,out/'runs'/f'{c}-{s}'/f'{step}.pt',out/'validation'/f'{c}-{s}-{step}-{suffix}.json',NAMESPACES['validation'],boards,MAIN,modes,min(work_deadline,time.time()+max(0,2700-state['validation_seconds'])),c,s,step);write(statepath,state)
        if interrupted:break
    minimum=min(read(out/'runs'/f'{c}-{s}'/'progress.json')['n'] for c,s in frozen['runs'])
    common=max([0]+[x for x in frozen['milestones'] if x<=minimum]);write(out/'common-step.json',dict(step=common,target=frozen['target'],incomplete=common<frozen['target']))
    if common==0:raise RuntimeError('no common trained checkpoint before deadline')
    if common!=frozen['target']:
        for c,s in frozen['runs']:
            state['validation_seconds']+=evaluate(out,out/'runs'/f'{c}-{s}'/f'{common}.pt',out/'validation'/f'{c}-{s}-{common}-both.json',NAMESPACES['validation'],frozen['validation_final_boards'],MAIN,['sampled','greedy'],min(work_deadline,time.time()+max(0,2700-state['validation_seconds'])),c,s,common);write(statepath,state)
    if not (out/'selection.json').exists():write(out/'selection.json',choose_candidate(out,frozen,common))
    status(out,'await_selection_copy');gate(out,'selection_verified',min(work_deadline,time.time()+600))
    # Final data are not touched until a selection manifest has an external verified copy.
    selected=read(out/'selection.json')['selected'];final_deadline=min(work_deadline,time.time()+max(0,5400-state['final_seconds']))
    tasks=[]
    for c,s in frozen['runs']:tasks.append((out/'runs'/f'{c}-{s}'/f'{common}.pt',out/'final'/f'{c}-{s}-sampled.json',NAMESPACES['final'],frozen['final_boards'],MAIN,['sampled'],c,s))
    c,s=selected['condition'],selected['seed'];tasks.append((out/'runs'/f'{c}-{s}'/f'{common}.pt',out/'confirmation.json',NAMESPACES['final']+frozen['final_boards'],frozen['confirmation_boards']-frozen['final_boards'],['base','contender'],[selected['mode']],c,s))
    for c,s in frozen['runs']:tasks.append((out/'runs'/f'{c}-{s}'/f'{common}.pt',out/'final'/f'{c}-{s}-greedy.json',NAMESPACES['final'],frozen['final_boards'],MAIN,['greedy'],c,s))
    for c,s in frozen['runs']:tasks.append((out/'runs'/f'{c}-{s}'/f'{common}.pt',out/'regression'/f'{c}-{s}.json',NAMESPACES['regression'],frozen['weak_boards'],WEAK,['sampled'],c,s))
    for s in [104729,130363,155921]:tasks.append((f'historical:standard-{s}',out/'context'/f'{s}.json',NAMESPACES['final'],frozen['final_boards'],MAIN,['sampled','greedy'],None,None))
    for checkpoint,path,start,boards,opponents,modes,c,s in tasks:
        if (out/'pause_requested').exists():status(out,'paused');return
        if time.time()>=final_deadline:break
        status(out,'final_evaluation',panel=path.name)
        try:state['final_seconds']+=evaluate(out,checkpoint,path,start,boards,opponents,modes,final_deadline,c,s,common);write(statepath,state)
        except (TimeoutError,RuntimeError):
            if time.time()<final_deadline-120:raise
            break
    # Hash index is written only after all checkpoint files are closed/atomic.
    write(out/'checkpoint-index.json',{str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*.pt')})
    status(out,'complete',common_step=common,target=frozen['target'],runs=len(frozen['runs']),incomplete_training=common<frozen['target'])
    monitor.terminate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    try:main(a.out,a.deadline)
    except BaseException as exc:status(a.out,'recoverable_failure' if 'RECOVERABLE:' in str(exc) else 'fatal',traceback=traceback.format_exc());raise

"""Sequential sweep. Launched only after local contract checks pass."""
import argparse,hashlib,json,os,platform,subprocess,time,traceback,resource
from pathlib import Path
import torch
from train import Trainer, CONFIG
from evaluate import evaluate,summarize
from env import MAPPING_HASH,TABLE
from worker_probe import probe

def write(path,obj):
    p=Path(path); tmp=p.with_suffix('.json.tmp'); tmp.write_text(json.dumps(obj,indent=2)); os.replace(tmp,p)
def telemetry():
    return dict(cpu_count=os.cpu_count(),load=os.getloadavg(),process_cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader'],text=True).strip())
def resume_training(folder,algo,width,seed,target,device,deadline):
    saved=list(folder.glob('*.pt'))
    t=Trainer.load(max(saved,key=lambda p:int(p.stem))) if saved else Trainer(algo,width,seed,target,device)
    assert (t.algo,t.width,t.seed,t.target)==(algo,width,seed,target)
    runmeta=folder/'run-budget.json'
    if runmeta.exists():
        stop=min(json.loads(runmeta.read_text())['deadline'],deadline-5400)
    else:
        validation_seconds=sum(sum(r['seconds'] for r in json.loads(p.read_text()).get('rows',[])) for p in folder.glob('*.json'))
        stop=min(time.time()+max(0,2700-t.elapsed-validation_seconds),deadline-5400)
        write(runmeta,dict(deadline=stop,prior_training_seconds=t.elapsed,prior_validation_seconds=validation_seconds))
    return t,stop

def main(out,deadline):
    out.mkdir(parents=True,exist_ok=True)
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    assert 'T4' in gpu and torch.cuda.is_available()
    # Actual CUDA backward/update verified independently of the simulator.
    m=torch.nn.Linear(8,1).cuda(); opt=torch.optim.Adam(m.parameters()); before=m.weight.detach().clone()
    m(torch.ones(4,8,device='cuda')).square().mean().backward(); opt.step(); torch.cuda.synchronize()
    assert not torch.equal(before,m.weight)
    write(out/'preflight.json',dict(gpu=gpu,torch=torch.__version__,python=platform.python_version(),cuda_update=True,telemetry=telemetry(),mapping_hash=MAPPING_HASH,actions=len(TABLE)))
    subprocess.run(['python','-m','pytest','-q','test_contract.py'],check=True,stdout=(out/'tests.log').open('w'),stderr=subprocess.STDOUT)
    subprocess.Popen(['python','monitor.py',str(out/'telemetry.csv'),str(deadline)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    workers=probe(); write(out/'workers.json',workers)
    if (out/'frozen.json').exists():
        frozen=json.loads((out/'frozen.json').read_text()); pilots=frozen['pilots']; target=frozen['target']
    else:
        pilots=[]
        for algo,width in CONFIG['configurations']:
            t=Trainer(algo,width,17,8192,'cuda')
            stats=t.run(out/f'pilot-{algo}-{width}.pt',4096,min(deadline-5400,time.time()+300))
            assert stats['complete'] and t.updates>0
            resumed=Trainer.load(out/f'pilot-{algo}-{width}.pt'); resumed.collect()
            assert resumed.n==4097
            rows=evaluate(t,1900000000,1,time.time()+90)
            assert len(rows)==6
            if algo=='ddqn':
                t.run(out/'pilot-ddqn-full-buffer.pt',20000,min(deadline-5400,time.time()+300))
                assert len(t.replay)==20000
                start_n,start_elapsed=t.n,t.elapsed
                steady=t.run(out/'pilot-ddqn-steady.pt',24096,min(deadline-5400,time.time()+300))
                assert steady['complete']
                stats['steady_decisions_s']=(t.n-start_n)/(t.elapsed-start_elapsed)
                stats['steady_update_total_seconds']=t.update_seconds
            stats.update(algo=algo,width=width,evaluation_seconds=sum(r['seconds'] for r in rows),telemetry=telemetry())
            pilots.append(stats); write(out/'pilots.json',pilots)
        # Require controller to verify an external checkpoint download/reload before sweep.
        write(out/'status.json',dict(phase='awaiting_external_recovery',pilots=pilots))
        while not (out/'recovery_verified').exists():
            if time.time()>deadline-5400: raise TimeoutError('external checkpoint verification not received')
            time.sleep(2)
        slow=min(p.get('steady_decisions_s',p['learner_decisions_s']) for p in pilots)
        training_seconds=min(1800,max(300,(deadline-time.time()-5400)/6*0.70))
        target=max(4096,min(500000,int(slow*training_seconds)//4096*4096))
        write(out/'frozen.json',dict(target=target,workers=1,worker_measurements=workers,calibration_seconds=training_seconds,pilots=pilots,config=CONFIG,deadline=deadline))
    selections=json.loads((out/'selections.json').read_text()) if (out/'selections.json').exists() else []
    for algo,width in CONFIG['configurations']:
        for seed in CONFIG['training_seeds']:
            if time.time()>deadline-5400: raise TimeoutError('training reserve reached')
            name=f'{algo}-{width}-{seed}'; folder=out/name; folder.mkdir(exist_ok=True)
            if any(sel['run']==name for sel in selections): continue
            t,stop=resume_training(folder,algo,width,seed,target,'cuda',deadline)
            checkpoints=sorted(set([0,target//4//4096*4096,target//2//4096*4096,3*target//4//4096*4096,target]))
            candidates=[]
            for n in checkpoints:
                path=folder/f'{n}.pt'
                prior=folder/f'{n}.json'
                if prior.exists():
                    old=json.loads(prior.read_text())
                    if old.get('stats',{}).get('complete') and old.get('rows'):
                        candidates.append((sum(r['reward'] for r in old['rows'])/len(old['rows']),-n,str(path)))
                        continue
                if n<t.n: continue
                stats=t.run(path,n,stop)
                rows=evaluate(t,CONFIG['validation_seed_start'],2,min(time.time()+120,stop))
                score=sum(r['reward'] for r in rows)/len(rows) if rows else -float('inf')
                write(folder/f'{n}.json',dict(stats=stats,rows=rows,summary=summarize(rows),metrics=t.metrics,episodes=t.episodes,telemetry=telemetry()))
                if rows: candidates.append((score,-n,str(path)))
                write(out/'status.json',dict(phase='training',run=name,n=t.n,target=target,stats=stats))
                if not stats['complete']: break
            if not candidates: raise RuntimeError('no complete validation block')
            selected=max(candidates)[2]; selections.append(dict(run=name,path=selected,complete=t.n>=target,n=t.n))
            write(out/'selections.json',selections)
    # Selection file is committed before any final-test board is touched.
    remaining=deadline-time.time()-1800
    pilot_eval=max(p['evaluation_seconds'] for p in pilots)
    boards=max(1,min(100,int(remaining/(len(selections)*pilot_eval*1.5))))
    write(out/'final_plan.json',dict(boards=boards,seed_start=CONFIG['final_seed_start']))
    final_results=[]
    for index,sel in enumerate(selections):
        existing=out/sel['run']/'final.json'
        if existing.exists():
            old=json.loads(existing.read_text()); final_results.append((sel,old['rows'])); continue
        eval_deadline=min(deadline-600,time.time()+max(1,(deadline-600-time.time())/(len(selections)-index)))
        t=Trainer.load(sel['path']); rows=evaluate(t,CONFIG['final_seed_start'],boards,eval_deadline)
        final_results.append((sel,rows))
        write(out/sel['run']/'final.json',dict(selection=sel,rows=rows,summary=summarize(rows)))
    common_boards=min(len(rows)//6 for _,rows in final_results)
    for sel,rows in final_results:
        write(out/sel['run']/'final-raw.json',dict(rows=rows))
        rows=rows[:common_boards*6]
        write(out/sel['run']/'final.json',dict(selection=sel,boards=common_boards,rows=rows,summary=summarize(rows)))
    write(out/'status.json',dict(phase='complete',final_boards_actual=common_boards,selections=selections,final_boards_target=boards))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--deadline',type=float,required=True); a=p.parse_args()
    try: main(a.out,a.deadline)
    except BaseException:
        a.out.mkdir(parents=True,exist_ok=True)
        write(a.out/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise

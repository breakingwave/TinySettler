"""Generate small, reproducible summaries and SVG curves from retrieved artifacts."""
import argparse,csv,hashlib,json,statistics,shutil
import numpy as np
from pathlib import Path
from datetime import datetime,timezone

def cluster_interval(values,rng):
    values=np.asarray(values,dtype=float); n=len(values)
    if not n: return [None,None], 'no completed board clusters'
    # At a boundary percentile bootstrap degenerates. Bound the probability
    # of any success/failure in a board cluster using exact binomial tails;
    # this conservatively bounds the per-game average within each cluster.
    if np.all(values==0): return [0.,float(1-.025**(1/n))], 'exact boundary bound on board-cluster any-win probability'
    if np.all(values==1): return [float(.025**(1/n)),1.], 'exact boundary bound on board-cluster any-loss probability'
    return np.quantile(rng.choice(values,(5000,n)).mean(1),[.025,.975]).tolist(), 'percentile bootstrap over paired-board clusters'

def main(a):
    src=a.artifacts/'retrieved'; a.output.mkdir(parents=True,exist_ok=True)
    controller=json.loads((a.artifacts/'controller.json').read_text())
    def digest(p):
        with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
    files={str(p.relative_to(a.artifacts)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in a.artifacts.rglob('*') if p.is_file() and p.name not in ('snapshot.tar','controller.log')}
    (a.output/'artifact-manifest.json').write_text(json.dumps(dict(location=str(a.artifacts),files=files,controller=controller,report_source_sha256=digest(Path(__file__)),created_at=datetime.now(timezone.utc).isoformat()),indent=2))
    frozen=json.loads((src/'frozen.json').read_text()) if (src/'frozen.json').exists() else {}
    records=[]; curves=[]; opponent_rows=[]; final_rows=[]
    for p in src.rglob('*.json'):
        if p.stat().st_size<=1000000:
            dest=a.output/'raw'/p.relative_to(src); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    for folder in sorted(src.iterdir()) if src.exists() else []:
        if not folder.is_dir(): continue
        final=folder/'final.json'
        evaluations=[]
        for p in folder.glob('*.json'):
            obj=json.loads(p.read_text())
            if 'stats' in obj:
                evaluations.append((int(p.stem),obj))
                curves.append(dict(run=folder.name,n=int(p.stem),elapsed_seconds=obj['stats']['elapsed'],win_rate=sum(r['win'] for r in obj['rows'])/max(1,len(obj['rows'])),terminal_frequency=obj['stats']['terminal_rewards']/max(1,obj['stats']['n'])))
        if not evaluations: continue
        n,last=max(evaluations)
        f=json.loads(final.read_text()) if final.exists() else {}
        rows=f.get('rows',[])
        final_rows.extend(dict(run=folder.name,**r) for r in rows)
        for opponent,summary in f.get('summary',{}).items():
            g=[r for r in rows if r['opponent']==opponent]; boards=sorted(set(r['board_seed'] for r in g))
            values=[np.mean([r['win'] for r in g if r['board_seed']==b]) for b in boards]
            ci,method=cluster_interval(values,np.random.default_rng(720))
            opponent_rows.append(dict(run=folder.name,opponent=opponent,**summary,reported_cluster_ci95=ci,interval_method=method))
        records.append(dict(run=folder.name,transitions=last['stats']['n'],complete=f.get('selection',{}).get('complete',last['stats']['n']>=frozen.get('target',float('inf'))),selected_transitions=Path(f.get('selection',{}).get('path','unknown')).stem,train_seconds=round(last['stats']['elapsed'],2),decisions_s=round(last['stats']['learner_decisions_s'],2),training_episodes=len(last.get('episodes',[])),training_wins=sum(e['reward']==1 for e in last.get('episodes',[])),training_losses=sum(e['reward']==-1 for e in last.get('episodes',[])),training_truncations=sum(e['truncated'] for e in last.get('episodes',[])),terminal_training_episodes=last['stats']['terminal_rewards'],terminal_reward_frequency=last['stats']['terminal_rewards']/max(1,last['stats']['n']),test_games=len(rows),wins=sum(r['win'] for r in rows),losses=sum(r['loss'] for r in rows),truncations=sum(r['truncated'] for r in rows),mean_points=round(statistics.mean(r['points'] for r in rows),2) if rows else None))
    for name,data in [('comparison',records),('learning-curves',curves),('opponent-comparison',opponent_rows),('final-games',final_rows)]:
        (a.output/(name+'.json')).write_text(json.dumps(data,indent=2))
        if data:
            with (a.output/(name+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=data[0]); w.writeheader(); w.writerows(data)
    training_curves=[]
    for folder in sorted(src.iterdir()):
        if not folder.is_dir(): continue
        points=[p for p in folder.glob('*.json') if p.stem.isdigit()]
        if not points: continue
        obj=json.loads(max(points,key=lambda p:int(p.stem)).read_text())
        episodes=obj.get('episodes',[])
        for i in range(49,len(episodes),25):
            window=episodes[i-49:i+1]
            training_curves.append(dict(run=folder.name,n=episodes[i]['n'],win_rate=sum(e['reward']==1 for e in window)/50,truncation_rate=sum(e['truncated'] for e in window)/50))
    (a.output/'training-curves.json').write_text(json.dumps(training_curves,indent=2))
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for name in sorted(set(r['run'] for r in curves)):
            group=sorted([r for r in curves if r['run']==name],key=lambda r:r['n'])
            axes[0].plot([r['n'] for r in group],[r['win_rate'] for r in group],marker='o',label=name)
            axes[1].plot([r['n'] for r in group if r['n']],[r['terminal_frequency'] for r in group if r['n']],marker='o',label=name)
        axes[0].set(ylabel='Validation wins / all games',ylim=(0,1),title='Two board clusters per opponent')
        axes[1].set(ylabel='Terminal rewards / learner decisions',title='Reward sparsity (cumulative)')
        for ax in axes: ax.set_xlabel('Learner transitions'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.savefig(a.output/'learning-curves.svg'); fig.savefig(a.output/'learning-curves.png',dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for name in sorted(set(r['run'] for r in curves)):
            group=sorted([r for r in curves if r['run']==name],key=lambda r:r['n'])
            ax.plot([r['elapsed_seconds']/60 for r in group],[r['win_rate'] for r in group],marker='o',label=name)
        ax.set(xlabel='Training minutes (rollout + updates; excludes evaluation)',ylabel='Validation wins / all games',ylim=(0,1),title='Wall-clock learning efficiency'); ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.savefig(a.output/'wall-clock-curves.svg'); fig.savefig(a.output/'wall-clock-curves.png',dpi=180); plt.close(fig)
        if training_curves:
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for name in sorted(set(r['run'] for r in training_curves)):
                group=[r for r in training_curves if r['run']==name]
                ax.plot([r['n'] for r in group],[r['win_rate'] for r in group],label=name)
            ax.set(xlabel='Learner transitions',ylabel='Wins / last 50 training episodes',ylim=(0,1),title='Training mixture; exploratory policies'); ax.legend(fontsize=7); ax.grid(alpha=.2)
            fig.savefig(a.output/'training-curves.svg'); fig.savefig(a.output/'training-curves.png',dpi=180); plt.close(fig)
    # Board clusters include both seats and all training seeds, preserving paired
    # tests across configurations. This CI measures board uncertainty, not the
    # population variance of training seeds (only two independent seeds exist).
    aggregated=[]; rng=np.random.default_rng(719)
    for model in sorted(set(r['run'].rsplit('-',1)[0] for r in final_rows)):
        group=[r for r in final_rows if r['run'].rsplit('-',1)[0]==model]
        for opponent in ['random','weighted','heuristic','all']:
            g=[r for r in group if opponent=='all' or r['opponent']==opponent]
            seeds=sorted(set(r['board_seed'] for r in g))
            pairs=np.array([np.mean([r['win'] for r in g if r['board_seed']==seed]) for seed in seeds])
            ci,method=cluster_interval(pairs,rng)
            aggregated.append(dict(model=model,opponent=opponent,games=len(g),wins=sum(r['win'] for r in g),losses=sum(r['loss'] for r in g),truncations=sum(r['truncated'] for r in g),win_rate=float(pairs.mean()) if len(pairs) else None,board_cluster_ci95=ci,interval_method=method))
    (a.output/'aggregate.json').write_text(json.dumps(aggregated,indent=2))
    paired=[]
    models=sorted(set(r['run'].rsplit('-',1)[0] for r in final_rows))
    for i,left in enumerate(models):
        for right in models[i+1:]:
            values={}
            for model in [left,right]:
                g=[r for r in final_rows if r['run'].rsplit('-',1)[0]==model]
                values[model]={b:np.mean([r['win'] for r in g if r['board_seed']==b]) for b in set(r['board_seed'] for r in g)}
            shared=sorted(set(values[left])&set(values[right]))
            delta=np.array([values[left][b]-values[right][b] for b in shared])
            if len(delta):
                ci=np.quantile(rng.choice(delta,(5000,len(delta))).mean(1),[.025,.975]).tolist()
                method='paired percentile bootstrap over board clusters, both seats and both training seeds retained'
                if np.ptp(delta)==0:
                    radius=(2*np.log(40)/len(delta))**.5
                    ci=[max(-1,float(delta.mean()-radius)),min(1,float(delta.mean()+radius))]
                    method='conservative Hoeffding bound for degenerate paired differences'
                paired.append(dict(model_a=left,model_b=right,win_rate_difference=float(delta.mean()),ci95=ci,board_clusters=len(delta),method=method))
    (a.output/'paired-comparisons.json').write_text(json.dumps(paired,indent=2))

    utilization={}
    for name in ['telemetry.csv','cpu-utilization.csv']:
        path=src/name
        if not path.exists(): continue
        shutil.copy2(path,a.output/name)
        with path.open() as f: samples=list(csv.DictReader(f))
        if name=='telemetry.csv':
            good=[]
            for r in samples:
                try:
                    fields=r['gpu_csv'].split(','); good.append((float(r['unix_time']),float(fields[1]),float(fields[3]),float(r['load1'])))
                except (ValueError,KeyError,IndexError,AttributeError): pass
            if good:
                v=np.asarray(good); utilization['gpu_and_load']=dict(samples=len(good),start_unix=v[0,0],end_unix=v[-1,0],gpu_util_mean_pct=float(v[:,1].mean()),gpu_util_p95_pct=float(np.quantile(v[:,1],.95)),gpu_memory_peak_mib=float(v[:,2].max()),load1_mean=float(v[:,3].mean()),scope='whole monitored session, includes validation, checkpointing and verification')
        else:
            good=[]
            for r in samples:
                try: good.append(tuple(float(r[k]) for k in ['unix_time','system_cpu_pct','trainer_cpu_pct_one_core','trainer_cpu_pct_capacity','trainer_rss_bytes']))
                except (ValueError,KeyError,TypeError): pass
            if good:
                v=np.asarray(good); utilization['cpu']=dict(samples=len(good),start_unix=v[0,0],end_unix=v[-1,0],system_mean_pct=float(v[:,1].mean()),trainer_mean_one_core_pct=float(v[:,2].mean()),trainer_mean_capacity_pct=float(v[:,3].mean()),trainer_peak_rss_gib=float(v[:,4].max()/2**30),scope='sampling started during second PPO-128 run')
    (a.output/'utilization.json').write_text(json.dumps(utilization,indent=2))
    status=json.loads((src/'status.json').read_text()) if (src/'status.json').exists() else {}
    lines=['# Overnight ablation findings','',f"Session: `{controller['session']}`. Persistent artifacts: `{a.artifacts}`.",f"Controller elapsed: {(controller.get('ended',controller['started'])-controller['started'])/60:.1f} minutes. Stop succeeded: {controller.get('stop_succeeded',False)}.",'',f"Outcome: **{status.get('phase','preflight failed')}**.",'']
    if records:
        keys=list(records[0]); lines+=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
        lines+=['| '+' | '.join(str(r[k]) for k in keys)+' |' for r in records]
    else: lines+=['No completed training comparison is available. No algorithm ranking or learning claim is justified.']
    lines+=['','Two training seeds are preliminary evidence. Validation has only two board clusters per opponent. Colab inference timing is not browser latency. PPO probabilities are not Q-values or calibrated win probabilities.']
    if controller.get('controller_exception'): lines+=['',f"Controller exception: `{controller['controller_exception']}`."]
    if status.get('traceback'): lines+=['','```',status['traceback'],'```']
    (a.output/'metrics-report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--artifacts',type=Path,required=True); p.add_argument('--output',type=Path,required=True); main(p.parse_args())

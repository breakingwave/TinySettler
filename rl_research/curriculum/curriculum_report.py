"""CPU postprocessing; board-cluster and paired-policy/source-cluster uncertainty."""
import argparse,csv,json,os,shutil,time,tarfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from reliability_common import read,write,sha
from curriculum_core import SEEDS,RUNS,OPPONENTS

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def summary(rows):
    n=len(rows)
    return dict(games=n,wins=sum(r['win'] for r in rows),losses=sum(r['loss'] for r in rows),truncations=sum(r['truncated'] for r in rows),completed=sum(r['terminal'] for r in rows),win_fraction=sum(r['win'] for r in rows)/max(1,n),mean_points=sum(r['points'] for r in rows)/max(1,n),illegal_actions=sum(r['illegal_actions'] for r in rows),inference_ms=1000*sum(r['inference_seconds'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),evaluation_seconds=sum(r['seconds'] for r in rows))

def paired_interval(rows,opponent=None,draws=10000):
    subset=[r for r in rows if opponent is None or r['opponent'] in opponent]
    lookup={(r['seed'],r['arm'],r['board_seed'],r['opponent'],r['seat']):r for r in subset}
    boards=sorted({r['board_seed'] for r in subset});opps=opponent or OPPONENTS
    pairs=[s for s in SEEDS if all((s,arm,b,o,seat) in lookup for arm in ['standard','curriculum'] for b in boards for o in opps for seat in [0,1])]
    if not pairs or not boards:return dict(complete_pairs=pairs,boards=len(boards),incomplete=True)
    result=dict(complete_pairs=pairs,boards=len(boards),draws=draws,contrast='curriculum minus standard',cluster='board, retaining both seats/opponents; crossed paired starting policies')
    rng=np.random.default_rng(112233)
    board_ids=rng.integers(len(boards),size=(draws,len(boards)));pair_ids=rng.integers(len(pairs),size=(draws,len(pairs)))
    for metric in ['win','truncated','points']:
        data=np.array([[np.mean([lookup[s,'curriculum',b,o,seat][metric]-lookup[s,'standard',b,o,seat][metric] for o in opps for seat in [0,1]]) for b in boards] for s in pairs],dtype=float)
        conditional=data[:,board_ids].mean(axis=(0,2))
        hierarchical=data[pair_ids[:,:,None],board_ids[:,None,:]].mean(axis=(1,2))
        result[metric]=dict(difference=float(data.mean()),by_pair={str(s):float(data[i].mean()) for i,s in enumerate(pairs)},board_only_95=np.quantile(conditional,[.025,.975]).tolist(),paired_seed_and_board_95=np.quantile(hierarchical,[.025,.975]).tolist())
    return result

def task_intervals(rows,task):
    subset=[r for r in rows if r['task']==task];games=sorted({r['source_game'] for r in subset})
    pairs=[s for s in SEEDS if all(any(r['seed']==s and r['arm']==a and r['source_game']==g for r in subset) for a in ['standard','curriculum'] for g in games)]
    if not games or not pairs:return {}
    rng=np.random.default_rng(4411);n=10000;gs=rng.integers(len(games),size=(n,len(games)));ss=rng.integers(len(pairs),size=(n,len(pairs)))
    result=dict(source_games=games,paired_starting_seeds=pairs)
    for metric in ['task_success','win','truncated']:
        data=[]
        for seed in pairs:
            data.append([np.mean([r[metric] for r in subset if r['seed']==seed and r['arm']=='curriculum' and r['source_game']==g])-np.mean([r[metric] for r in subset if r['seed']==seed and r['arm']=='standard' and r['source_game']==g]) for g in games])
        data=np.array(data);result[metric]=dict(difference=float(data.mean()),source_game_only_95=np.quantile(data[:,gs].mean(axis=(0,2)),[.025,.975]).tolist(),paired_seed_and_source_95=np.quantile(data[ss[:,:,None],gs[:,None,:]].mean(axis=(1,2)),[.025,.975]).tolist())
    return result

def main(root,out):
    out.mkdir(parents=True,exist_ok=True);remote=root/'retrieved';frozen=read(remote/'frozen.json')
    full=[];tasks=[];runs=[];curves=[];episode_rows=[];training_timeouts=[]
    for arm,seed in RUNS:
        folder=remote/f'{arm}-{seed}'
        result=read(folder/'result.json') if (folder/'result.json').exists() else dict(complete=False,reason='no result')
        row=dict(arm=arm,seed=seed,**result)
        if (folder/'training.json').exists():
            training=read(folder/'training.json');training_timeouts.extend(dict(arm=arm,seed=seed,**d) for d in training.get('truncation_diagnostics',[]));row.update({k:training[k] for k in ['stats','reset_counts','category_transitions','planning_totals']})
            episodes=training['episodes'];episode_rows.extend(dict(arm=arm,seed=seed,**e) for e in episodes);row['completed_by_category']={cat:dict(episodes=len(es),terminals=sum(e['terminal'] for e in es),truncations=sum(e['truncated'] for e in es),wins=sum(e['reward']==1 for e in es)) for cat in ['fresh','midgame','endgame'] if (es:=[e for e in episodes if e['category']==cat])}
            for m in training['metrics']:curves.append(dict(arm=arm,seed=seed,**m))
        runs.append(row)
        for directory,dest in [('evaluations',full),('tasks',tasks)]:
            path=remote/directory/f'{arm}-{seed}.json'
            if path.exists():
                for r in read(path)['rows']:dest.append(dict(arm=arm,seed=seed,**r))
    write(out/'training-timeout-diagnostics.json',training_timeouts);csv_write(out/'training-episodes.csv',episode_rows);csv_write(out/'full-game-outcomes.csv',full);csv_write(out/'exercise-outcomes.csv',tasks);csv_write(out/'learning-curves.csv',curves);write(out/'runs.json',runs)
    per_pair=[];per_opponent=[];aggregate={}
    for arm in ['standard','curriculum']:
        aggregate[arm]=summary([r for r in full if r['arm']==arm])
        for seed in SEEDS:per_pair.append(dict(arm=arm,seed=seed,**summary([r for r in full if r['arm']==arm and r['seed']==seed])))
        for opponent in OPPONENTS:per_opponent.append(dict(arm=arm,opponent=opponent,**summary([r for r in full if r['arm']==arm and r['opponent']==opponent])))
    write(out/'aggregate.json',aggregate);csv_write(out/'per-pair.csv',per_pair);csv_write(out/'per-opponent.csv',per_opponent)
    comparisons={'all':paired_interval(full),'base_and_contender':paired_interval(full,['base','contender'])}
    comparisons.update({o:paired_interval(full,[o]) for o in OPPONENTS});write(out/'paired-comparisons.json',comparisons)
    task_summary=[]
    for arm in ['standard','curriculum']:
        for task in ['largest_army','expansion','productive_trade','immediate_win']:
            rs=[r for r in tasks if r['arm']==arm and r['task']==task];completion=[r['completion']['learner_decisions'] for r in rs if r['task_success']]
            task_summary.append(dict(arm=arm,task=task,successes=sum(r['task_success'] for r in rs),success_fraction=sum(r['task_success'] for r in rs)/max(1,len(rs)),mean_decisions_if_success=float(np.mean(completion)) if completion else None,**summary(rs)))
    csv_write(out/'exercise-summary.csv',task_summary);write(out/'exercise-comparisons.json',{task:task_intervals(tasks,task) for task in ['largest_army','expansion','productive_trade','immediate_win']})
    action_rows=[]
    for arm in ['standard','curriculum']:
        for opponent in OPPONENTS:
            rs=[r for r in full if r['arm']==arm and r['opponent']==opponent];counts=Counter();planning=Counter()
            for r in rs:
                counts.update(r['diagnostic']['action_type_counts']['learner'])
                for p in r['diagnostic']['planning'].values():planning.update(p)
            action_rows.append(dict(arm=arm,opponent=opponent,games=len(rs),action_counts=dict(counts),planning=dict(planning)))
    write(out/'action-use.json',action_rows);write(out/'timeout-diagnostics.json',[r for r in full+tasks if r['truncated']])
    controller=read(root/'controller.json');write(out/'time-accounting.json',dict(allocated_upper_bound_seconds=controller.get('total_allocated_upper_bound_seconds'),training_seconds=sum(r.get('stats',{}).get('elapsed',0) for r in runs),evaluation_seconds=sum(r['seconds'] for r in full),task_seconds=sum(r['seconds'] for r in tasks),pilot_seconds=sum(r['elapsed'] for r in frozen['pilot_results']),deadline=controller['deadline'],started=controller['started'],ended=controller.get('ended'),termination_verified=controller.get('termination_verified')))
    for name in ['input-manifest.json','historical-checks.json','corpus-manifest.json','historical-diagnostics.json','opponent-preflight.json','launch-integrity.json','controller.json','user-stop.json','sessions-after-user-stop.txt','sessions-after.txt','technical-recovery.json']:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    for name in ['frozen.json','preflight.json','external-recovery.json','installed-freeze.txt','calibration-pilots.json','telemetry.csv','cpu-telemetry.csv','tests.log','artifact-index.json','preflight-before-replacement1.json','telemetry-before-replacement1.csv','cpu-telemetry-before-replacement1.csv','preflight-before-replacement2.json','telemetry-before-replacement2.csv','cpu-telemetry-before-replacement2.csv']:
        if (remote/name).exists():shutil.copy2(remote/name,out/name)
    write(out/'artifact-manifest.json',dict(root=str(root),files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file() and (p.is_relative_to(remote) or p.name in ['source.tar.gz','historical.tar.gz','corpus.pt','diagnostic-states.pt'])},report_source_sha256=sha(__file__)))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tinysettler-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for axis,opps,title in [(axes[0],OPPONENTS,'All five opponents'),(axes[1],['base','contender'],'Adapted base + held-out contender')]:
        for j,arm in enumerate(['standard','curriculum']):
            vals=[summary([r for r in full if r['seed']==s and r['arm']==arm and r['opponent'] in opps])['win_fraction']*100 for s in SEEDS]
            axis.bar(np.arange(3)+(j-.5)*.36,vals,width=.36,label=arm)
        axis.set(xticks=np.arange(3),xticklabels=SEEDS,ylabel='Full-game wins (%)',ylim=(0,100),title=title,xlabel='Paired starting PPO-128 seed');axis.legend()
    fig.tight_layout();fig.savefig(out/'paired-full-game-results.png',dpi=160);fig.savefig(out/'paired-full-game-results.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm,seed in RUNS:
        ms=[r for r in curves if r['arm']==arm and r['seed']==seed]
        axes[0 if arm=='standard' else 1].plot([r['n'] for r in ms],[r['loss'] for r in ms],label=str(seed),alpha=.75)
    for ax,arm in zip(axes,['standard','curriculum']):ax.set(title=arm,xlabel='Additional learner transitions',ylabel='PPO composite loss');ax.legend()
    fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(aggregate=aggregate,completed=sum(r['complete'] for r in runs),full_games=len(full),exercise_games=len(tasks),target=frozen['target'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.output)

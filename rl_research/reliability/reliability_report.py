"""Reproduce paired reliability summaries from raw games, retaining board/seed uncertainty."""
import argparse,csv,json,shutil,time,os
from collections import Counter
from pathlib import Path
import numpy as np
from reliability_common import read,write,sha,ALL_RUNS,name,STEPS,OLD_SEEDS,NEW_SEEDS

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def counts(rows):
    return dict(games=len(rows),wins=sum(r['win'] for r in rows),losses=sum(r['loss'] for r in rows),truncations=sum(r['truncated'] for r in rows),completed=sum(not r['truncated'] for r in rows),win_fraction=float(np.mean([r['win'] for r in rows])) if rows else None,terminal_reward_frequency=float(np.mean([r['terminal'] for r in rows])) if rows else None,terminal_rewards_per_decision=sum(r['terminal'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),mean_points=float(np.mean([r['points'] for r in rows])) if rows else None,illegal_actions=sum(r['illegal_actions'] for r in rows),inference_ms=1000*sum(r['inference_seconds'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),learner_decisions_s=sum(r['decisions'] for r in rows)/max(1e-9,sum(r['seconds'] for r in rows)),engine_actions_s=sum(r['engine_actions'] for r in rows)/max(1e-9,sum(r['seconds'] for r in rows)))
def intervals(matrix,rng):
    # Crossed resampling: paired fitted seeds and shared board clusters, retaining seats/opponents.
    ns,nb=matrix.shape;draws=10000
    bi=rng.integers(nb,size=(draws,nb));si=rng.integers(ns,size=(draws,ns))
    conditional=matrix.mean(0)[bi].mean(1)
    hierarchical=np.asarray([matrix[s[:,None],b[None,:]].mean() for s,b in zip(si,bi)])
    return dict(estimate=float(matrix.mean()),board_cluster_ci95=np.quantile(conditional,[.025,.975]).tolist(),paired_training_seed_and_board_ci95=np.quantile(hierarchical,[.025,.975]).tolist(),training_seeds=ns,board_clusters=nb,bootstrap_draws=draws)
def paired(rows):
    seeds=sorted(set(r['training_seed'] for r in rows));boards=sorted(set(r['board_seed'] for r in rows));rng=np.random.default_rng(82219);result={}
    for metric in ['win','truncated','points']:
        matrices={}
        for w in [128,256]:
            matrix=np.asarray([[np.mean([r[metric] for r in rows if r['width']==w and r['training_seed']==s and r['board_seed']==b]) for b in boards] for s in seeds])
            assert np.isfinite(matrix).all();matrices[w]=matrix
            result[f'width_{w}_{metric}']=intervals(matrix,rng)
        result[f'width_128_minus_256_{metric}']=intervals(matrices[128]-matrices[256],rng)
    return result

def main(artifacts,out):
    root=artifacts/'retrieved';out.mkdir(parents=True,exist_ok=True)
    frozen=read(root/'frozen.json');selections=read(root/'selections.json')['selections'];selected={x['run']:x for x in selections}
    allrows=[];perseed=[];curves=[];diagnostics=[];primary=[];secondary=[]
    for split in ['validation','final','secondary']:
        for p in sorted((root/'evaluations'/split).glob('*/*.json')):
            run=p.parent.name;_,w,s=run.split('-');n=int(p.stem)
            for raw in read(p)['rows']:
                row=dict(split=split,run=run,width=int(w),training_seed=int(s),step=n,**raw);allrows.append(row)
                if row.get('diagnostic'):diagnostics.append({k:v for k,v in row.items() if k!='diagnostic'}|row['diagnostic'])
    for w,s in ALL_RUNS:
        run=name(w,s);rows=[r for r in allrows if r['split']=='final' and r['run']==run and r['step']==434176]
        resultfile=root/run/'result.json';complete=s in OLD_SEEDS or resultfile.exists() and read(resultfile)['complete']
        stats=read(resultfile)['stats'] if resultfile.exists() and 'stats' in read(resultfile) else {}
        if s in OLD_SEEDS:stats=read(Path(frozen['historical_location'])/'retrieved'/run/'434176.json')['stats']
        if rows and complete:primary.extend(rows)
        perseed.append(dict(run=run,width=w,seed=s,cohort='historical' if s in OLD_SEEDS else 'new',training_complete=complete,transitions=434176 if complete else read(resultfile).get('n',0) if resultfile.exists() else 0,selected_step=selected.get(run,{}).get('step'),training_minutes=stats.get('elapsed',0)/60 if stats else None,training_decisions_s=stats.get('learner_decisions_s'),training_engine_actions_s=stats.get('engine_actions_s'),training_episodes=stats.get('episodes'),training_terminal_rewards=stats.get('terminal_rewards'),training_terminal_rewards_per_transition=stats.get('terminal_rewards',0)/max(1,stats.get('n',0)) if stats else None,**counts(rows)))
        for n in STEPS:
            val=[r for r in allrows if r['split']=='validation' and r['run']==run and r['step']==n]
            meta=(Path(frozen['historical_location'])/'retrieved' if s in OLD_SEEDS else root)/run/f'{n}.json';trainstats=read(meta)['stats'] if meta.exists() else {}
            curves.append(dict(run=run,width=w,seed=s,step=n,training_minutes=trainstats.get('elapsed',0)/60 if trainstats else None,**counts(val)))
        if resultfile.exists():
            latest=max((root/run).glob('[0-9]*.json'),key=lambda p:int(p.stem))
            for row in read(latest).get('diagnostics',[]):diagnostics.append(dict(split='training',run=run,width=w,training_seed=s,**row))
    # Require identical complete primary coverage before any paired inference.
    common_seeds=[s for s in OLD_SEEDS+NEW_SEEDS if all(any(r['width']==w and r['training_seed']==s for r in primary) for w in [128,256])]
    common_boards=min((len([r for r in primary if r['run']==name(w,s)])//6 for s in common_seeds for w in [128,256]),default=0)
    primary=[r for r in primary if r['training_seed'] in common_seeds and r['board_seed']<frozen['splits']['final']['seed_start']+common_boards]
    secondary_by_run={}
    for sel in selections:
        split='final' if sel['step']==434176 else 'secondary'
        secondary_by_run[sel['run']]=[r for r in allrows if r['split']==split and r['run']==sel['run'] and r['step']==sel['step']]
    sb=min((len(v)//6 for v in secondary_by_run.values()),default=0)
    for rows in secondary_by_run.values():secondary.extend(rows[:sb*6])
    aggregate=[dict(width=w,**counts([r for r in primary if r['width']==w])) for w in [128,256]]
    opponents=[dict(width=w,opponent=opp,**counts([r for r in primary if r['width']==w and r['opponent']==opp])) for w in [128,256] for opp in ['random','weighted','heuristic']]
    comparison=paired(primary) if primary else {}
    write(out/'paired-comparisons.json',comparison)
    if secondary and len(set(r['training_seed'] for r in secondary))==len(common_seeds):write(out/'selected-paired-comparisons.json',paired(secondary))
    write(out/'aggregate.json',aggregate);write(out/'opponent-comparison.json',opponents)
    if primary:write(out/'opponent-paired-comparisons.json',{opp:paired([r for r in primary if r['opponent']==opp]) for opp in ['random','weighted','heuristic']})
    write(out/'seed-variability.json',[dict(width=w,win_fractions=[r['win_fraction'] for r in perseed if r['width']==w and r['games']],observed_seed_sd=float(np.std([r['win_fraction'] for r in perseed if r['width']==w and r['games']],ddof=1))) for w in [128,256]])
    write(out/'timeout-diagnostics.json',diagnostics)
    timeout_summary=[]
    for split in ['training','validation','final','secondary']:
        for w in [128,256]:
            group=[d for d in diagnostics if d['split']==split and d['width']==w]
            if not group:continue
            timeout_summary.append(dict(split=split,width=w,truncations=len(group),turn_limit=sum(d['turn_limit_reached'] for d in group),action_limit=sum(d['action_limit_reached'] for d in group),mean_own_points=float(np.mean([d['own_points'] for d in group])),mean_learner_maritime_trades=float(np.mean([d['maritime_trades']['learner'] for d in group])),mean_learner_end_turn_frequency=float(np.mean([d['end_turn_frequency']['learner'] for d in group])),mean_repeated_public_own_hand_states=float(np.mean([d['repeated_public_own_hand_states'] for d in group])),max_state_visits=max(d['max_state_visits'] for d in group)))
    write(out/'timeout-summary.json',timeout_summary)
    secondary_table=[dict(run=k,step=selected[k]['step'],**counts(v[:sb*6])) for k,v in secondary_by_run.items()]
    write(out/'secondary-summary.json',dict(common_boards=sb,per_policy=secondary_table))
    csv_write(out/'selected-checkpoint-per-seed.csv',secondary_table)
    csv_write(out/'common-step-per-seed.csv',perseed);csv_write(out/'learning-curves.csv',curves)
    csv_write(out/'final-games.csv',[{k:v for k,v in r.items() if k!='diagnostic'} for r in primary])
    csv_write(out/'selected-games.csv',[{k:v for k,v in r.items() if k!='diagnostic'} for r in secondary])
    # Verify every retrieved checkpoint against the remote index before declaring artifact completion.
    index=read(root/'artifact-index.json') if (root/'artifact-index.json').exists() else {}
    for key,expected in index.items():assert sha(root/key)==expected['sha256'] and (root/key).stat().st_size==expected['bytes']
    files={str(p.relative_to(artifacts)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts.rglob('*') if p.is_file() and (('retrieved' in p.relative_to(artifacts).parts and not p.name.endswith('.tmp')) or p.name in ['source.tar.gz','input-manifest.json','controller.json','recovery-controller.py','calibration-repair.py','secondary-plan.py','operational-events.json','checkpoint-package-manifest.json','common-final-and-selected.tar.gz'])}
    write(out/'artifact-manifest.json',dict(location=str(artifacts),files=files,remote_index_verified=len(index),report_source_sha256=sha(__file__),controller=read(artifacts/'controller.json')))
    for filename in ['frozen.json','calibration.json','preflight.json','selections.json','external-recovery.json','secondary-plan.json','status.json','telemetry.csv','cpu-telemetry.csv','installed-freeze.txt']:
        if (root/filename).exists():shutil.copy2(root/filename,out/filename)
    shutil.copy2(artifacts/'input-manifest.json',out/'input-manifest.json')
    for filename in ['recovery-checks.json','checkpoint-package-manifest.json','operational-events.json','controller.json']:
        if (artifacts/filename).exists():shutil.copy2(artifacts/filename,out/filename)
    utilization={}
    def distribution(values):
        a=np.asarray(values,dtype=float)
        return dict(samples=len(a),mean=float(a.mean()),median=float(np.median(a)),maximum=float(a.max())) if len(a) else {}
    cpu_path=root/'cpu-telemetry.csv'
    if cpu_path.exists():
        cpu=list(csv.DictReader(cpu_path.open()))
        for field in ['system_cpu_pct','trainer_cpu_pct_one_core','trainer_cpu_pct_capacity','trainer_rss_bytes']:utilization[field]=distribution([r[field] for r in cpu])
    gpu_path=root/'telemetry.csv'
    if gpu_path.exists():
        gpu=list(csv.DictReader(gpu_path.open()))
        for field,column_index in [('gpu_utilization_pct',1),('gpu_memory_used_mib',3)]:utilization[field]=distribution([r['gpu_csv'].split(',')[column_index].strip() for r in gpu if len(r['gpu_csv'].split(','))==5])
    utilization['scope']='Samples across the replacement runtime workload, including training, evaluation and gates; not browser measurements.'
    write(out/'utilization.json',utilization)
    phase_seconds={split:sum(read(p).get('wall_seconds',0) for p in (root/'evaluations'/split).glob('*/*.json')) for split in ['validation','final','secondary']}
    write(out/'time-accounting.json',dict(new_training_seconds=sum(read(p).get('stats',{}).get('elapsed',0) for p in root.glob('ppo-*/result.json')),evaluation_seconds=phase_seconds,allocation_upper_bound_seconds=read(artifacts/'controller.json').get('total_allocated_upper_bound_seconds'),note='Training timers exclude validation. Evaluation wall times include resumed/cached calls; setup, transfers, calibration, recovery gates and cleanup are contained in allocation accounting.'))

    raw=out/'raw';raw.mkdir(exist_ok=True)
    for p in root.rglob('*.json'):
        target=raw/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tinysettler-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,w in zip(axes,[128,256]):
        for s in OLD_SEEDS+NEW_SEEDS:
            g=[r for r in curves if r['width']==w and r['seed']==s and r['games']]
            ax.plot([r['step'] for r in g],[r['win_fraction'] for r in g],marker='o',label=str(s),linestyle='--' if s in OLD_SEEDS else '-')
        ax.set(title=f'PPO-{w}: expanded validation',xlabel='Learner transitions',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.25)
    axes[0].set_ylabel('Observed win fraction');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'learning-curves.{ext}',dpi=160)
    fig2,axes2=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,w in zip(axes2,[128,256]):
        for seed in OLD_SEEDS+NEW_SEEDS:
            g=[r for r in curves if r['width']==w and r['seed']==seed and r['games'] and r['training_minutes'] is not None]
            ax.plot([r['training_minutes'] for r in g],[r['win_fraction'] for r in g],marker='o',label=str(seed),linestyle='--' if seed in OLD_SEEDS else '-')
        ax.set(title=f'PPO-{w}: validation vs training time',xlabel='Training minutes (evaluation excluded)',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.25)
    axes2[0].set_ylabel('Observed win fraction');fig2.tight_layout()
    for ext in ['png','svg']:fig2.savefig(out/f'wall-clock-curves.{ext}',dpi=160)
    fig3,ax=plt.subplots(figsize=(8,4))
    for w,color in [(128,'tab:blue'),(256,'tab:orange')]:
        g=[r for r in perseed if r['width']==w and r['games']]
        ax.plot([r['seed'] for r in g],[100*r['win_fraction'] for r in g],'o',label=f'PPO-{w}',color=color)
    ax.set(xlabel='Independent training seed',ylabel='Common-final test wins (%)',title='Fixed-budget policies: five training seeds per width',ylim=(0,100));ax.set_xticks(OLD_SEEDS+NEW_SEEDS);ax.ticklabel_format(style='plain',axis='x');ax.legend();ax.grid(alpha=.25);fig3.tight_layout()
    for ext in ['png','svg']:fig3.savefig(out/f'common-final-per-seed.{ext}',dpi=160)


    lines=['# PPO reliability results','',f'Primary endpoint: {common_boards} fresh boards × both seats × three fixed opponents, at 434,176 transitions. Paired training seeds: {len(common_seeds)}.','', '| Width | Wins | Losses | Truncations | Win fraction | Colab inference ms |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for x in aggregate:lines.append(f"| {x['width']} | {x['wins']} | {x['losses']} | {x['truncations']} | {x['win_fraction']:.2%} | {x['inference_ms']:.3f} |")
    lines+=['','Intervals retain each board’s two seats and three opponents together. Board-only intervals condition on the fitted policies. The separate crossed bootstrap resamples paired training seeds as well as shared board clusters. Five seeds give preliminary evidence about training variability.','',f'Secondary selection uses win fraction, fewer truncations, then earlier step; {sb} common test boards per policy. Selections were frozen before final testing. This differs from the original 12-game mean-reward selection.','','The fixed heuristic prioritizes ending the turn over every maritime trade. These results measure reliability against the specified opponents, not general Catan strength, negotiation quality, or browser latency. PPO probabilities are not Q-values or calibrated win probabilities.']
    (out/'metrics-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary_games=len(primary),secondary_games=len(secondary),diagnostics=len(diagnostics),verified_checkpoints=len(index),aggregate=aggregate)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.output)

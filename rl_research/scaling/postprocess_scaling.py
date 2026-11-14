"""Offline final audit, summaries and publication-label refinements; no model changes."""
import argparse,csv,json,hashlib,statistics,datetime,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tinysettler-scaling-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def csvwrite(p,data):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main(root,out):
 frozen=read(out/'frozen.json');runs=frozen['runs'];training=rows(out/'training-runs.csv');metrics=rows(out/'ppo-metrics.csv');episodes=rows(out/'training-episodes.csv');final=rows(out/'final-games.csv');validation=rows(out/'validation-games.csv');colors=dict(zip('ABCD',['#0072b2','#e69f00','#009e73','#cc79a7']));styles=['-','--',':'];seeds=[400009,400031,400033]
 labels={c:{'A':'MLP128 static','B':'MLP128 refresh','C':'MLP512 static','D':'MLP512 refresh'}[c] for c in 'ABCD'}
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 def save(fig,name,legend=False):
  if legend:
   handles=[Line2D([0],[0],color=colors[c],ls=styles[seeds.index(s)],label=f'{c}/{s}') for c,s in runs];fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=8);fig.tight_layout(rect=(0,.08,1,1))
  else:fig.tight_layout(rect=(0,.13,1,1) if fig.legends else (0,0,1,1))
  for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
  plt.close(fig)
 # Verify immutable executed source, all retained object generations and retrieved paths.
 manifest=read(root/'input-manifest.json');assert manifest['source_hashes']==read(root/'retrieved/preflight.json')['source_hashes']
 for name,h in manifest['source_hashes'].items():assert sha(root/'source'/name)==h
 for name,info in read(root/'launch-integrity.json').items():assert sha(root/name)==info['sha256']
 for name,key in [('archive.pt','archive_sha256'),('teacher.pt','teacher_sha256')]:assert sha(root/'inputs'/name)==manifest[key]
 generations=sorted((root/'generations').glob('*.json'));objects=set()
 for g in generations:
  for rel,info in read(g)['files'].items():
   p=root/'objects'/info['sha256'];assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'];objects.add(info['sha256'])
 for rel,info in read(generations[-1])['files'].items():assert sha(root/'retrieved'/rel)==info['sha256']
 write(out/'artifact-audit.json',dict(source_files=len(manifest['source_hashes']),source_archive_sha256=sha(root/'source.tar.gz'),retained_generations=len(generations),verified_unique_objects=len(objects),latest_verified_files=len(read(generations[-1])['files']),all_passed=True))
 write(out/'source-hashes.json',manifest['source_hashes'])
 # Independent row totals and complete-board coverage, without using report aggregates.
 totals={}
 for file in ['final-games.csv','validation-games.csv','regression-games.csv','historical-context.csv','confirmation-games.csv']:
  rr=rows(out/file);totals[file]=dict(games=len(rr),wins=sum(r['win']=='True' for r in rr),losses=sum(r['loss']=='True' for r in rr),truncations=sum(r['truncated']=='True' for r in rr),illegal_actions=sum(int(r['illegal_actions']) for r in rr),infrastructure_failures=sum(r['infrastructure_failure']=='True' for r in rr))
  assert all((r['win']=='True')+(r['loss']=='True')+(r['truncated']=='True')==1 for r in rr)
 assert sum(totals[k]['games'] for k in ['final-games.csv','regression-games.csv','historical-context.csv','confirmation-games.csv'])==frozen['final_games']
 condition_summary=[];seed_summary=[]
 for c in 'ABCD':
  tr=[r for r in training if r['condition']==c];ee=[r for r in episodes if r['condition']==c];rr=[r for r in final if r['condition']==c and r['mode']=='sampled'];diag=next(r for r in rows(out/'strategic-diagnostics.csv') if r['condition']==c)
  condition_summary.append(dict(condition=c,model=labels[c],seeds=len(tr),transitions_per_run=frozen['target'],mean_training_minutes=statistics.mean(float(r['elapsed'])/60 for r in tr),learner_decisions_per_second=sum(int(r['n']) for r in tr)/sum(float(r['elapsed']) for r in tr),engine_actions_per_second=sum(int(r['engine_actions']) for r in tr)/sum(float(r['elapsed']) for r in tr),base_wins=sum(r['win']=='True' for r in rr if r['opponent']=='base'),contender_wins=sum(r['win']=='True' for r in rr if r['opponent']=='contender'),heldout_wins=sum(r['win']=='True' for r in rr if r['opponent']=='heldout'),games_per_opponent=40*len(tr),completed_training_episodes=sum(r['terminal']=='True' for r in ee),training_truncations=sum(r['truncated']=='True' for r in ee),terminal_reward_frequency=sum(r['terminal']=='True' for r in ee)/sum(int(r['n']) for r in tr),first_city_never_fraction=float(diag['first_city_never_fraction']),third_settlement_never_fraction=float(diag['third_settlement_never_fraction'])))
  for t in tr:
   s=t['seed'];sr=[r for r in rr if r['seed']==s];seed_summary.append(dict(condition=c,seed=int(s),heuristic_wins=sum(r['win']=='True' for r in sr if r['opponent']!='heldout'),heuristic_games=80,heldout_wins=sum(r['win']=='True' for r in sr if r['opponent']=='heldout'),heldout_games=40,training_minutes=float(t['elapsed'])/60))
 csvwrite(out/'comparison-table.csv',condition_summary);csvwrite(out/'seed-summary.csv',seed_summary)
 auxiliary=[]
 for filename in ['regression-games.csv','historical-context.csv']:
  rr=rows(out/filename)
  for c in sorted({r['condition'] for r in rr}):
   for mode in sorted({r['mode'] for r in rr}):
    for opponent in sorted({r['opponent'] for r in rr}):
     group=[r for r in rr if r['condition']==c and r['mode']==mode and r['opponent']==opponent]
     if group:auxiliary.append(dict(panel=filename,condition=c or 'historical',mode=mode,opponent=opponent,games=len(group),wins=sum(r['win']=='True' for r in group),losses=sum(r['loss']=='True' for r in group),truncations=sum(r['truncated']=='True' for r in group)))
 csvwrite(out/'auxiliary-summary.csv',auxiliary)
 controller=read(out/'controller.json');alloc=controller['allocations'];assert all(a['terminated'] for a in alloc)
 for a,b in zip(alloc,alloc[1:]):assert a['stopped']<=b['requested']
 ledger=[dict(session=a['session'],requested_utc=datetime.datetime.fromtimestamp(a['requested'],datetime.timezone.utc).isoformat(),stopped_utc=datetime.datetime.fromtimestamp(a['stopped'],datetime.timezone.utc).isoformat(),seconds=a['stopped']-a['requested'],terminated=True) for a in alloc];csvwrite(out/'runtime-ledger.csv',ledger)
 t0=controller['started'];gpu=rows(out/'telemetry.csv');gpu_active=[r for r in gpu if float(r['cpu_pct_one_core'])>50];copies=[e['time'] for e in controller['events'] if e['kind']=='verified_generation'];training_copies=[x for x in copies if x>=frozen['created'] and x<=read(out/'selection.json')['time']];copy_gaps=np.diff(training_copies)
 resource=dict(elapsed_original_clock_seconds=controller['elapsed_upper_bound'],allocated_upper_bound_seconds=sum(a['seconds'] for a in ledger),hard_limit_seconds=50400,all_terminated=True,**read(out/'experiment-state.json'),experimental_transitions=sum(int(r['n']) for r in training),training_terminal_episodes=sum(r['terminal']=='True' for r in episodes),training_truncations=sum(r['truncated']=='True' for r in episodes),terminal_reward_frequency=sum(r['terminal']=='True' for r in episodes)/sum(int(r['n']) for r in training),training_copy_gap_median_seconds=float(np.median(copy_gaps)),training_copy_gap_max_seconds=float(max(copy_gaps)),active_worker_gpu_utilization_mean=float(np.mean([float(r['gpu'].split(',')[1]) for r in gpu_active])),nonfinite_losses=sum(int(r['nonfinite_count']) for r in metrics),evaluation=totals)
 write(out/'resource-summary.json',resource)
 # 2: visible run identities; G omitted, so teacher-inclusive per-run cost is identical.
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
 for ax in axes:
  for c,s in runs:
   mm=[m for m in metrics if m['condition']==c and int(m['seed'])==s];steps=sorted({int(r['n']) for r in validation if r['condition']==c and int(r['seed'])==s});xs=[];ys=[]
   for n in steps:
    prior=[m for m in mm if int(m['n'])<=n];xs.append(float(prior[-1]['elapsed'])/60 if prior else 0);rr=[r for r in validation if r['condition']==c and int(r['seed'])==s and int(r['n'])==n and r['mode']=='sampled' and r['opponent'] in ['base','contender']];ys.append(sum(r['win']=='True' for r in rr)/len(rr))
   ax.plot(xs,ys,color=colors[c],ls=styles[seeds.index(s)])
  ax.set(xlabel='Minutes per run',ylabel='Validation heuristic-family win fraction',ylim=(0,1))
 axes[0].set_title('Retained PPO time');axes[1].set_title('Teacher-inclusive time (G omitted; identical)');save(fig,'02-validation-time',True)
 # 4: explanatory labels and unclipped omitted contrasts.
 comparisons=read(out/'paired-comparisons.json');names={'self_play_128':'B − A: refresh, MLP128','capacity_static':'C − A: width, static','capacity_refresh':'D − B: width, refresh','structure_static':'E − C: structure, static','structure_refresh':'F − D: structure, refresh','teacher':'G − B: teacher','capacity_selfplay_interaction':'(D − C) − (B − A)','structure_selfplay_interaction':'(F − E) − (D − C)'}
 fig,axes=plt.subplots(1,2,figsize=(14,6))
 for ax,endpoint in zip(axes,['heuristic_family','heldout_neural']):
  for i,(name,d) in enumerate(comparisons.items()):
   v=d.get(endpoint,{})
   if 'difference' not in v:ax.text(0,i,'not estimated',va='center',fontsize=8);continue
   lo,hi=np.array(v['crossed_seed_board_95'])*100;mean=v['difference']*100;ax.plot([lo,hi],[i,i],color='black');ax.plot(mean,i,'ko');ax.scatter(np.array(list(v['per_seed'].values()))*100,np.arange(len(v['per_seed']))*.07+i+.14,s=20,color='#0072b2',alpha=.65)
  ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(len(names)),yticklabels=list(names.values()),ylim=(-.5,len(names)-.5),xlabel='Win difference (percentage points)\nBlack: crossed seed/board 95%; blue: individual seeds',title=endpoint.replace('_',' '))
 save(fig,'04-paired-contrasts')
 # 5: avoid overlapping direct labels; each color/marker combination identifies a seed.
 fig,axes=plt.subplots(1,2,figsize=(12,4.8));handles=[]
 for c,s in runs:
  rr=[r for r in final if r['condition']==c and int(r['seed'])==s and r['mode']=='sampled' and r['opponent']!='heldout'];tr=next(t for t in training if t['condition']==c and int(t['seed'])==s);y=sum(r['win']=='True' for r in rr)/len(rr);lat=1000*sum(float(r['end_to_end_decision_seconds']) for r in rr)/sum(int(r['decisions']) for r in rr);marker=['o','s','^'][seeds.index(s)]
  for ax,x in zip(axes,[int(tr['parameters']),lat]):ax.scatter(x,y,edgecolors=colors[c],facecolors='none',marker=marker,s=80,linewidths=1.7)
  handles.append(Line2D([],[],color=colors[c],marker=marker,ls='',markerfacecolor='none',label=f'{c}/{s}'))
 axes[0].set(xlabel='Trainable parameters',ylabel='Sampled heuristic-family win fraction',xlim=(100000,1000000));axes[1].set(xlabel='Colab end-to-end decision latency (ms)',ylabel='Sampled heuristic-family win fraction');fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));save(fig,'05-strength-cost')
 # 6: exact opportunity definitions and denominators; no claim that every trade is useful.
 d=rows(out/'strategic-diagnostics.csv');fig,axes=plt.subplots(2,3,figsize=(13,7));keys=['first_city_never_fraction','third_settlement_never_fraction','timeout_fraction','immediate_win_conversion','PLAY_KNIGHT_CARD_conversion','MARITIME_TRADE_conversion'];titles=['No city before game ended','No third settlement build before game ended','Administrative truncations','Immediate build/knight win chosen / legal','Knight played / legal decision opportunities','Maritime trade chosen / legal opportunities']
 for ax,key,title in zip(axes.flat,keys,titles):
  bars=ax.bar([x['condition'] for x in d],[float(x[key]) for x in d],color=[colors[x['condition']] for x in d]);ax.set(title=title,ylim=(0,1),xlabel='Condition')
  if key.endswith('_conversion'):
   prefix=key[:-11]
   for bar,x in zip(bars,d):ax.text(bar.get_x()+bar.get_width()/2,min(.96,bar.get_height()+.03),f"{x[prefix+'_chosen']}/{x[prefix+'_opportunities']}",ha='center',fontsize=8)
 fig.suptitle('Sampled full games against base/contender only',fontsize=11);save(fig,'06-expansion-timeouts')
 # 7: legends make the policy/critic diagnostics interpretable.
 fig,axes=plt.subplots(2,3,figsize=(14,7.5))
 for ax,key in zip(axes.flat,['entropy','approx_kl','clip_fraction','explained_variance','value_loss','grad_norm_before_clip']):
  for c,s in runs:
   mm=[m for m in metrics if m['condition']==c and int(m['seed'])==s];ax.plot([int(m['n']) for m in mm],[float(m[key]) if m[key] else np.nan for m in mm],color=colors[c],ls=styles[seeds.index(s)],alpha=.65)
  ax.set(title=key.replace('_',' '),xlabel='Learner transitions')
 save(fig,'07-ppo-critic',True)
 # 8: cumulative pool shares over training, as well as exact final shares.
 fig,axes=plt.subplots(2,3,figsize=(15,8.5));tags=[f"{t['condition']}/{t['seed']}" for t in training]
 for ax,key in zip(axes[0,:2],['episode_shares','transition_shares']):
  bottom=np.zeros(len(training))
  for group in ['base','weighted','heuristic','historical','own']:
   vals=[]
   for t in training:
    counts=json.loads(t[key]);vals.append(sum(n for identity,n in counts.items() if (identity if identity in ['base','weighted','heuristic'] else 'historical' if identity.startswith(('standard','parent')) else 'own')==group)/sum(counts.values()))
   ax.bar(tags,vals,bottom=bottom,label=group);bottom+=vals
  ax.set(title='All resets' if key=='episode_shares' else 'All learner transitions',ylim=(0,1));ax.tick_params(axis='x',rotation=90);ax.legend(fontsize=7)
 for c,s in runs:
  ee=[r for r in episodes if r['condition']==c and int(r['seed'])==s];ee.sort(key=lambda r:int(r['n']));last=0;lengths=[]
  for r in ee:lengths.append(int(r['n'])-last);last=int(r['n'])
  for ax,group,measure in [(axes[1,0],'base','episodes'),(axes[1,1],'own','episodes'),(axes[1,2],'own','transitions')]:
   xs=[];ys=[]
   for step in frozen['milestones']:
    ii=[i for i,r in enumerate(ee) if int(r['n'])<=step];den=sum(lengths[i] for i in ii) if measure=='transitions' else len(ii)
    match=[i for i in ii if (ee[i]['opponent']=='base' if group=='base' else ee[i]['opponent'].startswith(c+'-'))];num=sum(lengths[i] for i in match) if measure=='transitions' else len(match);xs.append(step);ys.append(num/den)
   ax.plot(xs,ys,color=colors[c],ls=styles[seeds.index(s)])
  t=next(t for t in training if t['condition']==c and int(t['seed'])==s);ages=json.loads(t['snapshot_age_sum']);counts=json.loads(t['transition_shares'])
  if ages:axes[0,2].scatter([int(k.split('-')[-1]) for k in ages],[v/counts[k] for k,v in ages.items()],color=colors[c],marker=['o','s','^'][seeds.index(s)],alpha=.65)
 axes[0,2].set(title='Mean own-snapshot age when encountered',xlabel='Snapshot publication step',ylabel='Learner transitions since publication')
 for ax,title in zip(axes[1],['Base share: completed episodes','Own share: completed episodes','Own share: completed-episode transitions']):ax.set(title=title,xlabel='Learner transitions',ylim=(0,1))
 save(fig,'08-opponent-pools',True)
 write(out/'postprocess-audit.json',dict(no_model_or_setting_changes=True,renderer_sha256=sha(Path(__file__)),charts_refined=['02','04','05','06','07','08'],independent_row_totals=totals))
 print(json.dumps(dict(conditions=condition_summary,resources=resource),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.output)

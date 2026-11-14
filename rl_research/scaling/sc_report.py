"""Saved-data analysis: matched board clusters and paired starting-seed uncertainty."""
from sc_common import *
import argparse,csv,collections,math
os.environ.setdefault('MPLCONFIGDIR','/tmp/tinysettler-scaling-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csvwrite(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def load_rows(directory):
 rows=[]
 for p in sorted(directory.glob('*.json')):
  data=read(p)
  if 'spec' not in data:continue
  for r in data['rows']:rows.append(dict(condition=data['spec'].get('condition'),seed=data['spec'].get('training_seed'),n=data['spec'].get('n'),panel=p.name,complete_board=r['board_seed'] in data['complete_boards'],**r))
 return rows

def summary(rows):
 n=len(rows);dec=sum(r['decisions'] for r in rows)
 return dict(games=n,wins=sum(r['win'] for r in rows),losses=sum(r['loss'] for r in rows),truncations=sum(r['truncated'] for r in rows),win_fraction=sum(r['win'] for r in rows)/n if n else None,mean_points=np.mean([r['points'] for r in rows]).item() if n else None,illegal_actions=sum(r['illegal_actions'] for r in rows),inference_ms=1000*sum(r['inference_seconds'] for r in rows)/max(1,dec),end_to_end_ms=1000*sum(r['end_to_end_decision_seconds'] for r in rows)/max(1,dec),seconds=sum(r['seconds'] for r in rows))

def contrast(rows,terms,seeds,opponents,mode='sampled',draws=10000):
 # terms maps conditions to coefficients; each board retains both seats/all specified opponents.
 lookup={(r['condition'],r['seed'],r['board_seed'],r['opponent'],r['mode'],r['seat']):r for r in rows if r['complete_board']}
 boards=sorted({r['board_seed'] for r in rows if r['mode']==mode})
 available=[b for b in boards if all((c,s,b,o,mode,seat) in lookup for c in terms for s in seeds for o in opponents for seat in [0,1])]
 if not available:return dict(inconclusive=True,reason='no complete matched boards',seeds=seeds)
 data=np.array([[sum(coef*np.mean([lookup[c,s,b,o,mode,seat]['win'] for o in opponents for seat in [0,1]]) for c,coef in terms.items()) for b in available] for s in seeds])
 rng=np.random.default_rng(741852);bi=rng.integers(len(available),size=(draws,len(available)));si=rng.integers(len(seeds),size=(draws,len(seeds)))
 return dict(terms=terms,seeds=seeds,boards=len(available),excluded_board_count=len(boards)-len(available),opponents=opponents,mode=mode,difference=float(data.mean()),per_seed={str(s):float(data[i].mean()) for i,s in enumerate(seeds)},board_only_95=np.quantile(data[:,bi].mean(axis=(0,2)),[.025,.975]).tolist(),crossed_seed_board_95=np.quantile(data[si[:,:,None],bi[:,None,:]].mean(axis=(1,2)),[.025,.975]).tolist(),draws=draws)

def savefig(fig,out,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)

def main(root,out,live=False):
 out.mkdir(parents=True,exist_ok=True);remote=root/'retrieved';frozen=read(remote/'frozen.json') if (remote/'frozen.json').exists() else dict(conditions=list('ABCDEFG'),runs=[],target=0)
 validation=load_rows(remote/'validation');training=[];metrics=[];episodes=[]
 for p in sorted((remote/'runs').glob('*/training.json')):
  c,s=p.parent.name.split('-');d=read(p);training.append(dict(condition=c,seed=int(s),**{k:v for k,v in d.items() if k not in ['metrics','episodes','failures']}))
  metrics.extend(dict(condition=c,seed=int(s),**x) for x in d['metrics']);episodes.extend(dict(condition=c,seed=int(s),**x) for x in d['episodes'])
 colors={c:plt.get_cmap('tab10')(i) for i,c in enumerate(CONDITIONS)}
 fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
 for ax,opp in zip(axes,['base','contender','heldout']):
  for c,s in frozen['runs']:
   steps=sorted({r['n'] for r in validation if r['condition']==c and r['seed']==s})
   ys=[summary([r for r in validation if r['condition']==c and r['seed']==s and r['n']==n and r['opponent']==opp and r['mode']=='sampled'])['win_fraction'] for n in steps]
   ax.plot(steps,ys,label=f'{c}/{s}',color=colors[c],linestyle=['-','--',':'][SEEDS.index(s)],alpha=.8)
  ax.set(title=opp,xlabel='Learner transitions',ylabel='Capped sampled win fraction',ylim=(0,1))
 if validation:axes[-1].legend(fontsize=6,ncol=2)
 savefig(fig,out,'01-validation-transitions')
 if live:return
 # Final results are only read after the externally committed selection exists.
 final=[]
 if (remote/'selection_verified').exists():final=load_rows(remote/'final')
 regression=load_rows(remote/'regression');context=load_rows(remote/'context')
 csvwrite(out/'validation-games.csv',validation);csvwrite(out/'final-games.csv',final);csvwrite(out/'regression-games.csv',regression);csvwrite(out/'historical-context.csv',context);csvwrite(out/'training-runs.csv',training);csvwrite(out/'ppo-metrics.csv',metrics);csvwrite(out/'training-episodes.csv',episodes)
 pooled=[];by_seed=[]
 for c,s in frozen['runs']:
  for mode in ['sampled','greedy']:
   for opponent in ['base','contender','heldout']:
    rs=[r for r in final if r['condition']==c and r['seed']==s and r['mode']==mode and r['opponent']==opponent]
    by_seed.append(dict(condition=c,seed=s,mode=mode,opponent=opponent,**summary(rs)))
 for c in frozen['conditions']:
  for mode in ['sampled','greedy']:
   for opponent in ['base','contender','heldout','heuristic_family']:
    opps=['base','contender'] if opponent=='heuristic_family' else [opponent]
    pooled.append(dict(condition=c,mode=mode,opponent=opponent,**summary([r for r in final if r['condition']==c and r['mode']==mode and r['opponent'] in opps])))
 csvwrite(out/'per-seed.csv',by_seed);csvwrite(out/'pooled.csv',pooled)
 definitions={'self_play_128':({'B':1,'A':-1},SEEDS),'capacity_static':({'C':1,'A':-1},SEEDS[:2]),'capacity_refresh':({'D':1,'B':-1},SEEDS[:2]),'structure_static':({'E':1,'C':-1},SEEDS[:2]),'structure_refresh':({'F':1,'D':-1},SEEDS[:2]),'teacher':({'G':1,'B':-1},SEEDS[:2]),'capacity_selfplay_interaction':({'D':1,'C':-1,'B':-1,'A':1},SEEDS[:2]),'structure_selfplay_interaction':({'F':1,'E':-1,'D':-1,'C':1},SEEDS[:2])}
 comparisons={}
 for name,(terms,seeds) in definitions.items():
  if not set(terms)<=set(frozen['conditions']):comparisons[name]=dict(inconclusive=True,reason='omitted before outcomes by calibrated scope');continue
  comparisons[name]={endpoint:contrast(final,terms,seeds,opps) for endpoint,opps in [('heuristic_family',['base','contender']),('heldout_neural',['heldout'])]}
 write(out/'paired-comparisons.json',comparisons)
 # Inference comparison preserves each training seed; re-label modes as a paired condition.
 inference={}
 for c in frozen['conditions']:
  rs=[dict(r,condition=r['mode'],mode='sampled') for r in final if r['condition']==c]
  ss=[s for cc,s in frozen['runs'] if cc==c]
  inference[c]={key:contrast(rs,{'sampled':1,'greedy':-1},ss,opps) for key,opps in [('heuristic_family',['base','contender']),('heldout_neural',['heldout'])]}
 write(out/'sampled-minus-greedy.json',inference)
 confirmation=[]
 if (remote/'confirmation.json').exists():
  d=read(remote/'confirmation.json');confirmation=d['rows'];csvwrite(out/'confirmation-games.csv',confirmation)
  selected=read(remote/'selection.json')['selected']
  prefix=[r for r in final if r['condition']==selected['condition'] and r['seed']==selected['seed'] and r['mode']==selected['mode'] and r['opponent'] in ['base','contender']]
  combined=prefix+confirmation;boards=sorted({r['board_seed'] for r in combined});result={'selected':selected,'summary':summary(combined),'boards':len(boards),'interpretation':'one validation-selected model; not an algorithm average'}
  rng=np.random.default_rng(12341)
  for opps in [['base'],['contender'],['base','contender']]:
   blocks=[np.mean([r['win'] for r in combined if r['board_seed']==b and r['opponent'] in opps]) for b in boards]
   result['+'.join(opps)]={'summary':summary([r for r in combined if r['opponent'] in opps]),'board_cluster_95':np.quantile(np.array(blocks)[rng.integers(len(blocks),size=(10000,len(blocks)))].mean(1),[.025,.975]).tolist()}
  write(out/'confirmation-summary.json',result)

 # 2: same validation endpoints against retained per-run time; teacher cost explicit.
 fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
 teach=read(ROOT/'teacher-manifest.json')['cpu_seconds'] if (ROOT/'teacher-manifest.json').exists() else read(root/'inputs/teacher-manifest.json')['cpu_seconds']
 for ax,include in zip(axes,[False,True]):
  for c,s in frozen['runs']:
   ms=[m for m in metrics if m['condition']==c and m['seed']==s];tr=next((t for t in training if t['condition']==c and t['seed']==s),{})
   steps=sorted({r['n'] for r in validation if r['condition']==c and r['seed']==s});xs=[];ys=[]
   for n in steps:
    prior=[m for m in ms if m['n']<=n];seconds=prior[-1]['elapsed'] if prior else 0
    if include and c=='G':seconds+=tr.get('teacher_seconds',0)+teach
    xs.append(seconds/60);ys.append(summary([r for r in validation if r['condition']==c and r['seed']==s and r['n']==n and r['mode']=='sampled' and r['opponent'] in ['base','contender']])['win_fraction'])
   ax.plot(xs,ys,label=f'{c}/{s}',color=colors[c],linestyle=['-','--',':'][SEEDS.index(s)])
  ax.set(title='Teacher-inclusive cost' if include else 'Retained PPO wall time',xlabel='Minutes per run (shared-GPU times do not add to GPU hours)',ylabel='Validation heuristic-family wins',ylim=(0,1))
 savefig(fig,out,'02-validation-time')
 # 3: architecture x schedule, separate opponent and inference panels; omitted cells masked.
 fig,axes=plt.subplots(2,3,figsize=(12,7));architecture=['MLP128','MLP512','Graph','Teacher128'];mapping=[['A','B'],['C','D'],['E','F'],[None,'G']]
 for row,mode in enumerate(['sampled','greedy']):
  for col,opp in enumerate(['base','contender','heldout']):
   mat=np.full((4,2),np.nan)
   for i in range(4):
    for j in range(2):
     match=next((x for x in pooled if x['condition']==mapping[i][j] and x['mode']==mode and x['opponent']==opp),None)
     if match and match['games']:mat[i,j]=match['win_fraction']
   ax=axes[row,col];ax.imshow(np.ma.masked_invalid(mat),vmin=0,vmax=1,cmap='viridis');ax.set(xticks=[0,1],xticklabels=['static','refresh'],yticks=range(4),yticklabels=architecture,title=f'{opp}: {mode}')
   for i in range(4):
    for j in range(2):ax.text(j,i,'omitted' if np.isnan(mat[i,j]) else f'{mat[i,j]*100:.1f}%',ha='center',va='center',color='black' if np.isnan(mat[i,j]) else 'white')
 savefig(fig,out,'03-factorial-heatmaps')
 # 4: uncertainty and raw seed contrasts, with omission visible in labels.
 fig,axes=plt.subplots(1,2,figsize=(13,6))
 for ax,endpoint in zip(axes,['heuristic_family','heldout_neural']):
  labels=[]
  for i,(name,d) in enumerate(comparisons.items()):
   labels.append(name);v=d.get(endpoint,{})
   if 'difference' not in v:ax.text(0,i,'not estimated',fontsize=8);continue
   lo,hi=v['crossed_seed_board_95'];ax.errorbar(v['difference']*100,i,xerr=[[100*(v['difference']-lo)],[100*(hi-v['difference'])]],fmt='o',color='black');ax.scatter([100*x for x in v['per_seed'].values()],[i+.15]*len(v['per_seed']),s=18,alpha=.6)
  ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Win difference (percentage points); crossed 95% interval',title=endpoint)
 savefig(fig,out,'04-paired-contrasts')
 # 5: equal-information strength/cost; each seed remains a visible point.
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for c,s in frozen['runs']:
  rs=[r for r in final if r['condition']==c and r['seed']==s and r['mode']=='sampled' and r['opponent'] in ['base','contender']];sm=summary(rs);tr=next((t for t in training if t['condition']==c and t['seed']==s),{})
  if not rs:continue
  for ax,x in zip(axes,[tr.get('parameters',0),sm['end_to_end_ms']]):ax.scatter(x,sm['win_fraction'],color=colors[c]);ax.annotate(f'{c}/{s}',(x,sm['win_fraction']),fontsize=6)
 axes[0].set(xlabel='Trainable parameters',ylabel='Sampled heuristic-family wins');axes[1].set(xlabel='Colab end-to-end decision latency (ms)',ylabel='Sampled heuristic-family wins')
 savefig(fig,out,'05-strength-cost')
 # 6: event attainment (never fraction complements), opportunities and timeouts.
 diagnostics=[]
 for c in frozen['conditions']:
  rs=[r for r in final if r['condition']==c and r['mode']=='sampled' and r['opponent'] in ['base','contender']]
  if not rs:continue
  d=dict(condition=c,games=len(rs))
  for event in ['first_city','third_settlement']:
   times=[r['diagnostic'][event] for r in rs if r['diagnostic'][event] is not None];d[event+'_never_fraction']=1-len(times)/len(rs);d[event+'_turns_conditional']=float(np.mean(times)) if times else None
  for family in ['PLAY_KNIGHT_CARD','MARITIME_TRADE','BUILD_SETTLEMENT','BUILD_CITY','immediate_win']:
   opportunities=sum(sum(v for k,v in r['diagnostic']['opportunities'].items() if k.endswith(family)) for r in rs);chosen=sum(sum(v for k,v in r['diagnostic']['chosen'].items() if k.endswith(family)) for r in rs);d[family+'_opportunities']=opportunities;d[family+'_chosen']=chosen;d[family+'_conversion']=chosen/opportunities if opportunities else None
  d['timeout_fraction']=sum(r['truncated'] for r in rs)/len(rs);diagnostics.append(d)
 csvwrite(out/'strategic-diagnostics.csv',diagnostics)
 fig,axes=plt.subplots(2,3,figsize=(13,7));keys=['first_city_never_fraction','third_settlement_never_fraction','timeout_fraction','immediate_win_conversion','PLAY_KNIGHT_CARD_conversion','MARITIME_TRADE_conversion']
 for ax,key in zip(axes.flat,keys):ax.bar([d['condition'] for d in diagnostics],[np.nan if d[key] is None else d[key] for d in diagnostics]);ax.set(title=key.replace('_',' '),ylim=(0,1),xlabel='Condition')
 savefig(fig,out,'06-expansion-timeouts')
 # 7: PPO diagnostics, undefined explained variance stays a gap.
 fig,axes=plt.subplots(2,3,figsize=(14,7))
 for ax,key in zip(axes.flat,['entropy','approx_kl','clip_fraction','explained_variance','value_loss','grad_norm_before_clip']):
  for c,s in frozen['runs']:
   ms=[m for m in metrics if m['condition']==c and m['seed']==s];ax.plot([m['n'] for m in ms],[np.nan if m[key] is None else m[key] for m in ms],color=colors[c],linestyle=['-','--',':'][SEEDS.index(s)],alpha=.65)
  ax.set(title=key,xlabel='Learner transitions')
 savefig(fig,out,'07-ppo-critic')
 # 8: reset and transition proportions; recorded own snapshot ages.
 shares=[]
 for t in training:
  for measure in ['episode_shares','transition_shares']:
   total=sum(t[measure].values())
   for identity,n in t[measure].items():shares.append(dict(condition=t['condition'],seed=t['seed'],measure=measure,opponent=identity,count=n,fraction=n/max(total,1),mean_snapshot_age=t['snapshot_age_sum'].get(identity,0)/max(t['transition_shares'].get(identity,0),1) if identity.startswith(tuple(CONDITIONS)) else None))
 csvwrite(out/'opponent-shares.csv',shares)
 fig,axes=plt.subplots(1,3,figsize=(15,4));labels=[f'{t["condition"]}/{t["seed"]}' for t in training]
 for ax,measure in zip(axes[:2],['episode_shares','transition_shares']):
  bottom=np.zeros(len(training))
  for group in ['base','weighted','heuristic','historical','own']:
   vals=[]
   for t in training:
    values=t[measure];den=max(1,sum(values.values()));vals.append(sum(n for identity,n in values.items() if (identity if identity in ['base','weighted','heuristic'] else 'historical' if identity.startswith(('parent','standard')) else 'own')==group)/den)
   ax.bar(labels,vals,bottom=bottom,label=group);bottom+=vals
  ax.tick_params(axis='x',rotation=90);ax.set(title=measure,ylim=(0,1));ax.legend(fontsize=7)
 for t in training:
  own=[r for r in shares if r['condition']==t['condition'] and r['seed']==t['seed'] and r['measure']=='transition_shares' and r['mean_snapshot_age'] is not None];axes[2].scatter([t['condition']]*len(own),[r['mean_snapshot_age'] for r in own],alpha=.5)
 axes[2].set(title='Own snapshot age when encountered',ylabel='Learner transitions since publication')
 savefig(fig,out,'08-opponent-pools')
 # 9: all allocations and copy events, without adding overlapping worker times.
 controller=read(root/'controller.json');telemetry=[]
 if (remote/'telemetry.csv').exists():telemetry=list(csv.DictReader((remote/'telemetry.csv').open()))
 fig,axes=plt.subplots(3,2,figsize=(12,10));t0=controller['started']
 if telemetry:
  xs=[(float(x['time'])-t0)/3600 for x in telemetry];axes[0,0].plot(xs,[float(x['gpu'].split(',')[1]) for x in telemetry]);axes[0,1].plot(xs,[float(x['cpu_pct_one_core']) for x in telemetry])
 axes[0,0].set(title='GPU utilization (%)');axes[0,1].set(title='Worker CPU utilization (100% = one core)')
 copies=[x for x in controller['events'] if x['kind']=='verified_generation'];axes[1,0].plot([(x['time']-t0)/3600 for x in copies],[(x['time']-(copies[i-1]['time'] if i else t0))/60 for i,x in enumerate(copies)],'.-');axes[1,0].set(title='Minutes between verified external generations')
 for i,a in enumerate(controller['allocations']):axes[1,1].barh(i,(a.get('stopped',time.time())-a['requested'])/3600,left=(a['requested']-t0)/3600)
 axes[1,1].set(title='Sequential runtime allocations',ylabel='Attempt')
 for t in training:
  ms=[m for m in metrics if m['condition']==t['condition'] and m['seed']==t['seed']];axes[2,0].plot([m['n'] for m in ms],[m['n']/max(1,m['elapsed']) for m in ms],color=colors[t['condition']],alpha=.6)
 axes[2,0].set(title='Retained learner transitions/s',xlabel='Learner transitions')
 if telemetry:axes[2,1].plot(xs,[float(x['rss_bytes'])/2**30 for x in telemetry])
 axes[2,1].set(title='Active-worker RSS (GiB)')
 for ax in [*axes[0],*axes[1],axes[2,1]]:ax.set_xlabel('Hours since first allocation request')
 savefig(fig,out,'09-utilization-recovery')
 for name in ['frozen.json','selection.json','confirmation.json','common-step.json','checkpoint-index.json','preflight.json','external-probe-returned.json','pilot-timings.json','evaluation-timing.json','experiment-state.json','installed-freeze.txt','telemetry.csv']:
  if (remote/name).exists():(out/name).write_bytes((remote/name).read_bytes())
 for name in ['controller.json','input-manifest.json','launch-integrity.json']:
  if (root/name).exists():(out/name).write_bytes((root/name).read_bytes())
 write(out/'coverage.json',dict(final_games=len(final),complete_board_games=sum(r['complete_board'] for r in final),validation_games=len(validation),regression_games=len(regression),context_games=len(context),planned=frozen.get('final_games'),training_runs=training))
 write(out/'timeouts.json',[r for r in final if r['truncated']])
 print(json.dumps(dict(conditions=frozen['conditions'],target=frozen['target'],final_games=len(final),comparisons=comparisons)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--live',action='store_true');a=p.parse_args();main(a.artifacts,a.output,a.live)

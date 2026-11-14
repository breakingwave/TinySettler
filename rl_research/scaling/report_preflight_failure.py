"""Rebuild the failed-preflight evidence without allocating compute or reading test outcomes."""
import argparse,csv,hashlib,json,shutil,datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
def csvfile(p,rows,fields):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main(first,repair,out):
 out.mkdir(parents=True,exist_ok=True);raw=out/'raw';raw.mkdir(exist_ok=True);charts=out/'charts';charts.mkdir(exist_ok=True)
 roots=[first,repair];checks=[];pilots=[];timings=[];telemetry=[]
 for attempt,root in enumerate(roots):
  tag='initial' if attempt==0 else 'repair1';dest=raw/tag;dest.mkdir(exist_ok=True)
  manifest=read(root/'input-manifest.json')
  for name,info in read(root/'launch-integrity.json').items():assert sha(root/name)==info['sha256']
  for name,h in manifest['source_hashes'].items():assert sha(root/'source'/name)==h,(root,name)
  for name,key in [('archive.pt','archive_sha256'),('teacher.pt','teacher_sha256')]:assert sha(root/'inputs'/name)==manifest[key]
  referenced=set();files=0
  generations=sorted((root/'generations').glob('*.json'))
  for generation in generations:
   for rel,info in read(generation)['files'].items():
    obj=root/'objects'/info['sha256'];assert sha(obj)==info['sha256'] and obj.stat().st_size==info['bytes'];referenced.add(info['sha256']);files+=1
  latest=read(generations[-1])
  for rel,info in latest['files'].items():assert sha(root/'retrieved'/rel)==info['sha256']
  checks.append(dict(attempt=tag,verified_generation_count=len(generations),generation_file_entries=files,unique_object_count=len(referenced),latest_files=len(latest['files']),source_hashes_verified=len(manifest['source_hashes']),source_archive_sha256=sha(root/'source.tar.gz'),inputs_archive_sha256=sha(root/'inputs.tar.gz')))
  for name in ['input-manifest.json','launch-integrity.json','controller.json','repair.json']:
   if (root/name).exists():shutil.copy2(root/name,dest/name)
  for name in ['preflight.json','pilot-timings.json','telemetry.csv','status.json','remote.log']:
   if (root/'retrieved'/name).exists():shutil.copy2(root/'retrieved'/name,dest/name)
  shutil.copytree(root/'retrieved/pilots',dest/'pilots',dirs_exist_ok=True)
  assert not (root/'retrieved/frozen.json').exists() and not (root/'retrieved/runs').exists()
  for p in sorted((root/'retrieved/pilots').glob('*/*/training.json')):
   x=read(p);c,mode=p.parent.parent.name.split('-');j=int(p.parent.name)
   recovery=p.parent/'recovery.json'
   pilots.append(dict(attempt=tag,condition=c,mode=mode,worker=j,seed=manifest['seeds'][j],transitions=x['n'],target=8192,complete=x['n']==8192,training_seconds=x['elapsed'],learner_decisions_per_second=x['n']/x['elapsed'],engine_actions=x['engine_actions'],engine_actions_per_second=x['engine_actions']/x['elapsed'],update_seconds=x['update_seconds'],recovery_passed=read(recovery)['bitwise_weights_rng_observation_equal'] if recovery.exists() else '',parameters=x['parameters']))
  for p in sorted((root/'retrieved/pilots').glob('*/timing.json')):
   c,mode=p.parent.name.split('-');x=read(p);timings.append(dict(attempt=tag,condition=c,mode=mode,**x))
  with (root/'retrieved/telemetry.csv').open() as f:
   for row in csv.DictReader(f):
    gpu=row['gpu'].split(',');telemetry.append(dict(attempt=tag,time=float(row['time']),gpu_utilization=float(gpu[1]),gpu_memory_mib=float(gpu[2]),cpu_pct_one_core=float(row['cpu_pct_one_core']),rss_bytes=float(row['rss_bytes'])))
 state=read(repair/'controller.json');assert state['all_terminated'] and all(a['terminated'] for a in state['allocations']);assert not state['complete'];assert state['elapsed_upper_bound']<50400
 alloc=[dict(session=a['session'],request_utc=datetime.datetime.fromtimestamp(a['requested'],datetime.timezone.utc).isoformat(),stop_utc=datetime.datetime.fromtimestamp(a['stopped'],datetime.timezone.utc).isoformat(),charged_upper_bound_seconds=a['stopped']-a['requested'],terminated=a['terminated']) for a in state['allocations']]
 resource=dict(original_started=state['started'],original_deadline=state['deadline'],end=state['ended'],wall_clock_seconds=state['elapsed_upper_bound'],allocated_upper_bound_seconds=sum(x['charged_upper_bound_seconds'] for x in alloc),hard_budget_seconds=50400,allocations=alloc,fault_replacements=state['fault_replacements'],planned_handoffs=state['planned_handoffs'],all_terminated=True,experimental_transitions=0,pilot_transitions=sum(x['transitions'] for x in pilots),scope_frozen=False,final_test_games=0)
 write(out/'resource-summary.json',resource);write(out/'artifact-verification.json',checks)
 csvfile(out/'pilot-workers.csv',pilots,list(pilots[0]));csvfile(out/'pilot-timings.csv',timings,list(timings[0]));csvfile(out/'runtime-ledger.csv',alloc,list(alloc[0]));csvfile(out/'telemetry.csv',telemetry,list(telemetry[0]))
 comparisons=[dict(condition=c,seed=s,status='not_started_preflight_failed',common_target='',retained_transitions=0,validation_games=0,final_games=0) for s in [400009,400031,400033] for c in 'ABCDEFG' if s!=400033 or c in 'AB']
 csvfile(out/'comparison-status.csv',comparisons,list(comparisons[0]))
 archive=read(first/'inputs/archive-manifest.json');write(out/'archive-provenance.json',archive)
 write(out/'teacher-provenance.json',read(first/'inputs/teacher-manifest.json'))
 # Only measured preflight quantities: no empty strength curves masquerading as results.
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 def save(fig,name):
  fig.savefig(charts/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(charts/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
 for ax,tag in zip(axes,['initial','repair1']):
  for offset,mode,color in [(-.18,'single','#4477aa'),(.18,'dual','#ee7733')]:
   vals=[next(x['aggregate_rate'] for x in timings if x['attempt']==tag and x['condition']==c and x['mode']==mode) for c in 'ABCD'];ax.bar(np.arange(4)+offset,vals,width=.35,label=mode,color=color)
  ax.set_xticks(range(4),list('ABCD'));ax.set_title(tag+' preflight only');ax.set_xlabel('Condition');ax.legend();ax.set_ylim(0,230)
 axes[0].set_ylabel('Aggregate learner decisions/s\n(includes process/checkpoint overhead)');save(fig,'01-pilot-throughput')
 fig,ax=plt.subplots(figsize=(10,2.8));t0=state['started']
 for i,a in enumerate(state['allocations']):ax.broken_barh([((a['requested']-t0)/60,(a['stopped']-a['requested'])/60)],(i-.3,.6),facecolors=['#4477aa','#ee7733'][i])
 for label,minutes in [('Planned preflight limit',45),('Preflight + recovery allowance',75)]:ax.axvline(minutes,color='gray',ls='--');ax.text(minutes,1.45,label,rotation=0,ha='right',fontsize=8)
 ax.set_yticks([0,1],['Initial T4','Replacement T4']);ax.set_xlabel('Minutes since original allocation request');ax.set_title('Both runtimes terminated; original hard deadline was 840 minutes');ax.set_ylim(-.6,2);save(fig,'02-allocation-timeline')
 fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
 for tag,color in [('initial','#4477aa'),('repair1','#ee7733')]:
  rows=[x for x in telemetry if x['attempt']==tag];x=[(r['time']-t0)/60 for r in rows]
  for ax,key,scale in zip(axes,['gpu_utilization','cpu_pct_one_core','rss_bytes'],[1,1,1e-9]):ax.plot(x,[r[key]*scale for r in rows],label=tag,color=color)
 for ax,label in zip(axes,['GPU utilization (%)','Worker CPU (% of one core)','Worker RSS (GB)']):ax.set_ylabel(label);ax.legend()
 axes[0].set_title('Preflight telemetry only; gaps include setup and CLI delays');axes[-1].set_xlabel('Minutes since original allocation request');save(fig,'03-preflight-utilization')
 write(out/'chart-availability.json',dict(provided=['pilot-throughput','allocation-timeline','preflight-utilization'],unavailable=['validation versus transitions/time','final strength heatmaps','paired strength contrasts','strength versus size/latency','final action-use diagnostics','experimental PPO curves','experimental opponent-pool composition'],reason='Calibration failed before scope freeze or experimental initialization; no valid experimental data for these charts.'))
 write(out/'source-hashes.json',dict(executed={tag:read(root/'input-manifest.json')['source_hashes'] for tag,root in zip(['initial','repair1'],roots)},current_repository={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},note='Repository controller received a post-run fatal-resume guard and all_terminated state correction; archived executed sources are immutable.'))
 print(json.dumps(resource,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--repair',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.first,a.repair,a.output)

"""Read-only status from verified local snapshots; no Colab credentials or allocation."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);a=p.parse_args()
c=json.loads((a.artifacts/'controller.json').read_text());root=a.artifacts/'retrieved'
s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'phase':'setup'}
result=dict(status=s,allocated_minutes=(c.get('ended',time.time())-c['started'])/60,remaining_minutes=(c['deadline']-time.time())/60,last_verified_copy_seconds_ago=time.time()-c.get('last_verified_snapshot',c['started']),termination_verified=c.get('termination_verified',False),controller_exception=c.get('exception'))
if s.get('run'):
    folder=root/s['run'];progress=folder/'progress.json'
    if progress.exists():
        saved_progress=json.loads(progress.read_text())
        if saved_progress['n']>=s.get('step',0):result['progress']=saved_progress
    if s['phase'] in ('validation','final','secondary'):
        e=root/'evaluations'/s['phase']/s['run']/f"{s['step']}.json"
        if e.exists():result['games_completed']=len(json.loads(e.read_text())['rows'])
print(json.dumps(result,indent=2))

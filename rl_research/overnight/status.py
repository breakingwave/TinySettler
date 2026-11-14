"""Read-only local status for a detached experiment controller."""
import argparse,json,time,re
from pathlib import Path

def status(path):
    c=json.loads((path/'controller.json').read_text())
    sfile=path/'retrieved/status.json'; s=json.loads(sfile.read_text()) if sfile.exists() else {'phase':'preflight'}
    progress=path/'retrieved'/s.get('run','')/'progress.json'
    p=json.loads(progress.read_text()) if progress.exists() else {}
    selections=path/'retrieved/selections.json'
    try: alive=b'rl_research/overnight/controller.py' in Path(f"/proc/{c['controller_pid']}/cmdline").read_bytes()
    except (FileNotFoundError,KeyError): alive=False
    result=dict(phase=s['phase'],run=s.get('run'),checkpoint=s.get('n'),periodic_checkpoint=p.get('n'),target=s.get('target'),selected_runs=len(json.loads(selections.read_text())) if selections.exists() else 0,controller_alive=alive,controller_ended='ended' in c,termination_verified=c.get('termination_verified',False),seconds_since_verified_copy=round(time.time()-c.get('last_verified_snapshot',c['started'])))
    plan=path/'retrieved/final_plan.json'
    if plan.exists():
        chosen=json.loads(selections.read_text())
        finished=[x['run'] for x in chosen if (path/'retrieved'/x['run']/'final.json').exists()]
        result['final_policies_evaluated']=len(finished)
        result['final_board_target']=json.loads(plan.read_text())['boards']
        if s['phase'] not in ('complete','failed'):
            result['phase']='final_evaluation'
            result['run']=next((x['run'] for x in chosen if x['run'] not in finished),'finalizing')
    if not alive and 'ended' not in c:
        log=path/'reattached-controller.log'
        result['last_log']=re.sub(r'([?&](?:colab-runtime-proxy-token|access_token|token)=)[^&\s)]+',r'\1<redacted>',log.read_text()[-2000:] if log.exists() else '')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('artifacts',type=Path); a=p.parse_args(); print(json.dumps(status(a.artifacts)))

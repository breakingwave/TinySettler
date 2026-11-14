"""Wait for an hourly review or a completion/fatal event; never reads run outcomes."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--hour',type=int,required=True);a=p.parse_args()
while True:
 try:
  s=json.loads((a.artifacts/'controller.json').read_text());target=s['started']+3600*a.hour
  if s.get('ended') or s.get('fatal'):
   print(json.dumps(dict(event='complete' if s.get('complete') else 'fatal',state=str(a.artifacts/'controller.json'))),flush=True);break
  try:os.kill(s['controller_pid'],0)
  except ProcessLookupError:
   print(json.dumps(dict(event='fatal',reason='supervisor process absent; independent watchdog still applies')),flush=True);break
  if time.time()>=target:
   print(json.dumps(dict(event='hourly',hour=a.hour,status_file=str(a.artifacts/'hourly-status.json'),chart=str(a.artifacts/'live-charts/01-validation-transitions.png'))),flush=True);break
 except FileNotFoundError:pass
 time.sleep(10)

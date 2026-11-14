"""Persistent single-T4 supervisor. One original deadline, deterministic bounded recovery."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from sc_transfer import make,receive,sha
CLI='/root/.local/bin/colab'
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
def read(path):return json.loads(Path(path).read_text())
class Supervisor:
 def __init__(self,root,session,resume):
  self.root=root;self.path=root/'controller.json'
  if self.path.exists():
   assert resume;self.s=read(self.path);assert self.s['base']==session
  else:
   assert not resume;now=time.time();self.s=dict(base=session,session=session,started=now,deadline=now+50400,planned_handoffs=0,fault_replacements=0,allocations=[],events=[],complete=False)
  self.s['controller_pid']=os.getpid();self.save();self.session=self.s['session'];self.next_copy=0;self.connection_since=None
 def save(self):write(self.path,self.s)
 def event(self,kind,**kw):self.s['events'].append(dict(time=time.time(),kind=kind,**kw));self.save()
 def cli(self,*args,code=None,timeout=180,attempts=3):
  for attempt in range(attempts):
   cleanup=args[0] in ['stop','sessions','usage'];remaining=self.s['deadline']-time.time()
   if not cleanup and remaining<120:raise TimeoutError('original allocation deadline')
   try:
    r=subprocess.run([CLI,'--auth','oauth2',*args],input=code,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout if cleanup else min(timeout,max(1,remaining-100)))
    output=re.sub(r'([?&](?:colab-runtime-proxy-token|access_token|token)=)[^&\s)]+',r'\1<redacted>',r.stdout)
    with (self.root/'controller.log').open('a') as f:f.write(output+'\n')
    if r.returncode:raise RuntimeError(output[-3000:])
    return re.sub(r'\x1b\[[0-9;]*m','',output)
   except (subprocess.TimeoutExpired,RuntimeError) as exc:
    if isinstance(exc,RuntimeError) and any(t in str(exc).lower() for t in ['quota','insufficient','balance','permission denied']):raise
    if attempt==attempts-1:raise
    self.event('bounded_cli_retry',command=args[0],attempt=attempt+1,error=type(exc).__name__);time.sleep([10,30][attempt])
 def remote(self,code,timeout=120):
  result=self.cli('exec','-s',self.session,'--timeout',str(timeout),code=code+"\nprint('SCALING_OK')\n",timeout=timeout+45)
  if not any(line.strip()=='SCALING_OK' for line in result.splitlines()):raise RuntimeError('remote Python failed: '+result[-2000:])
  return result
 def upload(self,path,dest):
  parts=[]
  with Path(path).open('rb') as inp:
   while data:=inp.read(6*1024*1024):
    local=self.root/'upload-part';local.write_bytes(data);part=dest+f'.part{len(parts)}';self.cli('upload','-s',self.session,str(local),part,timeout=240);parts.append(part)
  self.remote(f"import pathlib,hashlib,shutil\nwith open({dest!r},'wb') as out:\n for part in {parts!r}:\n  with open(part,'rb') as inp:shutil.copyfileobj(inp,out)\nwith open({dest!r},'rb') as inp:assert hashlib.file_digest(inp,'sha256').hexdigest()=={sha(path)!r}\nfor part in {parts!r}:pathlib.Path(part).unlink()\n")
  (self.root/'upload-part').unlink(missing_ok=True)
 def watchdog(self):
  # Independent process; only known experiment sessions, repeated stop then absence verification.
  script="import sys,time,subprocess,json;from pathlib import Path;p=Path(sys.argv[1]);s=json.loads(p.read_text());time.sleep(max(0,s['deadline']-120-time.time()));s=json.loads(p.read_text());[subprocess.run([sys.argv[2],'--auth','oauth2','stop','-s',a['session']],timeout=100,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for a in s['allocations'] if not a.get('terminated')]"
  p=subprocess.Popen([sys.executable,'-c',script,str(self.path),CLI],stdout=(self.root/'watchdog.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);self.s['watchdog_pid']=p.pid;self.save()
 def allocate(self):
  if time.time()>=self.s['deadline']-3600:raise RuntimeError('insufficient remaining setup/retrieval budget')
  self.s['session']=self.session;self.s['allocations'].append(dict(session=self.session,requested=time.time(),terminated=False));self.s['all_terminated']=False;self.save()
  self.cli('new','-s',self.session,'--gpu','T4',timeout=300,attempts=1);self.s['allocations'][-1]['ready']=time.time();self.save()
  self.upload(self.root/'source.tar.gz','/content/sc-source.tar.gz');self.upload(self.root/'inputs.tar.gz','/content/sc-inputs.tar.gz')
  self.remote("import pathlib,tarfile\nfor d in ['experiment','inputs','artifacts']:pathlib.Path('/content/'+d).mkdir(exist_ok=True)\nfor f,d in [('sc-source','experiment'),('sc-inputs','inputs')]:\n with tarfile.open('/content/'+f+'.tar.gz') as t:t.extractall('/content/'+d,filter='data')\n")
  # The notebook kernel only launches/polls. Retrying a CLI read never reruns pip.
  self.remote("import subprocess,pathlib,os\npid=pathlib.Path('/content/setup.pid')\nif not pid.exists():\n log=open('/content/artifacts/setup.log','a');p=subprocess.Popen(['python','/content/experiment/sc_setup.py'],stdout=log,stderr=log,start_new_session=True);pid.write_text(str(p.pid))\n",timeout=30)
  setup_limit=min(time.time()+900,self.s['deadline']-3600)
  while True:
   if time.time()>=setup_limit:raise TimeoutError('bounded asynchronous setup expired')
   try:
    result=self.remote("import pathlib,json\np=pathlib.Path('/content/artifacts/setup-result.json')\nprint('SETUP='+p.read_text() if p.exists() else 'SETUP=null')\n",timeout=30)
    value=json.loads(next(x.split('=',1)[1] for x in result.splitlines() if x.startswith('SETUP=')))
   except (subprocess.TimeoutExpired,RuntimeError):time.sleep(15);continue
   if value is not None:
    if not value['complete']:raise RuntimeError('dependency setup failed: '+str(value))
    break
   time.sleep(15)
  if (self.root/'retrieved').exists():
   with tarfile.open(self.root/'restore.tar.gz','w:gz',compresslevel=1) as tar:
    for p in (self.root/'retrieved').rglob('*'):
     if p.is_file() and p.name not in ['pause_requested','status.json','active-pids.json','heartbeat.json']:tar.add(p,arcname=str(p.relative_to(self.root/'retrieved')))
   self.upload(self.root/'restore.tar.gz','/content/restore.tar.gz')
   self.remote("import tarfile\nwith tarfile.open('/content/restore.tar.gz') as tar:tar.extractall('/content/artifacts',filter='data')\n")
  self.remote("import subprocess,torch,pathlib,os,time\nold=pathlib.Path('/content/artifacts/preflight.json')\nif old.exists():old.rename(old.with_name('preflight-'+str(time.time_ns())+'.json'))\nassert 'T4' in subprocess.check_output(['nvidia-smi'],text=True) and torch.cuda.is_available()\n"+f"log=open('/content/artifacts/remote.log','a');p=subprocess.Popen(['python','sc_remote.py','--out','/content/artifacts','--deadline',{str(self.s['deadline'])!r}],cwd='/content/experiment',stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONHASHSEED='0',SCALING_INPUTS='/content/inputs',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'));pathlib.Path('/content/worker.pid').write_text(str(p.pid))\n")
  self.event('worker_launched',session=self.session);self.next_copy=0
 def stop(self):
  last=self.s['allocations'][-1] if self.s['allocations'] else None
  if not last or last.get('terminated'):return
  try:self.cli('stop','-s',self.session,timeout=90,attempts=3)
  except Exception as exc:self.event('stop_command_error_verify_absence',error=type(exc).__name__)
  listing=self.cli('sessions',timeout=90);(self.root/f'sessions-after-{self.session}.txt').write_text(listing)
  assert self.session not in listing,'dedicated runtime remains listed'
  last['terminated']=True;last['stopped']=time.time();self.save()
 def copy(self):
  started=time.time()
  result=self.remote("import sys,json\nsys.path.insert(0,'/content/experiment')\nfrom sc_transfer import make\nprint('GENERATION='+json.dumps(make('/content/artifacts','/content/transfers')))\n",timeout=240)
  info=json.loads(next(x.split('=',1)[1] for x in result.splitlines() if x.startswith('GENERATION=')))
  self.cli('download','-s',self.session,'/content/transfers/transfer.tar.gz',str(self.root/'transfer.partial'),timeout=600)
  generation=receive(self.root/'transfer.partial',self.root,info['sha256'])
  self.s['last_verified_copy']=time.time();self.s['latest_generation']=generation['name'];self.save()
  # Acknowledgement advances only after every external object hash passed.
  self.remote("import pathlib,json\n"+f"pathlib.Path('/content/transfers/ack.json').write_text(json.dumps({generation['hashes']!r}))\n")
  self.next_copy=started+600
  try:subprocess.run([sys.executable,str(self.root/'source/sc_report.py'),'--artifacts',str(self.root),'--output',str(self.root/'live-charts'),'--live'],stdout=(self.root/'live-chart.log').open('a'),stderr=subprocess.STDOUT,timeout=60,env=dict(os.environ,SCALING_INPUTS=str(self.root/'inputs')))
  except Exception:pass
  self.event('verified_generation',files=generation['files'],bytes=info['bytes'],generation=generation['name'])
  return generation
 def health(self):
  code="import pathlib,json,os,time\np=pathlib.Path('/content/worker.pid');alive=False\nif p.exists():\n try:os.kill(int(p.read_text()),0);alive=True\n except ProcessLookupError:pass\ns=pathlib.Path('/content/artifacts/status.json');status=json.loads(s.read_text()) if s.exists() else {'phase':'setup'}\nprint('HEALTH='+json.dumps({'alive':alive,'status':status,'time':time.time()}))\n"
  result=self.remote(code);return json.loads(next(x.split('=',1)[1] for x in result.splitlines() if x.startswith('HEALTH=')))
 def dashboard(self,health):
  retrieved=self.root/'retrieved';runs=[]
  for p in sorted((retrieved/'runs').glob('*/progress.json')):runs.append(dict(run=p.parent.name,**read(p)))
  validations=[]
  for p in sorted((retrieved/'validation').glob('*.json')):
   v=read(p)
   if not v.get('complete'):continue
   for o in ['base','contender','heldout']:
    rr=[x for x in v['rows'] if x['opponent']==o and x['mode']=='sampled'];validations.append(dict(run=p.stem,opponent=o,games=len(rr),wins=sum(x['win'] for x in rr),truncations=sum(x['truncated'] for x in rr)))
  compact=dict(time=time.time(),deadline=self.s['deadline'],remaining_seconds=self.s['deadline']-time.time(),health=health,completed_or_pending=runs,latest_validation=validations[-24:],last_verified_copy=self.s.get('last_verified_copy'),copy_age=time.time()-self.s.get('last_verified_copy',self.s['started']),recoveries=dict(planned=self.s['planned_handoffs'],fault=self.s['fault_replacements']))
  latest={}
  for v in validations:
   key='/'.join(v['run'].split('-')[:2]+[v['opponent']])
   if key not in latest or int(v['run'].split('-')[2])>int(latest[key]['run'].split('-')[2]):latest[key]=v
  compact['latest_validation']=list(latest.values())
  write(self.root/'hourly-status.json',compact)
 def recovery(self,planned=False):
  key='planned_handoffs' if planned else 'fault_replacements'
  limit=1 if planned else self.s.get('max_fault_replacements',1)
  if self.s[key]>=limit:raise RuntimeError('recovery beyond frozen allowance')
  self.stop();self.s[key]+=1;self.session=self.s['base']+f'-r{len(self.s["allocations"])}';self.save();self.allocate()
 def run(self):
  if self.s.get('complete'):
   print('Experiment already complete; no runtime allocated.',flush=True);return
  if self.s.get('fatal'):
   raise RuntimeError('Fatal experiment is closed; --resume cannot bypass the recovery allowance')
  self.watchdog()
  try:
   if not self.s['allocations']:self.allocate()
   elif self.s['allocations'][-1].get('terminated'):self.recovery()
   while time.time()<self.s['deadline']-180:
    try:health=self.health();self.connection_since=None
    except Exception as exc:
     self.connection_since=self.connection_since or time.time();self.event('connection_unavailable_worker_not_assumed_dead',error=type(exc).__name__)
     if time.time()-self.connection_since>900:self.recovery();self.connection_since=None
     time.sleep(60);continue
    phase=health['status']['phase'];self.dashboard(health)
    if phase=='recoverable_failure':
     self.copy();self.recovery();continue
    if phase=='fatal':
     self.copy();raise RuntimeError('fatal worker integrity/stage failure: '+str(health['status'])[-1800:])
    if not health['alive'] and phase not in ['complete','paused']:
     self.copy();self.recovery();continue
    if time.time()>=self.next_copy or phase in ['complete','paused','await_external_recovery','await_frozen_copy','await_selection_copy']:
     try:self.copy()
     except Exception as exc:
      self.event('copy_failed_worker_kept_alive',error=repr(exc)[-500:]);self.next_copy=time.time()+60
      if time.time()-self.s.get('last_verified_copy',self.s['started'])>1800:raise RuntimeError('external copy stale beyond 30-minute allowance')
      time.sleep(60);continue
    if phase=='await_external_recovery':
     checkpoint=self.root/'retrieved/external-probe.pt';self.upload(checkpoint,'/content/returned-probe.pt')
     self.remote("import subprocess,json,hashlib,os,pathlib\nwith open('/content/returned-probe.pt','rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=="+repr(sha(checkpoint))+"\nsubprocess.run(['python','sc_checks.py','--probe','/content/returned-probe.pt','--out','/content/artifacts/external-probe-returned.json','--device','cuda'],cwd='/content/experiment',env=dict(os.environ,PYTHONHASHSEED='0',SCALING_INPUTS='/content/inputs',CUBLAS_WORKSPACE_CONFIG=':4096:8'),check=True)\na=json.loads(pathlib.Path('/content/artifacts/external-probe-expected.json').read_text());b=json.loads(pathlib.Path('/content/artifacts/external-probe-returned.json').read_text());assert a==b\npathlib.Path('/content/artifacts/external_recovery_verified').write_text('verified external bytes, frozen opponent, fresh-process CUDA continuation')\n",timeout=240)
    if phase=='await_frozen_copy':self.remote("from pathlib import Path\nPath('/content/artifacts/frozen_verified').write_text('verified external frozen manifest')\n")
    if phase=='await_selection_copy':
     selection_hash=sha(self.root/'retrieved/selection.json');self.s['selection_committed_at']=time.time();self.s['selection_sha256']=selection_hash;self.save()
     self.remote("from pathlib import Path\nimport hashlib\nassert hashlib.sha256(Path('/content/artifacts/selection.json').read_bytes()).hexdigest()=="+repr(selection_hash)+"\nPath('/content/artifacts/selection_verified').write_text("+repr(selection_hash)+")\n")
    if phase=='complete':self.s['complete']=True;self.s['result']=health['status'];self.save();break
    if time.time()-self.s['started']>=36000 and self.s['planned_handoffs']==0 and self.s['deadline']-time.time()>3600:
     self.remote("from pathlib import Path\nPath('/content/artifacts/pause_requested').write_text('planned lifetime handoff')\n")
     # Manager acknowledges only after checkpoints are closed; safe even during evaluation.
     if phase=='paused':self.recovery(planned=True)
    time.sleep(60)
  except BaseException as exc:
   self.s['fatal']=repr(exc);self.save()
  finally:
   if self.s.get('fatal'):
    try:self.copy()
    except Exception as exc:self.event('final_evidence_copy_failed',error=repr(exc)[-500:])
   try:self.stop()
   except BaseException as exc:self.s['stop_failure']=repr(exc);self.save()
   self.s['ended']=time.time();self.s['elapsed_upper_bound']=self.s['ended']-self.s['started'];self.s['all_terminated']=all(a.get('terminated') for a in self.s['allocations']);self.save()
   if self.s['all_terminated']:
    try:os.kill(self.s['watchdog_pid'],signal.SIGTERM)
    except ProcessLookupError:pass
   self.dashboard(dict(status=self.s.get('result',{'phase':'fatal'}),alive=False))
   print(json.dumps({'event':'complete' if self.s.get('complete') else 'fatal','state':str(self.path),'terminated':self.s['all_terminated']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();Supervisor(a.artifacts,a.session,a.resume).run()

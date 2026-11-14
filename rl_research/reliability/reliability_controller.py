"""Dedicated Colab session, immutable five-hour deadline, verified snapshots, bounded recovery."""
import argparse,hashlib,json,os,re,signal,subprocess,tarfile,time
from pathlib import Path
CLI='/root/.local/bin/colab'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def main(a):
    root=a.artifacts;path=root/'controller.json'
    if path.exists():
        assert a.resume,'existing allocation requires --resume'
        state=json.loads(path.read_text());assert state['session_base']==a.session
    else:
        assert not a.resume and not a.replace
        now=time.time();state=dict(session_base=a.session,session=a.session,started=now,deadline=now+18000,replacements=0,transfer_retries=0,attachments=[])
    assert state['deadline']-time.time()>1200,'original budget has insufficient retrieval reserve'
    if a.replace:
        assert state['replacements']<1,'runtime replacement limit reached'
        assert state.get('termination_verified'),'previous runtime termination must be verified first'
        state.setdefault('previous_attempts',[]).append({k:state.get(k) for k in ['session','ended','exception','result','termination_verified']})
        for key in ['ended','exception','result','stop_error','stop_succeeded']:state.pop(key,None)
        state['replacements']+=1;state['session']=a.session+'-r1';state['termination_verified']=False
    session=state['session'];state['controller_pid']=os.getpid();state['attachments'].append(dict(time=time.time(),pid=os.getpid(),controller_source_sha256=sha(__file__)))
    def save():write(path,state)
    save()
    def cli(*args,code=None,timeout=120):
        cleanup=args[0] in ('stop','sessions')
        remaining=state['deadline']-time.time()
        if not cleanup and remaining<120:raise TimeoutError('allocation deadline')
        r=subprocess.run([CLI,'--auth','oauth2',*args],input=code,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout if cleanup else min(timeout,max(1,remaining-90)))
        output=re.sub(r'([?&](?:colab-runtime-proxy-token|access_token|token)=)[^&\s)]+',r'\1<redacted>',r.stdout)
        with (root/'controller.log').open('a') as f:f.write(output+'\n')
        if r.returncode:raise RuntimeError(output[-2000:])
        return output
    def remote(code,timeout=90):
        marker='RELIABILITY_REMOTE_COMPLETED'
        for attempt in range(2):
            try:
                output=cli('exec','-s',session,'--timeout',str(timeout),code=code+'\nprint('+repr(marker)+')\n',timeout=timeout+45);break
            except subprocess.TimeoutExpired:
                if attempt or state.get('exec_retries',0)>=2:raise
                state['exec_retries']=state.get('exec_retries',0)+1;save()
        clean=re.sub(r'\x1b\[[0-9;]*m','',output)
        if not any(x.strip()==marker for x in clean.splitlines()):raise RuntimeError('remote Python failed; see sanitized controller.log')
        return clean
    def upload(p,dest):
        parts=[]
        with Path(p).open('rb') as f:
            while data:=f.read(8*1024*1024):
                tmp=root/'upload-part';tmp.write_bytes(data);part=dest+f'.part{len(parts)}'
                cli('upload','-s',session,str(tmp),part,timeout=180);parts.append(part)
        remote('import pathlib,hashlib,shutil\n'+f'with open({dest!r},"wb") as out:\n for part in {parts!r}:\n  with open(part,"rb") as inp:shutil.copyfileobj(inp,out)\n'+f'with open({dest!r},"rb") as inp:assert hashlib.file_digest(inp,"sha256").hexdigest()=={sha(p)!r}\n')
        (root/'upload-part').unlink(missing_ok=True)
    # An independent process survives controller failure and stops only this dedicated runtime.
    watchdog_code='import subprocess,time,sys;time.sleep(max(0,float(sys.argv[1])-time.time()));subprocess.run(sys.argv[2:],timeout=110)'
    watchdog=subprocess.Popen([a.python,'-c',watchdog_code,str(state['deadline']-120),CLI,'--auth','oauth2','stop','-s',session],stdout=(root/'watchdog.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    state.setdefault('watchdog_pids',[]).append(watchdog.pid);save()
    try:
        if not a.resume or a.replace:
            cli('new','-s',session,'--gpu','T4',timeout=300);state.setdefault('allocations',[]).append(dict(session=session,allocated=time.time()));save()
            upload(root/'source.tar.gz','/content/source.tar.gz')
            upload(root/'historical.tar.gz','/content/historical.tar.gz')
            remote("import tarfile,pathlib\nfor folder in ['experiment','historical','artifacts']:pathlib.Path('/content/'+folder).mkdir(exist_ok=True)\nwith tarfile.open('/content/source.tar.gz') as t:t.extractall('/content/experiment',filter='data')\nwith tarfile.open('/content/historical.tar.gz') as t:t.extractall('/content/historical',filter='data')\n")
            remote("import subprocess\nwith open('/content/artifacts/install.log','w') as f:subprocess.run(['python','-m','pip','install','-r','/content/experiment/requirements-lock.txt'],stdout=f,stderr=subprocess.STDOUT,check=True)\nwith open('/content/artifacts/installed-freeze.txt','w') as f:subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True)\n",timeout=900)
            if a.replace:
                for filename in ['preflight.json','status.json','telemetry.csv','cpu-telemetry.csv']:
                    old=root/'retrieved'/filename
                    if old.exists():old.rename(old.with_name(old.stem+'-before-replacement1'+old.suffix))
                with tarfile.open(root/'restore.tar.gz','w:gz',compresslevel=1) as t:
                    for p in (root/'retrieved').rglob('*'):
                        if p.is_file():t.add(p,arcname=str(p.relative_to(root/'retrieved')))
                upload(root/'restore.tar.gz','/content/restore.tar.gz')
                remote("import tarfile\nwith tarfile.open('/content/restore.tar.gz') as t:t.extractall('/content/artifacts',filter='data')\n")
            remote("import subprocess,torch\nassert 'T4' in subprocess.check_output(['nvidia-smi'],text=True) and torch.cuda.is_available()\n"+f"import pathlib,os\npidfile=pathlib.Path('/content/worker.pid')\nalive=False\nif pidfile.exists():\n try:os.kill(int(pidfile.read_text()),0);alive=True\n except ProcessLookupError:pass\nif not alive:\n log=open('/content/artifacts/remote.log','a');p=subprocess.Popen(['python','reliability_remote.py','--out','/content/artifacts','--historical','/content/historical','--deadline',{str(state['deadline'])!r}],cwd='/content/experiment',stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pidfile.write_text(str(p.pid))\nprint(pidfile.read_text())\n")
        while time.time()<state['deadline']-180:
            for attempt in range(2):
                try:
                    result=remote("import tarfile,pathlib,hashlib,time\nsnapshot_started=time.time()\nroot=pathlib.Path('/content/artifacts')\nwith tarfile.open('/content/snapshot.tar','w') as t:\n for p in root.rglob('*'):\n  if p.is_file() and not p.name.endswith('.tmp') and p.stat().st_mtime>globals().get('snapshot_cutoff',0):t.add(p,arcname=str(p.relative_to(root)))\nwith open('/content/snapshot.tar','rb') as f:print('DIGEST='+hashlib.file_digest(f,'sha256').hexdigest())\n")
                    digest=next(x.split('=',1)[1].strip() for x in result.splitlines() if x.startswith('DIGEST='))
                    cli('download','-s',session,'/content/snapshot.tar',str(root/'snapshot.partial'),timeout=420)
                    assert sha(root/'snapshot.partial')==digest
                    os.replace(root/'snapshot.partial',root/'snapshot.tar')
                    with tarfile.open(root/'snapshot.tar') as t:t.extractall(root/'retrieved',filter='data')
                    remote('snapshot_cutoff=snapshot_started\n');break
                except Exception:
                    if attempt or state['transfer_retries']>=2:raise
                    state['transfer_retries']+=1;save()
            state['last_verified_snapshot']=time.time();state['snapshot_sha256']=digest;save()
            sp=root/'retrieved/status.json';status=json.loads(sp.read_text()) if sp.exists() else dict(phase='setup')
            print(json.dumps(dict(time=time.time(),status=status)),flush=True)
            if status['phase']=='awaiting_external_recovery':
                # Inspect the external bytes with the archived implementation; then return those exact bytes to CUDA.
                script="import sys,torch;sys.path.insert(0,sys.argv[1]);from train import source_hashes;from reliability_common import DiagnosticTrainer\ns=torch.load(sys.argv[2],map_location='cpu',weights_only=False);assert s['source_hashes']==source_hashes();assert s['trainer'].n==4096;print('external checkpoint verified')"
                subprocess.run([a.python,'-c',script,str(root/'source'),str(root/'retrieved/pilot.pt')],check=True)
                upload(root/'retrieved/pilot.pt','/content/roundtrip.pt')
                remote("import subprocess\nsubprocess.run(['python','reliability_roundtrip.py'],cwd='/content/experiment',check=True)\n",timeout=120)
            if status['phase']=='awaiting_manifest_copy':remote("from pathlib import Path\nPath('/content/artifacts/manifest_verified').write_text('external manifest copy verified')\n")
            if status['phase']=='awaiting_selection_copy':
                digest=sha(root/'retrieved/selections.json')
                remote(f"from pathlib import Path\nPath('/content/artifacts/selection_verified').write_text({digest!r})\n")
                state['selection_commit_sha256']=digest;state['selection_committed_externally_at']=time.time();save()
            if status['phase'] in ('complete','failed'):state['result']=status;save();break
            time.sleep(40)
    except BaseException as exc:
        state['exception']=repr(exc);save();raise
    finally:
        for attempt in range(2):
            try:cli('stop','-s',session,timeout=90);state['stop_succeeded']=True;break
            except Exception as exc:state['stop_error']=repr(exc)
        state['ended']=time.time();state['total_allocated_upper_bound_seconds']=state['ended']-state['started'];save()
        listing=cli('sessions',timeout=90);(root/'sessions-after.txt').write_text(listing)
        if session in listing:raise RuntimeError('dedicated session still listed; watchdog retained')
        state['termination_verified']=True;save()
        for pid in state['watchdog_pids']:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--python',default='/tmp/tinysettler-rl-venv/bin/python');p.add_argument('--resume',action='store_true');p.add_argument('--replace',action='store_true');main(p.parse_args())

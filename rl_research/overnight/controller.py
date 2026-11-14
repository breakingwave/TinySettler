"""Bounded Colab lifecycle with cumulative budget and verified external copies."""
import argparse,hashlib,json,os,subprocess,tarfile,time,re
from pathlib import Path
CLI='/root/.local/bin/colab'
ROOT=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def upload_large(path,destination,artifacts,session,cli,remote):
    pieces=[]
    with Path(path).open('rb') as f:
        while data:=f.read(8*1024*1024):
            index=len(pieces)
            part=artifacts/f'upload-part-{index:04d}'
            part.write_bytes(data)
            target=f'{destination}.part{index:04d}'
            cli('upload','-s',session,str(part),target,timeout=180)
            pieces.append(target)
            part.unlink()
    code='import pathlib,hashlib,shutil\n'
    code+=f'with open({destination!r},"wb") as out:\n'
    code+=f' for name in {pieces!r}:\n'
    code+='  with open(name,"rb") as inp: shutil.copyfileobj(inp,out)\n'
    code+=f'with open({destination!r},"rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()=={sha(path)!r}\n'
    remote(code,timeout=120)

def main(a):
    a.artifacts.mkdir(parents=True,exist_ok=True)
    manifest=a.artifacts/'controller.json'
    if manifest.exists():
        if not a.resume: raise ValueError('existing manifest requires --resume; use a unique session for a replacement runtime')
        state=json.loads(manifest.read_text())
    else:
        now=time.time(); prior=json.loads(a.previous.read_text()) if a.previous else None
        used=prior.get('prior_allocated_seconds',0)+prior.get('ended',now)-prior['started'] if prior else 0
        deadline=min(now+28800-used,prior['deadline']) if prior else now+28800
        state=dict(session=a.session,started=now,deadline=deadline,prior_allocated_seconds=used,previous=str(a.previous) if a.previous else None,retries=0)
    assert state['session']==a.session
    if state['deadline']-time.time()<5400: raise TimeoutError('insufficient remaining original budget')
    state['controller_pid']=os.getpid()
    state.setdefault('attachments',[]).append(dict(time=time.time(),pid=os.getpid(),controller_source_sha256=sha(__file__)))
    def save():
        temp=manifest.with_suffix('.tmp'); temp.write_text(json.dumps(state,indent=2)); os.replace(temp,manifest)
    save()
    def cli(*args,code=None,timeout=120):
        remaining=state['deadline']-time.time()
        cleanup=args[0] in ('stop','sessions')
        if not cleanup and remaining<60: raise TimeoutError('global allocation deadline')
        r=subprocess.run([CLI,'--auth','oauth2',*args],input=code,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout if cleanup else min(timeout,max(30,remaining)))
        r.stdout=re.sub(r'([?&](?:colab-runtime-proxy-token|access_token|token)=)[^&\s)]+',r'\1<redacted>',r.stdout)
        with (a.artifacts/'controller.log').open('a') as f: f.write(r.stdout+'\n')
        if r.returncode: raise RuntimeError(r.stdout[-3000:])
        return r.stdout
    def remote(code,timeout=60):
        marker='CODEX_REMOTE_EXECUTION_COMPLETED'
        result=cli('exec','-s',a.session,'--timeout',str(timeout),code=code+'\nprint('+repr(marker)+')\n',timeout=timeout+30)
        clean=re.sub(r'\x1b\[[0-9;]*m','',result)
        if not any(line.strip()==marker for line in clean.splitlines()):
            raise RuntimeError('Remote Python did not complete; inspect sanitized controller log')
        return result
    # Independent local watchdog survives controller process failure (not host loss).
    watchdog_code="import subprocess,time,sys; time.sleep(max(0,float(sys.argv[1])-time.time())); subprocess.run(sys.argv[2:],timeout=120)"
    watchdog=subprocess.Popen([a.python,'-c',watchdog_code,str(state['deadline']-120),CLI,'--auth','oauth2','stop','-s',a.session],stdout=(a.artifacts/'watchdog.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    try:
        if not a.resume:
            cli('new','-s',a.session,'--gpu','T4',timeout=300)
            state['allocated']=time.time(); save()
            with tarfile.open(a.artifacts/'source.tar.gz','w:gz') as tar:
                for p in ROOT.glob('*'):
                    if p.is_file(): tar.add(p,arcname=p.name)
            state['source_hashes']={p.name:sha(p) for p in ROOT.glob('*') if p.is_file()}; save()
            cli('upload','-s',a.session,str(a.artifacts/'source.tar.gz'),'/content/source.tar.gz')
            remote("import tarfile,pathlib\npathlib.Path('/content/experiment').mkdir(exist_ok=True)\npathlib.Path('/content/artifacts').mkdir(exist_ok=True)\nwith tarfile.open('/content/source.tar.gz') as t: t.extractall('/content/experiment',filter='data')\n")
            remote("import subprocess\nwith open('/content/artifacts/install.log','w') as f:\n subprocess.run(['python','-m','pip','install','-r','/content/experiment/requirements-lock.txt'],check=True,stdout=f,stderr=subprocess.STDOUT)\nwith open('/content/artifacts/installed-freeze.txt','w') as f:\n subprocess.run(['python','-m','pip','freeze'],check=True,stdout=f)\n",timeout=900)
            if a.restore:
                with tarfile.open(a.artifacts/'restore.tar.gz','w:gz',compresslevel=1) as tar:
                    for p in a.restore.rglob('*'):
                        if p.is_file() and p.name!='recovery_verified': tar.add(p,arcname=str(p.relative_to(a.restore)))
                upload_large(a.artifacts/'restore.tar.gz','/content/restore.tar.gz',a.artifacts,a.session,cli,remote)
                remote("import tarfile\nwith tarfile.open('/content/restore.tar.gz') as t: t.extractall('/content/artifacts',filter='data')\n")
                state['restored_from']=str(a.restore); save()
            deadline=state['deadline']-300
            remote(f"import subprocess,pathlib\nlog=open('/content/artifacts/remote.log','w')\np=subprocess.Popen(['python','remote.py','--out','/content/artifacts','--deadline',{str(deadline)!r}],cwd='/content/experiment',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(p.pid)\n")
        while time.time()<state['deadline']-180:
            for attempt in range(2):
                try:
                    snapshot=remote("import tarfile,pathlib,hashlib,json,time\nsnapshot_started=time.time()\np=pathlib.Path('/content/artifacts')\nwith tarfile.open('/content/snapshot.tar','w') as t:\n for f in p.rglob('*'):\n  if f.is_file() and not f.name.endswith('.tmp') and f.stat().st_mtime > globals().get('snapshot_cutoff',0): t.add(f,arcname=str(f.relative_to(p)))\nwith open('/content/snapshot.tar','rb') as f: print('SNAPSHOT_SHA256='+hashlib.file_digest(f,'sha256').hexdigest())\n")
                    digest=next(line.split('=',1)[1].strip() for line in snapshot.splitlines() if line.startswith('SNAPSHOT_SHA256='))
                    cli('download','-s',a.session,'/content/snapshot.tar',str(a.artifacts/'snapshot.partial'),timeout=600)
                    assert sha(a.artifacts/'snapshot.partial')==digest
                    os.replace(a.artifacts/'snapshot.partial',a.artifacts/'snapshot.tar')
                    with tarfile.open(a.artifacts/'snapshot.tar') as tar: tar.extractall(a.artifacts/'retrieved',filter='data')
                    remote('snapshot_cutoff=snapshot_started\n') # acknowledge only verified copy
                    break
                except Exception:
                    if attempt: raise
                    state['retries']+=1; save()
                    if state['retries']>1: raise
            state['last_verified_snapshot']=time.time(); state['snapshot_sha256']=digest; save()
            statuspath=a.artifacts/'retrieved/status.json'
            status=json.loads(statuspath.read_text()) if statuspath.exists() else {'phase':'preflight'}
            progress=[json.loads(p.read_text())|{'run':p.parent.name} for p in (a.artifacts/'retrieved').glob('*/progress.json')]
            print(json.dumps(dict(time=time.time(),status=status,progress=progress)),flush=True)
            if status['phase']=='awaiting_external_recovery':
                script="import sys,torch; sys.path.insert(0,sys.argv[1]); from train import Trainer,source_hashes; from pathlib import Path\nfor p in Path(sys.argv[2]).glob('pilot-*.pt'):\n s=torch.load(p,map_location='cpu',weights_only=False); t=s['trainer']; assert t.n>=4096 and t.updates>0 and s['source_hashes']==source_hashes(); print(p.name,t.n)\n"
                subprocess.run([a.python,'-c',script,str(ROOT),str(a.artifacts/'retrieved')],check=True)
                remote("from pathlib import Path\nPath('/content/artifacts/recovery_verified').write_text('external snapshot hash and checkpoint reload verified')\n")
            if status['phase'] in ('complete','failed'):
                state['result']=status; save(); break
            time.sleep(45)
    except BaseException as exc:
        state['controller_exception']=repr(exc); save(); raise
    finally:
        for attempt in range(2):
            try:
                cli('stop','-s',a.session,timeout=120); state['stop_succeeded']=True; break
            except Exception as exc: state['stop_error']=str(exc)
        state['ended']=time.time(); state['total_allocated_upper_bound_seconds']=state.get('prior_allocated_seconds',0)+state['ended']-state['started']; save()
        listing=cli('sessions',timeout=120)
        (a.artifacts/'sessions-after.txt').write_text(listing)
        if a.session in listing: raise RuntimeError('session still listed after stop; watchdog retained')
        state['termination_verified']=True; save(); watchdog.terminate()
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--session',required=True); p.add_argument('--artifacts',type=Path,required=True); p.add_argument('--python',default='/tmp/tinysettler-rl-venv/bin/python'); p.add_argument('--previous',type=Path); p.add_argument('--restore',type=Path); p.add_argument('--resume',action='store_true'); main(p.parse_args())

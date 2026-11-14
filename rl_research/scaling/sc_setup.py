"""Detached, bounded and idempotent dependency installation; no notebook blocking."""
import json,os,subprocess,time
from pathlib import Path

def main():
 out=Path('/content/artifacts');out.mkdir(exist_ok=True)
 result=dict(started=time.time(),complete=False)
 try:
  with (out/'install.log').open('w') as f:
   subprocess.run(['python','-m','pip','install','--disable-pip-version-check','-r','/content/experiment/requirements-lock.txt'],stdout=f,stderr=f,check=True,timeout=720)
  with (out/'installed-freeze.txt').open('w') as f:subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True,timeout=30)
  result['complete']=True
 except BaseException as exc:result['error']=repr(exc)
 result['finished']=time.time();tmp=out/'setup-result.tmp';tmp.write_text(json.dumps(result));os.replace(tmp,out/'setup-result.json')

if __name__=='__main__':main()

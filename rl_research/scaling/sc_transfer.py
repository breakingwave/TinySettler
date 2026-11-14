"""Content-addressed immutable generations, published only after complete files."""
import hashlib,json,os,tarfile,time
from pathlib import Path

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def make(root,destination):
 root=Path(root);destination=Path(destination);destination.mkdir(exist_ok=True);objects=destination/'objects';objects.mkdir(exist_ok=True)
 ack=json.loads((destination/'ack.json').read_text()) if (destination/'ack.json').exists() else []
 previous=json.loads((destination/'cache.json').read_text()) if (destination/'cache.json').exists() else {};cache={};files={}
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.name.endswith('.tmp'):continue
  # Preserve one atomic pilot checkpoint per worker for failed-preflight recovery.
  if 'pilots' in p.parts and p.suffix=='.pt' and p.name!='latest.pt':continue
  rel=str(p.relative_to(root));stat=p.stat();signature=[stat.st_mtime_ns,stat.st_size,stat.st_ino]
  if rel in previous and previous[rel]['signature']==signature:
   info=previous[rel]['info']
  else:
   h=hashlib.sha256();temp=objects/'copy.tmp';size=0
   with p.open('rb') as inp,temp.open('wb') as out:
    while chunk:=inp.read(1024*1024):out.write(chunk);h.update(chunk);size+=len(chunk)
    out.flush();os.fsync(out.fileno())
   info=dict(sha256=h.hexdigest(),bytes=size);os.replace(temp,objects/info['sha256'])
  files[rel]=info;cache[rel]=dict(signature=signature,info=info)
 generation=dict(created=time.time(),files=files)
 (destination/'generation.json').write_text(json.dumps(generation))
 new=sorted({x['sha256'] for x in files.values()}-set(ack))
 with tarfile.open(destination/'transfer.tar.gz','w:gz',compresslevel=1) as tar:
  tar.add(destination/'generation.json',arcname='generation.json')
  for h in new:tar.add(objects/h,arcname='objects/'+h)
 (destination/'cache.json').write_text(json.dumps(cache))
 return dict(sha256=sha(destination/'transfer.tar.gz'),bytes=(destination/'transfer.tar.gz').stat().st_size,new_objects=len(new))

def receive(archive,root,expected):
 root=Path(root);assert sha(archive)==expected,'INTEGRITY: transfer tar SHA256'
 temp=root/'incoming';temp.mkdir(exist_ok=True);objects=root/'objects';objects.mkdir(exist_ok=True)
 with tarfile.open(archive) as tar:tar.extractall(temp,filter='data')
 generation=json.loads((temp/'generation.json').read_text())
 for rel,info in generation['files'].items():
  path=Path(rel);assert not path.is_absolute() and '..' not in path.parts
  src=temp/'objects'/info['sha256'];dst=objects/info['sha256']
  if src.exists():
   assert sha(src)==info['sha256'] and src.stat().st_size==info['bytes'];os.replace(src,dst)
  assert dst.exists() and sha(dst)==info['sha256'] and dst.stat().st_size==info['bytes']
 # Only now publish the durable generation manifest.
 gen=root/'generations';gen.mkdir(exist_ok=True);name=str(time.time_ns())+'.json';tmp=gen/'publish.tmp';tmp.write_text(json.dumps(generation));os.replace(tmp,gen/name)
 retrieved=root/'retrieved';retrieved.mkdir(exist_ok=True)
 for rel,info in generation['files'].items():
  dest=retrieved/rel;dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.unlink(missing_ok=True);os.link(objects/info['sha256'],tmp);os.replace(tmp,dest)
 for p in retrieved.rglob('*'):
  if p.is_file() and str(p.relative_to(retrieved)) not in generation['files']:p.unlink()
 manifests=sorted(gen.glob('*.json'))
 for p in manifests[:-2]:p.unlink()
 held={i['sha256'] for p in manifests[-2:] for i in json.loads(p.read_text())['files'].values()}
 for p in objects.iterdir():
  if p.name not in held:p.unlink()
 return dict(name=name,created=generation['created'],files=len(generation['files']),hashes=sorted({i['sha256'] for i in generation['files'].values()}))

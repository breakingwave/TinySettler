"""Consolidate common-final and distinct selected resumable policies after verified retrieval."""
import argparse,tarfile
from pathlib import Path
from reliability_common import read,write,sha,OLD_SEEDS

def main(root):
    frozen=read(root/'retrieved/frozen.json');selections=read(root/'retrieved/selections.json')['selections'];items=[]
    for sel in selections:
        base=Path(frozen['historical_location'])/'retrieved' if sel['seed'] in OLD_SEEDS else root/'retrieved'
        for n in sorted({434176,sel['step']}):
            p=base/sel['run']/f'{n}.pt'
            items.append(dict(path=p,archive_path=f"checkpoints/{sel['run']}/{n}.pt",sha256=sha(p),bytes=p.stat().st_size))
    archive=root/'common-final-and-selected.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for item in items:tar.add(item['path'],arcname=item['archive_path'])
        for filename in ['source.tar.gz','input-manifest.json','recovery-controller.py','calibration-repair.py','secondary-plan.py','operational-events.json']:
            if (root/filename).exists():tar.add(root/filename,arcname=filename)
        for filename in ['frozen.json','selections.json','external-recovery.json','calibration-correction.json','secondary-plan.json']:
            if (root/'retrieved'/filename).exists():tar.add(root/'retrieved'/filename,arcname=filename)
    # Read the bytes back from the package and verify each policy hash.
    import hashlib
    with tarfile.open(archive) as tar:
        for item in items:
            with tar.extractfile(item['archive_path']) as inp:assert hashlib.file_digest(inp,'sha256').hexdigest()==item['sha256']
    manifest=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),checkpoints=[dict(i,path=str(i['path'])) for i in items])
    write(root/'checkpoint-package-manifest.json',manifest);print(f'Packaged and verified {len(items)} resumable policies: {archive}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);main(p.parse_args().artifacts)

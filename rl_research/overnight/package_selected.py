"""Package selected resumable checkpoints with the exact executed source bundle."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main(a):
    controller=json.loads((a.artifacts/'controller.json').read_text())
    assert controller.get('termination_verified'), 'finish runtime cleanup before packaging'
    selections=json.loads((a.artifacts/'retrieved/selections.json').read_text())
    manifest=[]
    for sel in selections:
        path=a.artifacts/'retrieved'/sel['run']/Path(sel['path']).name
        manifest.append(dict(run=sel['run'],archive_path=f"checkpoints/{sel['run']}/{path.name}",sha256=digest(path),bytes=path.stat().st_size,complete_training=sel['complete']))
    destination=a.artifacts/'selected-checkpoints.tar.gz'
    temporary=destination.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        tar.add(a.artifacts/'source.tar.gz',arcname='executed-source.tar.gz')
        tar.add(a.artifacts/'retrieved/frozen.json',arcname='frozen.json')
        for row in manifest:
            tar.add(a.artifacts/'retrieved'/row['run']/Path(row['archive_path']).name,arcname=row['archive_path'])
        payload=json.dumps(manifest,indent=2).encode()
        info=tarfile.TarInfo('checkpoint-manifest.json'); info.size=len(payload); tar.addfile(info,io.BytesIO(payload))
        payload=b'''Extract executed-source.tar.gz into a source directory. Install its requirements-lock.txt in an isolated environment. The checkpoints contain trusted full Trainer state (including optimizer, game/RNG state and DDQN replay), not only weights. Evaluate on CUDA with: python source/evaluate.py checkpoints/RUN/STEP.pt --split final --boards 100 --output final.json. Use the repository's current README for budget-aware launch/recovery commands. Original final evaluation was inside Colab, not a browser. See the linked findings report for limitations and the selected-checkpoint table.\n'''
        info=tarfile.TarInfo('READ-ME.txt');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
    temporary.replace(destination)
    with tarfile.open(destination) as tar:
        for row in manifest:
            with tar.extractfile(row['archive_path']) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    result=dict(archive=str(destination),sha256=digest(destination),bytes=destination.stat().st_size,checkpoints=manifest)
    (a.artifacts/'selected-package-manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(archive=str(destination),bytes=destination.stat().st_size,checkpoints=len(manifest),verified=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);main(p.parse_args())

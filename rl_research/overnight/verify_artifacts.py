"""Validate externally retrieved trusted checkpoints and exercise CPU recovery."""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from train import Trainer, CONFIG, source_hashes
from env import MAPPING_HASH

def main(a):
    records=[]
    for folder in sorted(a.artifacts.glob('retrieved/*')):
        if not folder.is_dir(): continue
        checkpoints=sorted(folder.glob('*.pt'),key=lambda p:int(p.stem))
        if not checkpoints: continue
        selected=folder/'final.json'
        chosen=Path(json.loads(selected.read_text())['selection']['path']).name if selected.exists() else None
        for path in checkpoints:
            with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            s=torch.load(path,map_location='cpu',weights_only=False)
            assert s['source_hashes']==source_hashes() and s['mapping_hash']==MAPPING_HASH and s['config']==CONFIG
            t=s['trainer']; assert t.env.game.state.random is t.env.game.random
            assert all(torch.isfinite(p).all() for p in t.net.parameters())
            row=dict(path=str(path),sha256=digest,n=t.n,updates=t.updates,replay_size=len(t.replay),selected=path.name==chosen)
            if path==checkpoints[-1] or path.name==chosen:
                t.device='cpu'
                random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng']); torch.set_rng_state(s['torch_rng'])
                before=t.n
                t.collect(); assert t.n==before+1
                # Do not save the mutated local test copy over any checkpoint.
                if t.algo=='ppo': t.update()
                elif len(t.replay)>=CONFIG['ddqn']['batch']: t.update()
                row['cpu_recovery_collect_update_passed']=True
            records.append(row); del t,s
    a.output.write_text(json.dumps(records,indent=2)); print(json.dumps(dict(checkpoints=len(records),recovery_checks=sum(r.get('cpu_recovery_collect_update_passed',False) for r in records))))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--artifacts',type=Path,required=True); p.add_argument('--output',type=Path,required=True); main(p.parse_args())

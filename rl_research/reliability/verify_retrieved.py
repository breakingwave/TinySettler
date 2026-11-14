"""Offline hash/provenance/finiteness checks and CPU recovery smoke tests on trusted checkpoints."""
import argparse,random
from pathlib import Path
import numpy as np
import torch
from env import MAPPING_HASH
from train import CONFIG,source_hashes
from reliability_common import read,write,sha

def main(root):
    index=read(root/'retrieved/artifact-index.json');checks=[];recoveries=[]
    for key,expected in index.items():
        path=root/'retrieved'/key;assert sha(path)==expected['sha256'];assert path.stat().st_size==expected['bytes']
        saved=torch.load(path,map_location='cpu',weights_only=False);t=saved['trainer']
        assert saved['config']==CONFIG and saved['source_hashes']==source_hashes() and saved['mapping_hash']==MAPPING_HASH
        assert all(torch.isfinite(p).all() for p in t.net.parameters())
        assert all(torch.isfinite(v).all() for state in t.optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        checks.append(dict(path=key,sha256=expected['sha256'],transitions=t.n,updates=t.updates,finite=True,provenance=True))
        if path.name=='434176.pt':
            t.device='cpu';random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
            n=t.n
            for _ in range(32):t.collect()
            t.update()
            assert t.n==n+32 and all(torch.isfinite(p).all() for p in t.net.parameters())
            recoveries.append(dict(path=key,cpu_resume_collect_and_update=True,saved_checkpoint_unchanged=sha(path)==expected['sha256']))
    write(root/'recovery-checks.json',dict(checkpoints=checks,cpu_recoveries=recoveries,external_cuda_roundtrip=read(root/'retrieved/external-recovery.json')))
    print(f'Verified {len(checks)} checkpoints; {len(recoveries)} CPU recovery updates; original bytes unchanged.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);main(p.parse_args().artifacts)

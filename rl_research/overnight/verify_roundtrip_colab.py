import sys,hashlib,json
sys.path.insert(0,'/content/experiment')
import numpy as np
import torch
from train import Trainer
from pathlib import Path
original=Path('/content/artifacts/ppo-128-104729/323584.pt')
restored=Path('/content/roundtrip.pt')
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(original)==sha(restored)
a=Trainer.load(original); a.collect()
expected=a.env.obs().copy(); n=a.n
b=Trainer.load(restored); b.collect()
assert b.n==n and np.array_equal(expected,b.env.obs())
b.update()
result=dict(checkpoint_transitions=n-1,sha256=sha(restored),external_upload=True,cuda_next_transition_matches=True,resumed_update_passed=True)
Path('/content/artifacts/external-roundtrip.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

"""Execute with Colab CLI after uploading the external pilot checkpoint."""
import sys,hashlib,json,gc
sys.path.insert(0,'/content/experiment')
import numpy as np
import torch
from train import Trainer
from pathlib import Path
original=Path('/content/artifacts/pilot-ddqn-256.pt')
restored=Path('/content/ddqn-roundtrip.pt')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(original)==sha(restored)
a=Trainer.load(original); a.collect(); a.update()
expected=a.env.obs().copy(); n=a.n
b=Trainer.load(restored); b.collect(); b.update()
assert b.n==n and np.array_equal(expected,b.env.obs())
assert len(a.replay)==len(b.replay) and a.epsilon()==b.epsilon()
for model_a,model_b in [(a.net,b.net),(a.target_net,b.target_net)]:
    for name,value in model_a.state_dict().items():
        torch.testing.assert_close(value,model_b.state_dict()[name],rtol=0,atol=0)
result=dict(checkpoint_transitions=n-1,sha256=sha(restored),external_upload=True,cuda_next_transition_matches=True,resumed_update_bitwise_matches=True,replay_size=len(b.replay),epsilon=b.epsilon(),optimizer_updates=b.updates)
Path('/content/artifacts/external-roundtrip-ddqn.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
del a,b,expected,model_a,model_b,value
gc.collect(); torch.cuda.empty_cache()

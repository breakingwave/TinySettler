"""Verify external checkpoint round trip and exact CUDA continuation, without affecting sweep RNG."""
from pathlib import Path
import numpy as np
import torch
from train import Trainer
from reliability_common import sha,write
original=Path('/content/artifacts/pilot.pt');external=Path('/content/roundtrip.pt')
assert sha(original)==sha(external)
a=Trainer.load(original)
for _ in range(32):a.collect()
a.update();expected=[p.detach().clone() for p in a.net.parameters()]
b=Trainer.load(external)
for _ in range(32):b.collect()
b.update()
assert np.array_equal(a.env.obs(),b.env.obs())
assert all(torch.equal(x,y) for x,y in zip(expected,b.net.parameters()))
write('/content/artifacts/external-recovery.json',dict(sha256=sha(external),cuda_continuation_bitwise_equal=True,optimizer_update_equal=True,diagnostics_retained=a.env.public_states==b.env.public_states))
Path('/content/artifacts/recovery_verified').write_text(sha(external))

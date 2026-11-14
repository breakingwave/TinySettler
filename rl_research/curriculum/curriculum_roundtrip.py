from pathlib import Path
import torch,numpy as np
from train import Trainer
from curriculum_core import load_corpus
from reliability_common import sha,write
load_corpus('/content/historical/corpus.pt')
a_path=Path('/content/artifacts/pilot.pt');b_path=Path('/content/roundtrip.pt')
assert sha(a_path)==sha(b_path)
a=Trainer.load(a_path)
for _ in range(64):a.collect()
a.update();weights=[p.detach().clone() for p in a.net.parameters()]
b=Trainer.load(b_path)
for _ in range(64):b.collect()
b.update()
assert np.array_equal(a.env.obs(),b.env.obs())
assert all(torch.equal(x,y) for x,y in zip(weights,b.net.parameters()))
assert a.category_transitions==b.category_transitions
assert all(a.env.planners[c].rng.getstate()==b.env.planners[c].rng.getstate() for c in a.env.planners)
write('/content/artifacts/external-recovery.json',dict(sha256=sha(b_path),cuda_continuation_bitwise_equal=True,optimizer_update_equal=True,planning_rng_equal=True,category_counters_equal=True))
Path('/content/artifacts/recovery_verified').write_text(sha(b_path))

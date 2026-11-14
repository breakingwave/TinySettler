"""Shared deterministic configuration; independent of all historical bundles."""
import os,sys,json,hashlib,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
if not (HERE/'env.py').exists():
    for d in ['overnight','reliability','curriculum']:sys.path.append(str(HERE.parent/d))
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from env import *
from train import CONFIG as OLD_CONFIG, atomic_save, masked
from reliability_common import read,write,sha
SEEDS=[400009,400031,400033]
CONDITIONS={'A':('mlp128','static'),'B':('mlp128','refresh'),'C':('mlp512','static'),'D':('mlp512','refresh'),'E':('graph','static'),'F':('graph','refresh'),'G':('mlp128','refresh')}
NAMESPACES={'teacher':11000000000000,'pilot':12000000000000,'train':13000000000000,'validation':14000000000000,'final':15000000000000,'regression':16000000000000,'checks':17000000000000}
PPO=OLD_CONFIG['ppo'].copy()
CAP=50400; RESERVE=1800
ROOT=Path(os.environ.get('SCALING_INPUTS',str(HERE/'inputs')))
SCHEMA='compact-normalized-879-v1'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def hashes():
    paths=list(HERE.glob('sc_*.py'))+[HERE/'schema.json',HERE/'requirements-lock.txt']
    # Local execution resolves the unchanged historical helpers outside this directory.
    for module in ['env','train','opponent_adapter','curriculum_core','reliability_common']:
        m=__import__(module);paths.append(Path(m.__file__))
    paths.append(Path(__import__('train').__file__).with_name('config.json'))
    return {p.name:sha(p) for p in paths if p.exists()}
def snapshot_rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2].cpu())
    if s[3] and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in s[3]])
def run_ids(conditions):return [(c,s) for s in SEEDS for c in conditions if s!=SEEDS[2] or c in ('A','B')]
def count_params(net):return sum(p.numel() for p in net.parameters())

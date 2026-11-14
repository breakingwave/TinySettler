"""Freeze all inputs locally before any GPU allocation."""
import argparse,json,os,shutil,tarfile,time
from pathlib import Path
from curriculum_core import *
from prepare_data import checkpoint

def main(data,out):
    assert not out.exists(),'new experiment directory required'
    out.mkdir(parents=True);source=out/'source';source.mkdir()
    old=Path('/root/tinysettler-artifacts/tinysettler-ablation-20261111-b')
    with tarfile.open(old/'source.tar.gz') as tar:tar.extractall(source,filter='data')
    here=Path(__file__).resolve().parent
    for p in (here.parent/'reliability').glob('*.py'):shutil.copy2(p,source/p.name)
    for p in here.glob('*.py'):shutil.copy2(p,source/p.name)
    shutil.copy2(here.parent/'overnight/cpu_monitor.py',source/'cpu_monitor.py')
    for name in ['train.py','env.py','config.json','requirements-lock.txt']:assert sha(source/name)==source_hashes()[name]
    for p in data.iterdir():
        if p.is_file():shutil.copy2(p,out/p.name)
    inputs=out/'inputs';inputs.mkdir()
    checks=read(data/'historical-checks.json');starting={}
    for seed in SEEDS:
        src=checkpoint(seed);assert sha(src)==checks[str(seed)]['sha256'];shutil.copy2(src,inputs/f'{seed}.pt');starting[str(seed)]=checks[str(seed)]
    shutil.copy2(data/'corpus.pt',inputs/'corpus.pt')
    corpus=read(data/'corpus-manifest.json');assert sha(inputs/'corpus.pt')==corpus['sha256']
    # Fresh namespaces are above every legacy range; inspect prior manifests/rows.
    found=set()
    def scan(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('board_seed','seed_start','board_seed_start') and isinstance(v,int):found.add(v)
                scan(v)
        elif isinstance(x,list):
            for y in x:scan(y)
    for p in (here.parent/'results').rglob('*.json'):
        try:scan(read(p))
        except (ValueError,UnicodeError):pass
    namespaces=dict(continuation=[710000000000,710030000000],opponent_preflight=[820000000000,820000000004],corpus_train=[830000000000,830000000060],corpus_validation=[840000000000,840000000012],corpus_test=[850000000000,850000000016],task_future=[9360000000000,9360000010000],calibration=[870000000000,870000000101],final=[880000000000,880000000020])
    assert all(not any(lo<=seed<hi for seed in found) for lo,hi in namespaces.values())
    manifest=dict(version=1,created=time.time(),allocation_cap_seconds=14400,retrieval_reserve_seconds=1200,phase_budgets_seconds=dict(preflight=1500,training=8100,evaluation=3000,retrieval=1200,recovery=600),workers=1,starting_seeds=SEEDS,starting_step=434176,starting_checkpoints=starting,order=RUNS,transition_cap=262144,opponent_mixture=MIX,reset_mixture=RESET,sampled_worlds=2,opponent_label=AdaptedValuePlayer.label,opponent_belief='conservative public resource intervals; conditioned observed legal set; shared sampled worlds; float64 upstream scores',held_out_opponent='contender: same heuristic family with different upstream weights',corpus_sha256=sha(inputs/'corpus.pt'),corpus_manifest_sha256=sha(out/'corpus-manifest.json'),core_hashes=source_hashes(),learning_settings=CONFIG['ppo'],gamma=CONFIG['gamma'],reward=CONFIG['reward'],rules=CONFIG['rules'],seed_namespaces=namespaces,prior_unique_seeds_checked=len(found),pythonhashseed='0',simulator_revision='ecf931181b9a65bb4116a2153fb78c16f1438e00',primary='six common final additional-transition policies, fresh full games; no checkpoint selection',diagnostic_reconstruction='one of the two requested 25-card/zero-knight/8-VP cases exactly reproduced; the other labeled approximate because historical hash seed/action trace was not archived',source_hashes={p.name:sha(p) for p in source.iterdir() if p.is_file()})
    write(out/'input-manifest.json',manifest);write(source/'input-manifest.json',manifest)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for p in source.iterdir():
            if p.is_file():tar.add(p,arcname=p.name)
    with tarfile.open(out/'historical.tar.gz','w:gz',compresslevel=1) as tar:
        for p in inputs.iterdir():tar.add(p,arcname=p.name)
    write(out/'launch-integrity.json',dict(source_sha256=sha(out/'source.tar.gz'),inputs_sha256=sha(out/'historical.tar.gz'),corpus_sha256=sha(inputs/'corpus.pt')))
    print(json.dumps(dict(artifact_root=str(out),source_sha256=sha(out/'source.tar.gz'),input_archive_bytes=(out/'historical.tar.gz').stat().st_size)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();main(a.data,a.artifacts)

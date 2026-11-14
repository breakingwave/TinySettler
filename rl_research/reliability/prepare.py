"""Verify historical bytes and build a self-contained, immutable launch bundle before GPU allocation."""
import argparse,hashlib,json,tarfile,time,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent; OLD=HERE.parent/'overnight'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(out):
    assert not out.exists(),'use a new artifact directory'
    historical=Path('/root/tinysettler-artifacts/tinysettler-ablation-20261111-b')
    recorded=json.loads((HERE.parent/'results/overnight-20261111/artifact-manifest.json').read_text())
    checks={}
    for w in [128,256]:
        for s in [104729,130363]:
            for n in [106496,217088,323584,434176]:
                key=f'retrieved/ppo-{w}-{s}/{n}.pt';p=historical/key
                assert sha(p)==recorded['files'][key]['sha256'],key
                checks[key]={'sha256':sha(p),'bytes':p.stat().st_size}
    out.mkdir(parents=True);stage=out/'source';stage.mkdir()
    with tarfile.open(historical/'source.tar.gz') as tar:tar.extractall(stage,filter='data')
    for f in ['train.py','env.py','config.json','requirements-lock.txt']:
        assert sha(stage/f)==sha(OLD/f),'current core differs from archive'
    for p in HERE.glob('*.py'):shutil.copy2(p,stage/p.name)
    shutil.copy2(OLD/'cpu_monitor.py',stage/'cpu_monitor.py')
    # Read every prior recorded evaluation row; reserve entire possible training episode ranges.
    recorded_boards=set(); sources=[]
    def scan(v):
        if isinstance(v,dict):
            if 'board_seed' in v:recorded_boards.add(v['board_seed'])
            for x in v.values():scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
    for p in (HERE.parent/'results').rglob('*.json'):
        try:scan(json.loads(p.read_text()));sources.append(str(p.relative_to(HERE.parent)))
        except (ValueError,UnicodeError):pass
    ranges=[(s*10000+1,s*10000+434177) for s in [104729,130363,155921,196613,262147]]
    splits={'validation':{'seed_start':2100000000,'boards':20},'final':{'seed_start':3100000000,'boards':50},'calibration':{'seed_start':1800000000,'boards':2}}
    used=set(recorded_boards)|set(range(1900000000,1900000010))|set(range(2000000000,2000000002))|set(range(3000000000,3000000100))
    for split in splits.values():
        values=set(range(split['seed_start'],split['seed_start']+split['boards']))
        assert not used&values
        assert not any(lo<=v<=hi for lo,hi in ranges for v in values)
        used|=values
    manifest=dict(version=1,created=time.time(),allocation_cap_seconds=18000,budget_override='User explicitly allowed up to five hours; supersedes copied four-hour plan',retrieval_reserve_seconds=1200,run_training_limit_seconds=2100,workers=1,target=434176,steps=[106496,217088,323584,434176],new_seeds=[155921,196613,262147],historical_seeds=[104729,130363],widths=[128,256],order=[[w,s] for i,s in enumerate([155921,196613,262147]) for w in ([128,256] if i%2==0 else [256,128])],splits=splits,selection_rule='win fraction; fewer truncations; earlier checkpoint',primary='common final 434176 transitions',secondary='distinct expanded-validation-selected checkpoints',optional_challenge='omitted to prioritize primary coverage',historical_location=str(historical),historical_checkpoint_checks=checks,archived_source_sha256=sha(historical/'source.tar.gz'),core_hashes={f:sha(stage/f) for f in ['train.py','env.py','config.json','requirements-lock.txt']},source_hashes={p.name:sha(p) for p in stage.iterdir() if p.is_file()},seed_audit=dict(prior_unique_board_seeds=len(recorded_boards),scanned_manifests=sources,training_episode_ranges=ranges,disjoint=True),simulator_revision='ecf931181b9a65bb4116a2153fb78c16f1438e00',learning_config=json.loads((stage/'config.json').read_text()),operational_override_fields=['training_seeds','validation_seed_start','final_seed_start','gpu_limit_seconds','run_limit_seconds','configurations'],inference_location='Colab T4; network and masked action only, excludes observation encoding and host-to-device transfer')
    (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(out/'input-manifest.json',stage/'input-manifest.json')
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for p in stage.iterdir():
            if p.is_file():tar.add(p,arcname=p.name)
    with tarfile.open(out/'historical.tar.gz','w:gz',compresslevel=1) as tar:
        for key in checks:tar.add(historical/key,arcname=key.removeprefix('retrieved/'))
    print(json.dumps({'artifact_directory':str(out),'verified_checkpoints':len(checks),'historical_archive_bytes':(out/'historical.tar.gz').stat().st_size,'source_bundle_sha256':sha(out/'source.tar.gz')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);main(p.parse_args().artifacts)

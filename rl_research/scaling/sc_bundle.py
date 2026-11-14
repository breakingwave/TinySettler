"""Freeze a new source/input bundle only after local bounded checks pass."""
from sc_train import *
import argparse,shutil,tarfile

def main(root,inputs,checks):
 assert not root.exists(),'new dedicated artifact root required'
 root.mkdir(parents=True);source=root/'source';source.mkdir();archive=Path('/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a/source')
 for p in archive.iterdir():
  if p.is_file():shutil.copy2(p,source/p.name)
 for p in HERE.iterdir():
  if p.is_file():shutil.copy2(p,source/p.name)
 shutil.copytree(inputs,root/'inputs');shutil.copytree(checks,root/'local-checks',ignore=shutil.ignore_patterns('*.pt'))
 manifest=dict(experiment='fourteen-hour-scaling-20261114',created=time.time(),schema=schema(),source_hashes=hashes(),mapping_hash=MAPPING_HASH,
  seeds=SEEDS,namespaces=read(inputs/'namespaces.json'),conditions=CONDITIONS,ppo=PPO,parameter_counts={k:count_params(model(k)) for k in ['mlp128','mlp512','graph']},graph_width=GRAPH_WIDTH,graph_width_grid=GRID,
  archive_sha256=sha(inputs/'archive.pt'),archive_manifest=read(inputs/'archive-manifest.json'),teacher_sha256=sha(inputs/'teacher.pt'),teacher_manifest=read(inputs/'teacher-manifest.json'),
  deadline_seconds=50400,reserve_seconds=1800,phase_budgets_seconds=dict(preflight=2700,training=36000,validation=2700,final=5400,retrieval=1800,recovery=1800),
  local_checks=read(checks/'checks.json'),runtime_recovery=dict(planned_handoff_at_seconds=36000,max_planned_handoffs=1,max_fault_replacements=1,cli_attempts=3,backoff_seconds=[10,30],connection_unavailable_seconds=900,external_stale_fatal_seconds=1800),
  review_every_seconds=3600,checkpoint_seconds=300,verified_external_copy_seconds=600,retained_generations=2,initialization='fresh PyTorch default; same architecture/seed identical across schedules; G teacher then fresh PPO optimizer',
  rules=dict(players=2,board='BASE',victory_points=10,discard_limit=7,max_turns=1000,max_engine_actions=20000,domestic_negotiation=False),
  simulator_revision='ecf931181b9a65bb4116a2153fb78c16f1438e00',cli_version='0.7.5',compute_units_before=274.95,search_gate='omitted: repaired search not validated; no raw stock search benchmark',
  optimizer='Adam lr=0.0003, fixed',teacher=dict(epochs=3,batch=256,strata=[.2,.4,.4],optimizer='Adam lr=0.0003 actor/body only; fresh PPO Adam afterward'),opponent_reset_probabilities=dict(base=.25,weighted=.125,heuristic=.125,latest_or_reference=.25,archive=.25),snapshot_interval=65536,max_retained_own_snapshots=8,model_evaluation_mode_primary='sampled temperature1',fixed_topology_removed=1170)
 write(source/'input-manifest.json',manifest);write(root/'input-manifest.json',manifest)
 for name,folder in [('source',source),('inputs',root/'inputs')]:
  with tarfile.open(root/f'{name}.tar.gz','w:gz',compresslevel=1) as tar:
   for p in folder.iterdir():
    if p.is_file():tar.add(p,arcname=p.name)
 write(root/'launch-integrity.json',{f'{n}.tar.gz':dict(sha256=sha(root/f'{n}.tar.gz'),bytes=(root/f'{n}.tar.gz').stat().st_size) for n in ['source','inputs']})
 print(json.dumps(dict(root=str(root),source_sha256=sha(root/'source.tar.gz'),inputs_bytes=(root/'inputs.tar.gz').stat().st_size)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--checks',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.inputs,a.checks)

"""Independent final integrity/coverage audit and portable policy package."""
from sc_train import *
import argparse,tarfile

def main(root,out):
 remote=root/'retrieved';controller=read(root/'controller.json');assert controller['all_terminated'] and all(a['terminated'] for a in controller['allocations'])
 assert controller['deadline']-controller['started']==50400 and controller['elapsed_upper_bound']<=50400
 frozen=read(remote/'frozen.json');common=read(remote/'common-step.json')['step'];index=read(remote/'checkpoint-index.json');checks=[]
 for rel,info in index.items():
  p=remote/rel;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'];checks.append(dict(path=str(p),**info))
 assert read(remote/'external-probe-expected.json')==read(remote/'external-probe-returned.json')
 assert controller['selection_sha256']==sha(remote/'selection.json')
 assert all(read(p)['spec']['seed_start']>=NAMESPACES['final'] for p in (remote/'final').glob('*.json'))
 pairs=[]
 for a,b in [('A','B'),('C','D'),('E','F')]:
  if a not in frozen['conditions']:continue
  for seed in [s for c,s in frozen['runs'] if c==a]:
   x=ScalingTrainer.load(remote/'runs'/f'{a}-{seed}'/'0.pt','cpu');y=ScalingTrainer.load(remote/'runs'/f'{b}-{seed}'/'0.pt','cpu')
   assert x.initial_weight_sha256==y.initial_weight_sha256
   assert all(torch.equal(v,y.net.state_dict()[k]) for k,v in x.net.state_dict().items())
   assert not x.optimizer.state and not y.optimizer.state;pairs.append(dict(a=a,b=b,seed=seed,identical_initial_weights=True,fresh_optimizers=True))
 if 'G' in frozen['conditions']:
  for seed in SEEDS[:2]:
   x=ScalingTrainer.load(remote/'runs'/f'B-{seed}'/'0.pt','cpu');y=ScalingTrainer.load(remote/'runs'/f'G-{seed}'/'0.pt','cpu')
   assert x.initial_weight_sha256==y.initial_weight_sha256 and not y.optimizer.state
   assert all(torch.equal(v,y.net.value.state_dict()[k]) for k,v in x.net.value.state_dict().items());pairs.append(dict(a='B',b='G-before-teacher',seed=seed,identical_initial_weights=True,critic_head_unchanged=True))
 recovery=[];policies=root/'final-policy-exports';policies.mkdir(exist_ok=True)
 for c,s in frozen['runs']:
  p=remote/'runs'/f'{c}-{s}'/f'{common}.pt';t=ScalingTrainer.load(p,'cpu');assert t.n==common and t.schema==SCHEMA
  assert sum(t.transition_shares.values())==t.n
  exported=export(t.net,t.kind,f'{c}-{s}-{common}',common);atomic_save(exported,policies/f'{c}-{s}.pt')
  before=t.n
  for _ in range(8):t.collect()
  t.update();assert t.n==before+8 and all(torch.isfinite(p).all() for p in t.net.parameters());recovery.append(dict(condition=c,seed=s,n=common,sha256=sha(p),cpu_collect_update=True))
 panels=[]
 for group in ['validation','final','regression','context']:
  for p in (remote/group).glob('*.json'):
   d=read(p);keys=set()
   for row in d['rows']:
    key=tuple(row[k] for k in ['board_seed','opponent','mode','seat']);assert key not in keys;keys.add(key)
    assert int(row['win'])+int(row['loss'])+int(row['truncated'])==1 and row['illegal_actions']==0
   expected=d['spec']['boards']*len(d['spec']['opponents'])*len(d['spec']['modes'])*2
   panels.append(dict(panel=str(p.relative_to(remote)),games=len(keys),expected=expected,complete=len(keys)==expected))
 if (remote/'confirmation.json').exists():
  d=read(remote/'confirmation.json');panels.append(dict(panel='confirmation.json',games=len(d['rows']),expected=4*(frozen['confirmation_boards']-frozen['final_boards']),complete=d.get('complete',False)))
 package=root/'policies-source-results.tar.gz'
 with tarfile.open(package,'w:gz',compresslevel=1) as tar:
  for p in policies.glob('*.pt'):tar.add(p,arcname='policies/'+p.name)
  for p in [root/'source.tar.gz',root/'inputs.tar.gz',root/'input-manifest.json',remote/'frozen.json',remote/'selection.json']:tar.add(p,arcname=p.name)
  for p in out.iterdir():
   if p.is_file():tar.add(p,arcname='reports/'+p.name)
 with tarfile.open(package) as tar:
  for p in policies.glob('*.pt'):assert hashlib.file_digest(tar.extractfile('policies/'+p.name),'sha256').hexdigest()==sha(p)
 audit=dict(checkpoints=checks,initial_pairs=pairs,cpu_recoveries=recovery,panels=panels,all_panels_complete=all(p['complete'] for p in panels) and len(list((remote/'final').glob('*.json')))==2*len(frozen['runs']) and len(list((remote/'regression').glob('*.json')))==len(frozen['runs']) and len(list((remote/'context').glob('*.json')))==3 and (remote/'confirmation.json').exists(),training_complete=common==frozen['target'],common_step=common,target=frozen['target'],all_runtimes_terminated=True,original_elapsed_seconds=controller['elapsed_upper_bound'],allocated_interval_upper_bound=sum(a['stopped']-a['requested'] for a in controller['allocations']),package=dict(path=str(package),sha256=sha(package),bytes=package.stat().st_size),inference_exports={p.name:sha(p) for p in policies.glob('*.pt')})
 write(out/'verification.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['checkpoints','initial_pairs','cpu_recoveries','panels','inference_exports']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.artifacts,a.output)

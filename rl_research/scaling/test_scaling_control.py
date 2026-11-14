import json,hashlib
from pathlib import Path
from sc_plan import *
from sc_transfer import make,receive
from sc_env import *

def test_scope_and_seed_balance():
 p={c:{'single':{'aggregate_rate':300,'teacher_seconds':5,'recovery_passed':True},'dual':{'aggregate_rate':500,'teacher_seconds':5,'recovery_passed':True}} for c in CONDITIONS}
 timing={k:{'seconds_per_game':.5} for k in ['mlp128','mlp512','graph']}
 f=freeze(p,timing,1000,1000+47000,2)
 assert len(f['runs'])==16 and f['target']%16384==0 and f['workers']==2
 assert f['final_games']==5360 and f['validation_games']==3840
 for c in ['E','F']:p[c]['single']['aggregate_rate']=5;p[c]['dual']['aggregate_rate']=8
 f=freeze(p,timing,1000,1000+47000,2)
 assert f['conditions']==list('ABCD')
 assert [x['conditions'] for x in f['scope_decisions']]==[list('ABCDEFG'),list('ABCDEF'),list('ABCD')]
 assert sum(c=='A' for c,s in f['runs'])==3 and sum(c=='C' for c,s in f['runs'])==2
 assert f['projection']['training_seconds']<=36000
 assert all(f['milestones'][i]<f['milestones'][i+1] for i in range(3))

def test_immutable_external_generations(tmp_path):
 source=tmp_path/'remote';source.mkdir();transport=tmp_path/'transport';external=tmp_path/'external';external.mkdir()
 (source/'checkpoint.pt').write_bytes(b'first complete checkpoint');(source/'partial.tmp').write_bytes(b'not complete')
 info=make(source,transport);first=receive(transport/'transfer.tar.gz',external,info['sha256'])
 assert len(list((external/'generations').glob('*.json')))==1
 (transport/'ack.json').write_text(json.dumps(first['hashes']))
 (source/'checkpoint.pt').write_bytes(b'second complete checkpoint');info=make(source,transport);receive(transport/'transfer.tar.gz',external,info['sha256'])
 assert len(list((external/'generations').glob('*.json')))==2
 assert (external/'retrieved/checkpoint.pt').read_bytes()==b'second complete checkpoint'
 assert not (external/'retrieved/partial.tmp').exists()
 old=hashlib.sha256(b'first complete checkpoint').hexdigest();assert (external/'objects'/old).read_bytes()==b'first complete checkpoint'

def test_opponent_cache_does_not_advance_learning_rng():
 state=torch.get_rng_state().clone();spec=torch.load(ROOT/'archive.pt',map_location='cpu',weights_only=False)[0];FrozenPolicy(spec)
 assert torch.equal(state,torch.get_rng_state())

def test_cpu_initialization_never_reseeds_cuda(monkeypatch):
 calls=[]
 monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:calls.append(seed))
 spec=torch.load(ROOT/'archive.pt',map_location='cpu',weights_only=False)[0]
 FrozenPolicy(spec);model('graph',400009)
 assert calls==[]

def test_cuda_opponent_cache_rng_isolation():
 if not torch.cuda.is_available():return
 torch.cuda.manual_seed_all(12345);before=torch.cuda.get_rng_state_all()
 spec=torch.load(ROOT/'archive.pt',map_location='cpu',weights_only=False)[0]
 FrozenPolicy(spec);model('graph',400009)
 assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))

def test_schedule_initial_weights():
 for a,b in [('A','B'),('C','D'),('E','F')]:
  for seed in SEEDS[:2]:
   x=model(CONDITIONS[a][0],seed);y=model(CONDITIONS[b][0],seed)
   assert all(torch.equal(v,y.state_dict()[k]) for k,v in x.state_dict().items())

def test_physical_hidden_world_observation_invariance():
 from opponent_adapter import hand
 e=ScalingEnv(NAMESPACES['checks']+43,1,'weighted')
 for i in range(150):
  _,_,terminal,truncated,_=e.step(int(np.flatnonzero(e.mask())[i%sum(e.mask())]))
  if terminal or truncated:break
 if not (terminal or truncated):
  for _ in range(3):
   world=e.planners[e.color]._world(e.game,e.knowledge[e.color])
   assert np.array_equal(e.obs(),compact(observe(world,e.color)))
   assert set(world.playable_actions)==set(e.game.playable_actions)
   assert all(world.state.resource_freqdeck[i]+sum(hand(world.state,c)[i] for c in COLORS)==19 for i in range(5))

def test_resume_preserves_original_deadline_and_refuses_fresh_state(tmp_path):
 from sc_controller import Supervisor
 root=tmp_path/'controller';root.mkdir()
 a=Supervisor(root,'no-runtime-test',False);deadline=a.s['deadline'];started=a.s['started']
 b=Supervisor(root,'no-runtime-test',True)
 assert b.s['deadline']==deadline and b.s['started']==started and deadline-started==50400
 import pytest
 with pytest.raises(AssertionError):Supervisor(root,'no-runtime-test',False)
 assert not b.s['allocations']

def test_corrupt_copy_cannot_publish_generation(tmp_path):
 import pytest
 source=tmp_path/'remote';source.mkdir();(source/'latest.pt').write_bytes(b'atomic checkpoint')
 info=make(source,tmp_path/'transfer');archive=tmp_path/'transfer/transfer.tar.gz'
 archive.write_bytes(archive.read_bytes()+b'corrupt')
 with pytest.raises(AssertionError):receive(archive,tmp_path/'external',info['sha256'])
 assert not (tmp_path/'external/generations').exists()

def test_fatal_resume_cannot_allocate_again(tmp_path,monkeypatch):
 import pytest
 from sc_controller import Supervisor
 root=tmp_path/'closed';root.mkdir();s=Supervisor(root,'never-allocate',False)
 s.s['fatal']='bounded preflight failure';s.s['fault_replacements']=1;s.save()
 monkeypatch.setattr(s,'allocate',lambda:pytest.fail('must not allocate'))
 monkeypatch.setattr(s,'watchdog',lambda:pytest.fail('must not start watchdog'))
 with pytest.raises(RuntimeError,match='closed'):s.run()

def test_pilot_latest_is_preserved_for_forensics(tmp_path):
 source=tmp_path/'remote';p=source/'pilots/E-single/0';p.mkdir(parents=True)
 (tmp_path/'external').mkdir()
 (p/'latest.pt').write_bytes(b'recoverable');(p/'0.pt').write_bytes(b'initial');(p/'8192.pt').write_bytes(b'duplicate')
 info=make(source,tmp_path/'transfer');receive(tmp_path/'transfer/transfer.tar.gz',tmp_path/'external',info['sha256'])
 assert (tmp_path/'external/retrieved/pilots/E-single/0/latest.pt').read_bytes()==b'recoverable'
 assert not (tmp_path/'external/retrieved/pilots/E-single/0/0.pt').exists()

def test_explicit_additional_recovery_is_bounded(tmp_path,monkeypatch):
 import pytest
 from sc_controller import Supervisor
 root=tmp_path/'authorized';root.mkdir();s=Supervisor(root,'never-allocate',False)
 s.s['fault_replacements']=1;s.s['max_fault_replacements']=2;calls=[]
 monkeypatch.setattr(s,'stop',lambda:calls.append('stop'));monkeypatch.setattr(s,'allocate',lambda:calls.append('allocate'))
 s.recovery();assert calls==['stop','allocate'] and s.s['fault_replacements']==2
 with pytest.raises(RuntimeError,match='allowance'):s.recovery()

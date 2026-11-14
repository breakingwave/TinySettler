"""CPU-only preparation: strict historical exports, source-game split teacher data."""
from sc_env import *
import argparse,subprocess,shutil,tarfile
OLDROOT=Path('/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a')

def export_archive(out):
    # Load the original modules, never relax provenance; compare exported tensors/logits.
    meta=read(OLDROOT/'input-manifest.json');paths=[]
    for seed in [104729,130363,155921]:
        paths.append((f'parent-{seed}',OLDROOT/'inputs'/f'{seed}.pt',meta['starting_checkpoints'][str(seed)]['sha256']))
        rel=f'standard-{seed}/176128.pt';info=read(OLDROOT/'retrieved/artifact-index.json')[rel]
        paths.append((f'standard-{seed}',OLDROOT/'retrieved'/rel,info['sha256']))
    # First historical seed outside the training archive's three lineages, not selected by outcomes.
    held=Path('/root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a/retrieved/ppo-128-196613/434176.pt')
    checks=read(OLDROOT/'historical-checks.json');paths.append(('heldout-196613',held,checks['196613']['sha256']))
    exports=[];audit=[]
    for name,path,expected in paths:
        assert sha(path)==expected,(name,path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['source_hashes']==meta['core_hashes'] and state['mapping_hash']==MAPPING_HASH and state['config']==OLD_CONFIG
        t=state['trainer'];net=MLP(128,2049);net.load_state_dict(t.net.state_dict());net.eval();t.net.eval()
        maxerr=0.
        for i in range(8):
            e=ScalingEnv(NAMESPACES['checks']+i,i%2,'weighted');x=torch.from_numpy(observe(e.game,e.color))
            with torch.no_grad():
                old=t.net(x);new=net(x);maxerr=max(maxerr,float((old[0]-new[0]).abs().max()))
                assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1])
        exports.append(dict(id=name,schema='legacy-2049',kind='mlp128',source_checkpoint=str(path),source_sha256=expected,state_dict=net.state_dict()))
        audit.append(dict(id=name,path=str(path),sha256=expected,source_checked=True,logits_bitwise_equal=True,max_error=maxerr))
    atomic_save(exports,out/'archive.pt');write(out/'archive-manifest.json',dict(policies=audit,reference='standard-104729',initial=[x['id'] for x in exports[:-1]],heldout=exports[-1]['id'],archive_sha256=sha(out/'archive.pt')))

def teacher_data(out):
    start=time.perf_counter();splits={'train':[],'validation':[],'test':[]};seen=Counter();games=[]
    reservoirs={k:random.Random(519001+i) for i,k in enumerate(splits)};limits={'train':20000,'validation':2500,'test':2500}
    for i in range(200):
        split='train' if i<160 else 'validation' if i<180 else 'test'
        seed=NAMESPACES['teacher']+i;opponent=['base','base','weighted','heuristic'][i%4];e=ScalingEnv(seed,i%2,opponent)
        decisions=0
        while True:
            actions=e.game.playable_actions;c=e.color
            action=e.planners[c].decide(e.game,e.knowledge[c]);a=action_index(action,c)
            if len(actions)>1:
                phase='setup' if e.game.state.is_initial_build_phase else 'early' if e.points()<=4 else 'remaining'
                row=dict(obs=e.obs(),mask=e.mask(),action=a,stratum=phase,source_game=seed,seat=i%2)
                seen[split]+=1;buf=splits[split]
                if len(buf)<limits[split]:buf.append(row)
                else:
                    j=reservoirs[split].randrange(seen[split])
                    if j<len(buf):buf[j]=row
                decisions+=1
            _,r,term,trunc,_=e.step(a)
            if term or trunc:break
        games.append(dict(source_game=seed,split=split,seat=i%2,opponent=opponent,nonforced_decisions=decisions,terminal=term,truncated=trunc))
        if i%20==19:print(json.dumps(dict(teacher_games=i+1,elapsed=time.perf_counter()-start,stored={k:len(v) for k,v in splits.items()})),flush=True)
    atomic_save(splits,out/'teacher.pt')
    write(out/'teacher-manifest.json',dict(games=games,observed_nonforced=dict(seen),stored={k:len(v) for k,v in splits.items()},strata={k:dict(Counter(r['stratum'] for r in v)) for k,v in splits.items()},unique_states={k:len({hashlib.sha256(r['obs'].tobytes()+r['mask'].tobytes()).hexdigest() for r in v}) for k,v in splits.items()},cpu_seconds=time.perf_counter()-start,sha256=sha(out/'teacher.pt'),max_games=200,max_stored=25000,split_before_generation=True,reservoir_seed=[519001,519002,519003]))

def namespace_check(out):
    found=[]
    for p in (HERE.parent/'results').rglob('*.json'):
        if p.stat().st_size>20000000:continue
        def visit(v,k=''):
            if isinstance(v,dict):
                for key,value in v.items():visit(value,key)
            elif isinstance(v,list):
                for value in v:visit(value,k)
            elif isinstance(v,int) and ('seed' in k.lower() or k in ['source_game','board']):found.append(v)
        try:visit(read(p))
        except (ValueError,UnicodeError):pass
    assert all(not any(lo<=x<lo+100000000000 for x in found) for lo in NAMESPACES.values())
    assert not set(SEEDS)&set(found)
    write(out/'namespaces.json',dict(namespaces=NAMESPACES,range_size=100000000000,training_seeds=SEEDS,prior_seed_values_checked=len(found),disjoint=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    namespace_check(a.out)
    if not (a.out/'archive.pt').exists():export_archive(a.out)
    if not (a.out/'teacher.pt').exists():teacher_data(a.out)

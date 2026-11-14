"""Bounded contracts; no strength-based changes or GPU allocation."""
from sc_train import *
import tempfile,subprocess

def recovery_probe(path,device='cpu',steps=32):
    outputs=[]
    for repeat in range(2):
        t=ScalingTrainer.load(path,device)
        # Compare cold versus warm inference caches; neither may affect learner RNG.
        POLICY_CACHE.clear()
        if repeat:
            for policy in initial_archive()+t.archive:frozen(policy)
        before=t.n
        for _ in range(steps):t.collect()
        t.update()
        spec=export(t.net,t.kind,'probe',t.n)
        outputs.append(dict(weights=spec['weight_sha256'],obs=t.env.obs().tolist(),n=t.n,learner_rng=t.learner_rng.bit_generator.state,pool_rng=repr(t.pool_rng.getstate()),opponent_rng=t.env.neural_rng.bit_generator.state,game_rng=repr(t.env.game.random.getstate()),archive=[s['id'] for s in t.archive]))
        assert t.n==before+steps
    differing=[key for key in outputs[0] if outputs[0][key]!=outputs[1][key]]
    assert not differing,'INTEGRITY: resume continuation diverged: '+','.join(differing)
    return dict(bitwise_weights_rng_observation_equal=True,steps=steps,weights=outputs[0]['weights'])

def checks(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);results={}
    assert count_params(model('mlp128'))==172109 and count_params(model('mlp512'))==884045
    assert .8*884045<=count_params(model('graph'))<=1.2*884045
    e=ScalingEnv(NAMESPACES['checks'],0,'weighted');old=observe(e.game,e.color)
    assert np.array_equal(compact(old)*DIV,old[KEEP]);assert len(set(FEATURE_NAMES))==879
    # Same physical actor despite swapping seat index; actor-relative action roundtrips.
    for color in COLORS:
        for i in range(332):assert action_index(decode(i,color),color)==i
    for i in range(128):
        assert e.game.state.current_color()==e.color
        for a in e.game.playable_actions:assert e.mask()[action_index(a,e.color)]
        _,r,term,trunc,_=e.step(int(np.flatnonzero(e.mask())[i%sum(e.mask())]))
        if term or trunc:e=ScalingEnv(NAMESPACES['checks']+i+1,i%2,'weighted')
    # Hidden state mutation with unchanged permitted facts leaves all models invariant.
    clone=copy.deepcopy(e);s=clone.game.state;other=next(c for c in s.colors if c!=clone.color);key=player_key(s,other)
    s.development_listdeck=list(reversed(s.development_listdeck));s.random=random.Random(444);clone.game.random=s.random
    hand=[s.player_state[key+'_'+r+'_IN_HAND'] for r in RESOURCES]
    for r,v in zip(RESOURCES,hand[1:]+hand[:1]):s.player_state[key+'_'+r+'_IN_HAND']=v
    s.player_state[key+'_ACTUAL_VICTORY_POINTS']+=1
    assert np.array_equal(e.obs(),clone.obs()) and np.array_equal(e.mask(),clone.mask())
    for kind in ['mlp128','mlp512','graph']:
        net=model(kind).eval()
        with torch.no_grad():assert torch.equal(net(torch.from_numpy(e.obs()))[0],net(torch.from_numpy(clone.obs()))[0])
    # Conservation, planning isolation and hidden-world invariance are inherited and retested.
    # Explicit isolated chance-state check, independently of the inherited test names.
    p=e.planners[e.color];live=e.game.random.getstate();p.decide(e.game,e.knowledge[e.color]);assert live==e.game.random.getstate()
    results['schema_privacy_mask_turns']=True
    # Terminal suppresses bootstrap; truncation bootstraps but breaks GAE chaining.
    row=lambda r,term,trunc,v,nv:(None,None,None,r,term,trunc,v,nv,None)
    assert np.allclose(advantages([row(0,False,True,.2,.7),row(1,True,False,.3,0)]),[.5,.7])
    results['reward_gae']=True
    for condition in ['A','B','E','G']:
        t=ScalingTrainer(condition,SEEDS[0]);initial=export(t.net,t.kind,'init',0)
        if condition in ['A','B']:
            if condition=='A':reference=initial['weight_sha256']
            else:assert initial['weight_sha256']==reference
        for _ in range(64):t.collect()
        t.update();t.save(folder/f'{condition}.pt');results[condition]=recovery_probe(folder/f'{condition}.pt')
    t=ScalingTrainer('B',SEEDS[0]);t.n=65536;t.publish();assert t.last_publication==65536 and len(t.archive)==1
    for i in range(2,15):t.n=i*65536;t.publish()
    assert len(t.archive)==8 and t.archive[-1]['step']==14*65536
    old_ids=[s['id'] for s in t.archive];t.publish();assert old_ids==[s['id'] for s in t.archive]
    # Pin the actual own-architecture neural opponent across publication and recovery.
    t.env=ScalingEnv(NAMESPACES['checks']+999,0,'neural',policy=t.archive[-1]);t.opponent_id=t.archive[-1]['id'];t.opponent_step=t.archive[-1]['step'];t.save(folder/'frozen-opponent.pt')
    results['frozen_opponent_recovery']=recovery_probe(folder/'frozen-opponent.pt')
    # Real sampler frequencies, no descriptive-only mixture fields.
    sampler=ScalingTrainer('A',SEEDS[0]);counts=Counter(sampler.choose_opponent()[0] for _ in range(20000))
    assert abs(counts['base']/20000-.25)<.02 and abs(counts['neural']/20000-.5)<.02
    results['mixture_counts']=dict(counts)
    teacher=torch.load(ROOT/'teacher.pt',map_location='cpu',weights_only=False)
    sets=[{r['source_game'] for r in teacher[k]} for k in ['train','validation','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert sum(map(len,teacher.values()))<=25000
    assert all(r['mask'][r['action']] and len(r['obs'])==879 for rows in teacher.values() for r in rows)
    results['teacher_source_split_legal']=True;write(folder/'checks.json',results);return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--probe');p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.probe:write(a.out,recovery_probe(a.probe,a.device))
    else:print(json.dumps(checks(a.out)))

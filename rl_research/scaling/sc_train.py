"""Instrumented unchanged PPO equations, full-state atomic recovery, frozen self-play."""
from sc_env import *
import argparse,copy,functools
initial_archive=functools.lru_cache(maxsize=1)(initial_archive)

def export(net,kind,identifier,step):
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return dict(id=identifier,schema=SCHEMA,kind=kind,state_dict=state,step=step,weight_sha256=h.hexdigest())

def advantages(data,lam=.95):
    gae=0.;out=[]
    for _,_,_,r,terminal,truncated,v,nv,_ in reversed(data):
        assert not terminal or nv==0
        gae=r+nv-v+lam*(not(terminal or truncated))*gae;out.append(gae)
    return np.array(out[::-1],np.float32)

class ScalingTrainer:
    def __init__(self,condition,seed,device='cpu',teacher=True,pilot=False):
        self.namespace=(NAMESPACES['pilot']+list(CONDITIONS).index(condition)*1000000) if pilot else (NAMESPACES['train']+SEEDS.index(seed)*100000000)
        self.condition=condition;self.seed=seed;self.kind,self.schedule=CONDITIONS[condition];self.device=device
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.net=model(self.kind,seed).to(device);self.initial_weight_sha256=export(self.net,self.kind,'initial',0)['weight_sha256'];self.learner_rng=np.random.default_rng(seed+1001);self.pool_rng=random.Random(seed+2001);self.seat_rng=random.Random(seed+3001)
        self.n=0;self.updates=0;self.elapsed=0.;self.update_seconds=0.;self.teacher_seconds=0.;self.teacher_info=None
        self.episode=0;self.rollout=[];self.metrics=[];self.episodes=[];self.failures=[];self.engine_actions=0
        self.archive=[];self.last_publication=0;self.archive_events=[];self.transition_shares=Counter();self.episode_shares=Counter();self.snapshot_age_sum=Counter();self.component_episode_shares=Counter();self.component_transition_shares=Counter()
        self.source_hashes=hashes();self.schema=SCHEMA;self.mapping_hash=MAPPING_HASH
        if condition=='G' and teacher:self.teach()
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=PPO['lr']);self.env=self.new_env()
    def tensor(self,x,dtype=torch.float32):return torch.as_tensor(np.asarray(x),dtype=dtype,device=self.device)
    def teach(self):
        start=time.perf_counter();data=torch.load(ROOT/'teacher.pt',map_location='cpu',weights_only=False)
        train=data['train'];strata=['setup','early','remaining'];ids=[[i for i,x in enumerate(train) if x['stratum']==s] for s in strata]
        weights=np.array([.2,.4,.4])*np.array([bool(x) for x in ids]);weights/=weights.sum()
        rng=np.random.default_rng(self.seed+4101);optimizer=torch.optim.Adam(list(self.net.body.parameters())+list(self.net.policy.parameters()),lr=.0003)
        updates=0;presentations=0
        for epoch in range(3):
            for offset in range(0,len(train),256):
                size=min(256,len(train)-offset);selected=[train[int(rng.choice(ids[int(s)]))] for s in rng.choice(3,size=size,p=weights)]
                logits,_=self.net(self.tensor([r['obs'] for r in selected]));m=self.tensor([r['mask'] for r in selected],torch.bool);a=self.tensor([r['action'] for r in selected],torch.long)
                loss=torch.nn.functional.cross_entropy(masked(logits,m),a);optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;presentations+=size
        agreement={}
        for split in ['validation','test']:
            agreement[split]={}
            for stratum in strata:
                rows=[r for r in data[split] if r['stratum']==stratum];correct=0
                for offset in range(0,len(rows),256):
                    batch=rows[offset:offset+256]
                    with torch.no_grad():q,_=self.net(self.tensor([r['obs'] for r in batch]));pred=masked(q,self.tensor([r['mask'] for r in batch],torch.bool)).argmax(-1)
                    correct+=int((pred==self.tensor([r['action'] for r in batch],torch.long)).sum())
                agreement[split][stratum]=dict(correct=correct,total=len(rows))
        self.teacher_seconds=time.perf_counter()-start
        self.teacher_info=dict(updates=updates,presentations=presentations,stored_records=len(train),unique_states=len({hashlib.sha256(r['obs'].tobytes()+r['mask'].tobytes()).hexdigest() for r in train}),source_games=len({r['source_game'] for r in train}),stratum_probabilities=weights.tolist(),agreement=agreement,dataset_sha256=sha(ROOT/'teacher.pt'),seconds=self.teacher_seconds)
    def choose_opponent(self):
        u=self.pool_rng.random()
        if u<.25:self.pool_component='anchor_base';return 'base',None
        if u<.375:self.pool_component='anchor_weighted';return 'weighted',None
        if u<.5:self.pool_component='anchor_heuristic';return 'heuristic',None
        initial=initial_archive()[:-1]
        if u<.75:
            self.pool_component='latest_or_reference'
            spec=self.archive[-1] if self.schedule=='refresh' and self.archive else next(x for x in initial if x['id']=='standard-104729')
        else:
            self.pool_component='archive'
            spec=self.pool_rng.choice(initial+(self.archive if self.schedule=='refresh' else []))
        return 'neural',spec
    def new_env(self):
        self.episode+=1;opponent,spec=self.choose_opponent();seed=self.namespace+self.episode
        self.opponent_id=spec['id'] if spec else opponent;self.opponent_step=spec.get('step') if spec else None;self.episode_shares[self.opponent_id]+=1;self.episode_component=self.pool_component;self.component_episode_shares[self.episode_component]+=1
        e=ScalingEnv(seed,self.seat_rng.randrange(2),opponent,seed+900000000,policy=spec);self.engine_actions+=e.actions
        return e
    def publish(self,folder=None):
        if self.schedule!='refresh' or self.n==0 or self.n%65536 or self.n<=self.last_publication:return
        assert not self.rollout
        spec=export(self.net,self.kind,f'{self.condition}-{self.seed}-{self.n}',self.n);self.archive.append(spec)
        if len(self.archive)>8:
            # Deterministic online temporal coverage: remove closest interior pair's older member.
            times=[s['step'] for s in self.archive]
            j=min(range(1,len(times)-1),key=lambda i:(times[i+1]-times[i-1],times[i]))
            self.archive.pop(j)
        self.last_publication=self.n;self.archive_events.append(dict(n=self.n,retained=[x['id'] for x in self.archive],sha256=spec['weight_sha256']))
        if folder:atomic_save(spec,Path(folder)/'opponent-snapshots'/f'{self.n}.pt')
    def collect(self):
        x,m=self.env.obs(),self.env.mask()
        with torch.no_grad():
            logits,v=self.net(self.tensor(x));lp=masked(logits,self.tensor(m,torch.bool)).log_softmax(-1);p=lp.exp().cpu().numpy().astype(np.float64);p/=p.sum()
            a=int(self.learner_rng.choice(332,p=p));logp=float(lp[a].item());value=float(v.item())
        self.env.record_decision(a,m,p,check_win=True,observation=x)
        nx,r,terminal,truncated,actions=self.env.step(a)
        with torch.no_grad():nv=0. if terminal else float(self.net(self.tensor(nx))[1].item())
        self.rollout.append((x,m,a,r,terminal,truncated,value,nv,logp));self.n+=1;self.engine_actions+=actions
        self.transition_shares[self.opponent_id]+=1;self.component_transition_shares[self.episode_component]+=1
        if self.opponent_step is not None:self.snapshot_age_sum[self.opponent_id]+=self.n-self.opponent_step
        if terminal or truncated:
            d=self.env.diagnostic();self.episodes.append(dict(n=self.n,opponent=self.opponent_id,opponent_step=self.opponent_step,pool_component=self.episode_component,reward=r,terminal=terminal,truncated=truncated,points=self.env.points(),**d))
            if truncated and len(self.failures)<10:self.failures.append(dict(n=self.n,diagnostic=d,trace=self.env.trace))
            self.env=self.new_env()
    def update(self):
        start=time.perf_counter();data=self.rollout;raw=self.tensor(advantages(data));values=self.tensor([d[6] for d in data]);returns=raw+values
        adv=(raw-raw.mean())/(raw.std(unbiased=False)+1e-8);x=self.tensor([d[0] for d in data]);m=self.tensor([d[1] for d in data],torch.bool);a=self.tensor([d[2] for d in data],torch.long);old=self.tensor([d[8] for d in data]);stats=[]
        for _ in range(4):
            for idx in torch.randperm(len(data),device=self.device).split(256):
                q,v=self.net(x[idx]);dist=torch.distributions.Categorical(logits=masked(q,m[idx]));logratio=dist.log_prob(a[idx])-old[idx];ratio=logratio.exp()
                pg=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean();vl=(v-returns[idx]).square().mean();entropy=dist.entropy().mean();loss=pg+.5*vl-.01*entropy
                if not torch.isfinite(loss):raise RuntimeError('INTEGRITY: nonfinite PPO loss')
                self.optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),.5,error_if_nonfinite=True);self.optimizer.step();self.updates+=1
                stats.append([z.detach().item() for z in [pg,vl,entropy,((ratio-1)-logratio).mean(),((ratio-1).abs()>.2).float().mean(),grad,loss]])
        var=float(returns.var(unbiased=False));ev=None if var==0 else float(1-(returns-values).var(unbiased=False)/var)
        row=dict(zip(['policy_loss','value_loss','entropy','approx_kl','clip_fraction','grad_norm_before_clip','loss'],np.mean(stats,0).tolist()))
        self.metrics.append(row|dict(n=self.n,updates=self.updates,explained_variance=ev,advantage_mean=float(raw.mean()),advantage_std=float(raw.std(unbiased=False)),return_mean=float(returns.mean()),return_std=float(returns.std(unbiased=False)),value_mean=float(values.mean()),nonfinite_count=0,elapsed=self.elapsed))
        self.update_seconds+=time.perf_counter()-start;self.rollout=[]
    def save(self,path):
        atomic_save(dict(trainer=self,rng=snapshot_rng(),source_hashes=self.source_hashes,schema=SCHEMA,mapping_hash=MAPPING_HASH,ppo=PPO),path)
    @staticmethod
    def load(path,device=None):
        s=torch.load(path,map_location=device or 'cpu',weights_only=False)
        assert s['source_hashes']==hashes() and s['schema']==SCHEMA and s['mapping_hash']==MAPPING_HASH and s['ppo']==PPO,'INTEGRITY: checkpoint provenance'
        t=s['trainer'];restore_rng(s['rng'])
        if device is not None:t.device=device;t.net.to(device)
        if t.env.frozen_opponent is not None:
            t.env.frozen_opponent.net.cpu()
            t.env.frozen_opponent.spec['state_dict']={k:v.cpu() for k,v in t.env.frozen_opponent.spec['state_dict'].items()}
        for spec in t.archive:spec['state_dict']={k:v.cpu() for k,v in spec['state_dict'].items()}
        return t
    def report(self,folder):
        write(Path(folder)/'training.json',dict(n=self.n,elapsed=self.elapsed,teacher_seconds=self.teacher_seconds,teacher=self.teacher_info,metrics=self.metrics,episodes=self.episodes,failures=self.failures,transition_shares=dict(self.transition_shares),episode_shares=dict(self.episode_shares),snapshot_age_sum=dict(self.snapshot_age_sum),component_episode_shares=dict(self.component_episode_shares),component_transition_shares=dict(self.component_transition_shares),archive_events=self.archive_events,engine_actions=self.engine_actions,update_seconds=self.update_seconds,parameters=count_params(self.net)))
    def run(self,folder,target,deadline):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();cp=tick;health=tick
        while self.n<target and time.time()<deadline and not (folder.parent.parent/'pause_requested').exists():
            self.collect()
            if len(self.rollout)==4096:
                self.update();self.metrics[-1]['elapsed']=self.elapsed+time.perf_counter()-tick;self.publish(folder)
            now=time.perf_counter()
            if now-health>=60:
                self.elapsed+=now-tick;tick=now;health=now
                write(folder/'progress.json',dict(n=self.n,target=target,elapsed=self.elapsed,time=time.time(),episode=self.episode,opponent=self.opponent_id,step_rate=self.n/max(1,self.elapsed)))
            if now-cp>=300:
                self.elapsed+=now-tick;tick=now;self.save(folder/'latest.pt');self.report(folder);cp=now
        self.elapsed+=time.perf_counter()-tick
        self.save(folder/'latest.pt');self.report(folder)
        if self.n==target:self.save(folder/f'{target}.pt')
        write(folder/'progress.json',dict(n=self.n,target=target,elapsed=self.elapsed,time=time.time(),step_rate=self.n/max(1,self.elapsed)))
        return self.n==target

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--device',default='cuda');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.folder.mkdir(parents=True,exist_ok=True)
    try:
        if (a.folder/'latest.pt').exists():t=ScalingTrainer.load(a.folder/'latest.pt',a.device)
        else:
            t=ScalingTrainer(a.condition,a.seed,a.device,pilot=a.pilot)
            if a.pilot and t.schedule=='refresh':
                # Populated own-architecture pool for honest later-stage timing, with its own IDs.
                t.archive=[export(t.net,t.kind,f'pilot-own-{i}',0) for i in range(8)];t.env=t.new_env()
            t.save(a.folder/'0.pt');t.save(a.folder/'latest.pt')
        ok=t.run(a.folder,a.target,a.deadline)
        write(a.folder/'chunk-result.json',dict(complete=ok,n=t.n,target=a.target,elapsed=t.elapsed,finished=time.time()))
    except BaseException as exc:
        write(a.folder/'fatal.json',dict(error=repr(exc),time=time.time()));raise
if __name__=='__main__':
    import sc_train
    sc_train.main()

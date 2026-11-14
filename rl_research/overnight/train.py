"""Masked PPO / Double DQN with atomic, exact-state recovery."""
import argparse, copy, hashlib, json, os, random, time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch import nn
from env import Env, TABLE, MAPPING_HASH

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())
torch.set_num_threads(1)
class Net(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.policy = nn.Linear(width,len(TABLE))
        self.value = nn.Linear(width,1)
    def forward(self,x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)

def source_hashes():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['train.py','env.py','config.json','requirements-lock.txt']}

def masked(q,m): return q.masked_fill(~m, -1e9)
def atomic_save(obj,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def double_target(online,target,obs,mask,reward,terminal):
    # Online argmax, target evaluation; terminal rows may have empty masks.
    with torch.no_grad():
        action = masked(online(obs)[0],mask).argmax(-1)
        return reward + (~terminal).float()*target(obs)[0].gather(1,action[:,None]).squeeze(1)

class Trainer:
    def __init__(self,algo,width,seed,target,device='cpu'):
        self.algo,self.width,self.seed,self.target,self.device = algo,width,seed,target,device
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.episode = 0; self.env = self.new_env()
        self.net = Net(len(self.env.obs()),width).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=CONFIG[algo]['lr'])
        self.n = self.updates = self.engine_actions = 0
        self.elapsed = self.update_seconds = 0.
        self.replay = deque(maxlen=CONFIG['ddqn']['replay_capacity'])
        self.metrics = []; self.episodes = []; self.rollout = []
    def new_env(self):
        self.episode += 1
        return Env(self.seed*10000+self.episode,self.rng.randrange(2))
    def tensor(self,x,dtype=torch.float32): return torch.as_tensor(np.asarray(x),dtype=dtype,device=self.device)
    def epsilon(self):
        c=CONFIG['ddqn']; return max(c['epsilon_end'], c['epsilon_start']-(c['epsilon_start']-c['epsilon_end'])*self.n/max(1,self.target*c['epsilon_fraction']))
    def collect(self):
        x,m=self.env.obs(),self.env.mask()
        with torch.no_grad():
            q,v=self.net(self.tensor(x)); dist=torch.distributions.Categorical(logits=masked(q,self.tensor(m,torch.bool)))
            if self.algo=='ppo': a=int(dist.sample().item())
            elif self.rng.random()<self.epsilon(): a=int(self.rng.choice(np.flatnonzero(m)))
            else: a=int(masked(q,self.tensor(m,torch.bool)).argmax().item())
            logp=float(dist.log_prob(torch.tensor(a,device=self.device)).item())
        nx,r,term,trunc,actions=self.env.step(a)
        nm=self.env.mask()
        with torch.no_grad(): nv=float(self.net(self.tensor(nx))[1].item()) if not term else 0.
        if self.algo=='ppo': self.rollout.append((x,m,a,r,term,trunc,float(v.item()),nv,logp))
        else: self.replay.append((x,m,a,r,term,trunc,nx,nm))
        self.n+=1; self.engine_actions+=actions
        if term or trunc:
            self.episodes.append(dict(n=self.n,reward=r,terminal=term,truncated=trunc,points=self.env.points(),actions=self.env.actions,opponent=self.env.opponent))
            self.env=self.new_env(); self.engine_actions+=self.env.actions
    def update(self):
        c=CONFIG[self.algo]; start=time.perf_counter()
        losses=[]
        if self.algo=='ppo':
            data=self.rollout; adv=[]; gae=0.
            for x,m,a,r,term,trunc,v,nv,lp in reversed(data):
                delta=r+nv-v
                gae=delta+c['gae_lambda']*(not(term or trunc))*gae
                adv.append(gae)
            adv=self.tensor(adv[::-1]); values=self.tensor([d[6] for d in data]); returns=adv+values
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            x=self.tensor([d[0] for d in data]); masks=self.tensor([d[1] for d in data],torch.bool)
            actions=self.tensor([d[2] for d in data],torch.long); old=self.tensor([d[8] for d in data])
            for _ in range(c['epochs']):
                for ids in torch.randperm(len(data),device=self.device).split(c['batch']):
                    logits,value=self.net(x[ids]); dist=torch.distributions.Categorical(logits=masked(logits,masks[ids]))
                    ratio=(dist.log_prob(actions[ids])-old[ids]).exp()
                    loss=-torch.minimum(ratio*adv[ids],ratio.clamp(1-c['clip'],1+c['clip'])*adv[ids]).mean()+c['value_coefficient']*(value-returns[ids]).square().mean()-c['entropy']*dist.entropy().mean()
                    self.optimize(loss,c['max_grad_norm']); losses.append(loss.item())
            self.rollout=[]
        else:
            data=self.rng.sample(list(self.replay),c['batch'])
            x=self.tensor([d[0] for d in data]); nx=self.tensor([d[6] for d in data]); nm=self.tensor([d[7] for d in data],torch.bool)
            actions=self.tensor([d[2] for d in data],torch.long)
            y=double_target(self.net,self.target_net,nx,nm,self.tensor([d[3] for d in data]),self.tensor([d[4] for d in data],torch.bool))
            q=self.net(x)[0].gather(1,actions[:,None]).squeeze(1)
            loss=nn.functional.smooth_l1_loss(q,y); self.optimize(loss,c['max_grad_norm']); losses.append(loss.item())
            if self.updates % c['target_every']==0: self.target_net.load_state_dict(self.net.state_dict())
        self.update_seconds+=time.perf_counter()-start
        self.metrics.append(dict(n=self.n,loss=float(np.mean(losses)),updates=self.updates,epsilon=self.epsilon() if self.algo=='ddqn' else None))
    def optimize(self,loss,clip):
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(),clip); self.optimizer.step(); self.updates+=1
    def save(self,path):
        atomic_save(dict(trainer=self,config=CONFIG,source_hashes=source_hashes(),mapping_hash=MAPPING_HASH,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),path)
    @staticmethod
    def load(path):
        s=torch.load(path,weights_only=False)
        assert s['config']==CONFIG and s['mapping_hash']==MAPPING_HASH
        assert s['source_hashes']==source_hashes(), 'checkpoint source mismatch'
        random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng']); torch.set_rng_state(s['torch_rng'])
        if s['cuda_rng']: torch.cuda.set_rng_state_all(s['cuda_rng'])
        return s['trainer']
    def run(self,path,stop_n,deadline):
        start=time.monotonic(); last=start
        while self.n<stop_n and time.time()<deadline:
            self.collect()
            c=CONFIG[self.algo]
            if self.algo=='ppo' and (len(self.rollout)>=c['rollout'] or self.n==stop_n): self.update()
            if self.algo=='ddqn' and self.n>=c['warmup'] and self.n%c['update_every']==0: self.update()
            now=time.monotonic()
            if now-last>=CONFIG['checkpoint_seconds']:
                self.elapsed+=now-start; start=now; self.save(path); last=now
                Path(path).parent.joinpath('progress.json').write_text(json.dumps(dict(n=self.n,target=self.target,elapsed=self.elapsed,updates=self.updates)))
        self.elapsed+=time.monotonic()-start; self.save(path)
        return dict(n=self.n,complete=self.n>=stop_n,elapsed=self.elapsed,learner_decisions_s=self.n/max(self.elapsed,1e-9),engine_actions_s=self.engine_actions/max(self.elapsed,1e-9),update_seconds=self.update_seconds,episodes=len(self.episodes),terminal_rewards=sum(e['terminal'] for e in self.episodes),parameters=sum(p.numel() for p in self.net.parameters()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--algo',choices=['ppo','ddqn'],required=True); p.add_argument('--width',type=int,default=256); p.add_argument('--seed',type=int,default=104729); p.add_argument('--target',type=int,required=True); p.add_argument('--seconds',type=int,default=2700); p.add_argument('--device',default='cpu'); p.add_argument('--checkpoint',required=True); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    t=Trainer.load(a.checkpoint) if a.resume else Trainer(a.algo,a.width,a.seed,a.target,a.device)
    assert (t.algo,t.width,t.seed,t.target)==(a.algo,a.width,a.seed,a.target), 'resume settings differ from checkpoint'
    print(json.dumps(t.run(a.checkpoint,a.target,time.time()+a.seconds)))
if __name__=='__main__':
    import train
    train.main()

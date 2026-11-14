"""Orchestration and diagnostics only; the archived learning implementation is immutable."""
import hashlib, json, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from env import Env, RESOURCES, DEVELOPMENT_CARDS, player_key
from train import Trainer

STEPS = [106496, 217088, 323584, 434176]
OLD_SEEDS = [104729, 130363]
NEW_SEEDS = [155921, 196613, 262147]
RUNS = [(w,s) for i,s in enumerate(NEW_SEEDS) for w in ([128,256] if i%2==0 else [256,128])]
ALL_RUNS = [(w,s) for s in OLD_SEEDS+NEW_SEEDS for w in [128,256]]

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)

def read(path): return json.loads(Path(path).read_text())
def name(width,seed): return f'ppo-{width}-{seed}'
def rng_state(): return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2])
    if s[3]: torch.cuda.set_rng_state_all(s[3])

def public_key(env):
    """Only public pieces/robber and own exact hand; excludes clock and opponent hand."""
    s=env.game.state; b=s.board; k=player_key(s,env.color)
    hand=tuple(s.player_state[k+'_'+r+'_IN_HAND'] for r in RESOURCES+DEVELOPMENT_CARDS)
    return repr((sorted(b.buildings.items()),sorted(b.roads.items()),b.robber_coordinate,hand))

class DiagnosticEnv(Env):
    def __init__(self,*args,**kwargs):
        self.action_counts={'learner':Counter(),'opponent':Counter()}; self.public_states=Counter()
        super().__init__(*args,**kwargs)
    def execute(self,a):
        super().execute(a)
        self.action_counts['learner' if a.color==self.color else 'opponent'][a.action_type.value]+=1
    def step(self,a):
        self.public_states[hashlib.sha256(public_key(self).encode()).hexdigest()]+=1
        return super().step(a)
    def diagnostic(self):
        s=self.game.state; other=next(c for c in s.colors if c!=self.color)
        counts={k:dict(v) for k,v in self.action_counts.items()}
        return dict(action_type_counts=counts,
            end_turn_frequency={k:v.get('END_TURN',0)/max(1,sum(v.values())) for k,v in counts.items()},
            maritime_trades={k:v.get('MARITIME_TRADE',0) for k,v in counts.items()},
            repeated_public_own_hand_states=sum(n-1 for n in self.public_states.values()),
            unique_public_own_hand_states=len(self.public_states),max_state_visits=max(self.public_states.values(),default=0),
            most_repeated_state_hashes=self.public_states.most_common(5),
            turns=s.num_turns,engine_actions=self.actions,own_points=self.points(),
            opponent_public_points=s.player_state[player_key(s,other)+'_VICTORY_POINTS'],
            turn_limit_reached=s.num_turns>=self.max_turns,action_limit_reached=self.actions>=self.max_actions)

class DiagnosticTrainer(Trainer):
    def new_env(self):
        self.episode+=1
        return DiagnosticEnv(self.seed*10000+self.episode,self.rng.randrange(2))
    def collect(self):
        old=self.env
        super().collect()
        if old is not self.env and self.episodes[-1]['truncated']:
            if not hasattr(self,'truncation_diagnostics'): self.truncation_diagnostics=[]
            self.truncation_diagnostics.append(dict(n=self.n,episode=self.episode-1,opponent=old.opponent,**old.diagnostic()))

def selection(candidates):
    """Candidates must use equal complete board coverage. Ties: fewer stalls, earlier step."""
    assert candidates and len({len(c['rows']) for c in candidates})==1
    assert all(c['rows'] and len(c['rows'])%6==0 for c in candidates)
    return max(candidates,key=lambda c:(sum(r['win'] for r in c['rows'])/len(c['rows']),-sum(r['truncated'] for r in c['rows']),-c['step']))

def fixed_deadline(path,deadline):
    if Path(path).exists():
        old=read(path); assert old['deadline']==deadline,'resume must retain the original deadline'
        return old
    result=dict(deadline=deadline,allocation_cap_seconds=18000,retrieval_reserve_seconds=1200)
    write(path,result); return result

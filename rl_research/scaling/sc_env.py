"""Fixed rules, canonical action ordering, isolated neural opponents and diagnostics."""
from sc_models import *
from curriculum_core import ExperimentEnv
from opponent_adapter import private_copy
from collections import Counter
POLICY_CACHE={}

def mask_for(game,color):
    m=np.zeros(332,bool)
    for a in game.playable_actions:m[action_index(a,color)]=True
    return m
class FrozenPolicy:
    def __init__(self,spec):
        self.spec=spec
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.net=MLP(128,2049) if spec['schema']=='legacy-2049' else model(spec['kind'])
        self.net.load_state_dict(spec['state_dict']);self.net.cpu().eval()
        for p in self.net.parameters():p.requires_grad_(False)
    def act(self,game,color,rng):
        x=observe(game,color);x=x if self.spec['schema']=='legacy-2049' else compact(x)
        mask=mask_for(game,color)
        with torch.no_grad():q,_=self.net(torch.from_numpy(x));probs=masked(q,torch.from_numpy(mask)).softmax(-1).numpy().astype(np.float64)
        probs/=probs.sum();return decode(int(rng.choice(332,p=probs)),color)

def frozen(spec):
    key=spec['id']
    if key not in POLICY_CACHE:
        if len(POLICY_CACHE)>24:POLICY_CACHE.clear()
        POLICY_CACHE[key]=FrozenPolicy(spec)
    return POLICY_CACHE[key]

def initial_archive():return torch.load(ROOT/'archive.pt',map_location='cpu',weights_only=False)

class ScalingEnv(ExperimentEnv):
    def __init__(self,seed,seat,opponent,opponent_rng_seed=None,policy=None):
        self.frozen_opponent=frozen(policy) if policy is not None else None
        self.neural_rng=np.random.default_rng(opponent_rng_seed if opponent_rng_seed is not None else seed+600000000)
        self.first_city=None;self.third_settlement=None;self.settlement_actions=0
        self.opportunities=Counter();self.chosen=Counter();self.probability_sums=Counter();self.phase_decisions=Counter();self.trace=[]
        super().__init__(seed,seat,opponent)
    def execute(self,a):
        turn=self.game.state.num_turns
        if a.color==self.color:
            if a.action_type.value=='BUILD_CITY' and self.first_city is None:self.first_city=turn
            if a.action_type.value=='BUILD_SETTLEMENT':
                self.settlement_actions+=1
                if self.settlement_actions==3:self.third_settlement=turn
        super().execute(a)
        self.game.playable_actions.sort(key=lambda a:action_index(a,a.color))
    def advance(self):
        count=0
        self.game.playable_actions.sort(key=lambda a:action_index(a,a.color))
        while self.game.winning_color() is None and self.game.state.current_color()!=self.color:
            c=self.game.state.current_color()
            if self.frozen_opponent is not None:a=self.frozen_opponent.act(self.game,c,self.neural_rng)
            elif self.opponent in ('base','contender'):a=self.planners[c].decide(self.game,self.knowledge[c])
            else:a=opponent_action(self.game,self.opponent,self.rng)
            self.execute(a);count+=1
            if count>20000:raise RuntimeError('opponent advancement guard')
    def obs(self):return compact(observe(self.game,self.color))
    def record_decision(self,a,mask,probs,check_win=False,observation=None):
        phase='setup' if self.game.state.is_initial_build_phase else self.game.state.current_prompt.value
        self.phase_decisions[phase]+=1
        counts=np.bincount(TYPE_IDS,weights=mask,minlength=len(TYPES));mass=np.bincount(TYPE_IDS,weights=probs,minlength=len(TYPES))
        for i,kind in enumerate(TYPES):
            key=phase+'/'+kind;self.opportunities[key]+=int(counts[i]>0);self.probability_sums[key]+=float(mass[i])
        self.chosen[phase+'/'+TABLE[a][0].value]+=1
        if check_win and self.points()>=8:
            winning=[]
            for action in self.game.playable_actions:
                if action.action_type.value not in ['BUILD_SETTLEMENT','BUILD_CITY','BUILD_ROAD','PLAY_KNIGHT_CARD']:continue
                g=private_copy(self.game,random.Random(0));g.execute(action)
                if g.winning_color()==self.color:winning.append(action_index(action,self.color))
            self.opportunities['immediate_win']+=int(bool(winning));self.chosen['immediate_win']+=int(a in winning)
        if len(self.trace)>=64:self.trace.pop(0)
        self.trace.append(dict(turn=self.game.state.num_turns,phase=phase,action=a,points=self.points(),legal=np.flatnonzero(mask).tolist(),legal_probabilities=probs[mask].tolist(),observation=(self.obs() if observation is None else observation).tolist()))
    def diagnostic(self):
        return super().diagnostic()|dict(first_city=self.first_city,third_settlement=self.third_settlement,
            opportunities=dict(self.opportunities),chosen=dict(self.chosen),probability_sums=dict(self.probability_sums),phase_decisions=dict(self.phase_decisions))

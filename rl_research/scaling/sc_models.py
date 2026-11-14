"""One lossless normalized information schema; typed graph and semantic candidates."""
from sc_common import *
from torch import nn
KEEP=np.r_[np.arange(855),np.arange(2025,2049)]
assert len(KEEP)==879 and len(TABLE)==332
TAIL=['prompt_'+p.value for p in ActionPrompt]+['acting_self','turn_self','setup','discarding','moving_knight','road_building','free_roads','resolving_trade','turns_already_divided_1000','target_vp','discard_threshold','self_pending_discard','opponent_pending_discard']+['own_age_'+d for d in DEVELOPMENT_CARDS if d!='VICTORY_POINT']
FEATURE_NAMES=list(FEATURES)+[f'exact_tile_{t}_dice_{n}' for t in range(19) for n in range(13)]+TAIL
assert len(FEATURE_NAMES)==879

def divisor(name):
    if name.startswith('P') and name[:2] in ['P0','P1']:
        if 'NUM_RESOURCES' in name:return 95
        if 'NUM_DEVS' in name:return 25
        if any('_'+r+'_IN_HAND' in name for r in RESOURCES):return 19
        if 'VPS' in name:return 10
        if 'CITIES_LEFT' in name:return 4
        if 'SETTLEMENTS_LEFT' in name:return 5
        if 'ROADS_LEFT' in name or 'LONGEST_ROAD' in name:return 15
        for d,total in [('KNIGHT',14),('MONOPOLY',2),('YEAR_OF_PLENTY',2),('ROAD_BUILDING',2),('VICTORY_POINT',5)]:
            if d in name and ('IN_HAND' in name or 'PLAYED' in name):return total
    return {'free_roads':2,'target_vp':10,'discard_threshold':7,'self_pending_discard':95,'opponent_pending_discard':95}.get(name,1)
DIV=np.array([divisor(x) for x in FEATURE_NAMES],np.float32)
def compact(old):
    assert old.shape[-1]==2049
    return old[...,KEEP]/DIV

def schema():return dict(version=SCHEMA,old_dimension=2049,dimension=879,removed_indices=[855,2025],features=[dict(index=i,name=n,divisor=float(DIV[i])) for i,n in enumerate(FEATURE_NAMES)],clipping=False,mapping_hash=MAPPING_HASH)

_map=Game([Player(c) for c in COLORS],seed=0).state.board.map
EDGES=sorted(tuple(sorted(e)) for e in get_edges(_map.land_nodes))
NI=[[FEATURES.index(f'NODE{n}_P{p}_{kind}') for p in range(2) for kind in ['CITY','SETTLEMENT']] for n in range(54)]
EI=[[FEATURES.index(f'EDGE{e}_P{p}_ROAD') for p in range(2)] for e in EDGES]
TI=[[FEATURES.index(f'TILE{t}_{f}') for f in ['HAS_ROBBER','IS_BRICK','IS_DESERT','IS_ORE','IS_SHEEP','IS_WHEAT','IS_WOOD','PROBA']]+list(range(608+t*13,608+(t+1)*13)) for t in range(19)]
USED=set(sum(NI,[])+sum(EI,[])+sum(TI,[]));GI=[i for i in range(879) if i not in USED]
assert len(USED)+len(GI)==879
NT=np.array([[float(_map.tiles_by_id[t] in _map.adjacent_tiles[n]) for t in range(19)] for n in range(54)],np.float32)
NE=np.array([[float(n in e) for e in EDGES] for n in range(54)],np.float32)
def norm(a):return a/np.maximum(a.sum(1,keepdims=True),1)
TYPES=sorted({a.value for a,_ in TABLE});TYPE_IDS=[TYPES.index(a.value) for a,_ in TABLE]
DESC=[];SPATIAL=[]
for kind,val in TABLE:
    d=np.zeros(len(TYPES)+13,np.float32);d[TYPES.index(kind.value)]=1;idx=0
    if kind.value in ['BUILD_CITY','BUILD_SETTLEMENT']:idx=1+int(val)
    elif kind.value=='BUILD_ROAD':idx=55+EDGES.index(tuple(sorted(val)))
    elif kind.value=='MOVE_ROBBER':
        coord,victim=val;idx=127+_map.tiles[coord].id;d[len(TYPES)+10+([None,*COLORS].index(victim))]=1
    elif kind.value=='MARITIME_TRADE':
        for resource in val[:-1]:
            if resource is not None:d[len(TYPES)+RESOURCES.index(resource)]+=1/4
        d[len(TYPES)+5+RESOURCES.index(val[-1])]=1
    elif kind.value in ['DISCARD_RESOURCE','PLAY_MONOPOLY']:d[len(TYPES)+RESOURCES.index(val)]=1
    elif kind.value=='PLAY_YEAR_OF_PLENTY':
        for resource in val:d[len(TYPES)+5+RESOURCES.index(resource)]+=.5
    DESC.append(d);SPATIAL.append(idx)
assert len({(SPATIAL[i],tuple(DESC[i])) for i in range(len(TABLE))})==len(TABLE)

class MLP(nn.Module):
    def __init__(self,width,dim=879):
        super().__init__();self.body=nn.Sequential(nn.Linear(dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh());self.policy=nn.Linear(width,332);self.value=nn.Linear(width,1)
    def forward(self,x):h=self.body(x);return self.policy(h),self.value(h).squeeze(-1)

class Graph(nn.Module):
    def __init__(self,h):
        super().__init__()
        for name,val in [('ni',NI),('ei',EI),('ti',TI),('gi',GI),('spatial',SPATIAL)]:self.register_buffer(name,torch.tensor(val,dtype=torch.long))
        for name,val in [('nt',norm(NT)),('tn',norm(NT.T)),('ne',norm(NE)),('en',norm(NE.T)),('desc',np.array(DESC))]:self.register_buffer(name,torch.tensor(val))
        self.node=nn.Linear(4,h);self.edge=nn.Linear(2,h);self.tile=nn.Linear(21,h)
        self.context=nn.Sequential(nn.Linear(len(GI),h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.rounds=nn.ModuleList([nn.ModuleList([nn.Linear(3*h,h),nn.Linear(2*h,h),nn.Linear(2*h,h)]) for _ in range(3)])
        self.global_head=nn.Sequential(nn.Linear(4*h,h),nn.Tanh())
        self.score=nn.Sequential(nn.Linear(2*h+len(DESC[0]),h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
        self.value=nn.Linear(h,1)
    def forward(self,x):
        single=x.ndim==1
        if single:x=x[None]
        n=torch.tanh(self.node(x[:,self.ni]));e=torch.tanh(self.edge(x[:,self.ei]));t=torch.tanh(self.tile(x[:,self.ti]));c=self.context(x[:,self.gi])
        for ln,le,lt in self.rounds:
            n,e,t=torch.tanh(ln(torch.cat([n,self.nt@t,self.ne@e],-1))),torch.tanh(le(torch.cat([e,self.en@n],-1))),torch.tanh(lt(torch.cat([t,self.tn@n],-1)))
        g=self.global_head(torch.cat([c,n.mean(1),e.mean(1),t.mean(1)],-1))
        local=torch.cat([torch.zeros_like(g[:,None]),n,e,t],1)[:,self.spatial]
        q=self.score(torch.cat([g[:,None].expand(-1,332,-1),local,self.desc[None].expand(len(x),-1,-1)],-1)).squeeze(-1);v=self.value(g).squeeze(-1)
        return (q[0],v[0]) if single else (q,v)
GRID=[128,144,160,176,192]
# Selection is parameter-count only, before any strength measurements.
GRAPH_WIDTH=min(GRID,key=lambda h:abs(count_params(Graph(h))-884045))
def model(kind,seed=0):
    with torch.random.fork_rng(devices=[]):
        # CPU initialization must not reseed CUDA's learner minibatch stream.
        torch.random.default_generator.manual_seed(seed)
        result=Graph(GRAPH_WIDTH) if kind=='graph' else MLP(int(kind[3:]))
    return result

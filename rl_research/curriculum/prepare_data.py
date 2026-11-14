"""CPU-only provenance verification, diagnostic reconstruction and legal corpus generation."""
import argparse,copy,json,random,time,tarfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from curriculum_core import *
from reliability_common import DiagnosticEnv

REPO=Path(__file__).resolve().parents[2]
OLD=Path('/root/tinysettler-artifacts/tinysettler-ablation-20261111-b')
REL=Path('/root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a')

def checkpoint(seed,step=434176):
    return (OLD if seed in (104729,130363) else REL)/'retrieved'/f'ppo-128-{seed}'/f'{step}.pt'

def load_policy(p):
    s=torch.load(p,map_location='cpu',weights_only=False)
    assert s['source_hashes']==source_hashes() and s['config']==CONFIG
    t=s['trainer'];t.device='cpu';return t

def verify_inputs(out):
    checks={}
    for seed in SEEDS+[196613]:
        p=checkpoint(seed);manifest=read(REPO/'rl_research/results'/('overnight-20261111' if seed in (104729,130363) else 'ppo-reliability-20261111')/'artifact-manifest.json')
        key=str(p.relative_to(OLD if seed in (104729,130363) else REL))
        assert sha(p)==manifest['files'][key]['sha256']
        t=load_policy(p);assert t.n==434176 and t.width==128
        checks[str(seed)]=dict(path=str(p),sha256=sha(p),n=t.n)
    write(out/'historical-checks.json',checks)

def action_info(a,e,probs):
    index=action_index(a,e.color);assert decode(index,e.color)==a
    return dict(index=index,type=a.action_type.value,value=repr(a.value),probability=float(probs[index]))

def diagnostic_cases(out):
    rows=read(REPO/'rl_research/results/ppo-reliability-20261111/timeout-diagnostics.json')
    chosen=[r for r in rows if r['split']=='final' and r['width']==128 and r['action_type_counts']['learner'].get('BUY_DEVELOPMENT_CARD')==25 and not r['action_type_counts']['learner'].get('PLAY_KNIGHT_CARD',0) and r['points']==8]
    assert len(chosen)==2
    # Additional representative common-final failure, selected for timeout (not a training source).
    extra=next(r for r in rows if r['split']=='final' and r['width']==128 and r['training_seed']==130363)
    chosen.append(extra);results=[];snapshots=[]
    for case_id,row in enumerate(chosen):
        t=load_policy(checkpoint(row['training_seed']));e=DiagnosticEnv(row['board_seed'],row['seat'],row['opponent'])
        opportunity=Counter();picked={};trace=[]
        while True:
            a,probs=policy_action(t,e);flags=task_flags(e)
            flags['expansion_route']=[x for x in e.game.playable_actions if x.action_type.value=='BUILD_ROAD' and not e.game.state.is_initial_build_phase]
            for task,options in flags.items():
                if options:
                    opportunity[task]+=1
                    # Late enough to diagnose stalled economy but retain at least 200 turns.
                    if task not in picked and (150 if task=='expansion_route' else 400)<=e.game.state.num_turns<=800:
                        picked[task]=copy.deepcopy(e)
            trace.append(dict(turn=e.game.state.num_turns,index=a,type=decode(a,e.color).action_type.value,points=e.points()))
            _,reward,terminal,truncated,_=e.step(a)
            if terminal or truncated:break
        reproduced=(truncated and e.points()==row['points'] and e.action_counts['learner']==Counter(row['action_type_counts']['learner']))
        case=dict(case_id=case_id,source={k:row[k] for k in ['board_seed','seat','opponent','training_seed','step']},reproduced=bool(reproduced),historical_record=row,reconstruction_note='Historical PYTHONHASHSEED was not recorded; CPU versus CUDA execution can also differ. Exact match is asserted only when all outcome and action-type counts match.',final=e.diagnostic(),legal_opportunities=dict(opportunity),positions=[],continuations=[])
        # Limit to one snapshot per distinct physical state; explicit task labels retained.
        seen=set()
        for task,position in picked.items():
            ident=(position.actions,position.game.state.num_turns)
            if ident in seen:continue
            seen.add(ident);a,probs=policy_action(t,position);flags=task_flags(position);flags['expansion_route']=[x for x in position.game.playable_actions if x.action_type.value=='BUILD_ROAD' and not position.game.state.is_initial_build_phase];k=player_key(position.game.state,position.color);ps=position.game.state.player_state
            desc=dict(task=task,turn=position.game.state.num_turns,engine_actions=position.actions,points=position.points(),observation=position.obs().tolist(),legal_mask=position.mask().tolist(),probabilities=probs.tolist(),greedy=action_info(decode(a,position.color),position,probs),legal_actions=[action_info(x,position,probs) for x in position.game.playable_actions],own_development={d:dict(in_hand=ps[k+'_'+d+'_IN_HAND'],played=ps[k+'_PLAYED_'+d],owned_at_start=ps.get(k+'_'+d+'_OWNED_AT_START')) for d in DEVELOPMENT_CARDS},legal_tasks=[key for key,v in flags.items() if v])
            case['positions'].append(desc);snapshots.append(dict(case_id=case_id,task=task,env=position))
            for repeat in range(8):
                for mode in ['greedy','sample']:
                    c=copy.deepcopy(position);future=920000000000+case_id*100000+position.actions*10+repeat
                    c.game.random=random.Random(future);c.game.state.random=c.game.random;c.rng=random.Random(future+981723)
                    policy_rng=np.random.default_rng(future+333) if mode=='sample' else None
                    actions=[];success={key:False for key,v in flags.items() if v};steps=0
                    start_counts=c.action_counts['learner'].copy();initial_army=ps[k+'_HAS_ARMY'];start_pieces=len(c.game.state.buildings_by_color[c.color]['SETTLEMENT'])+len(c.game.state.buildings_by_color[c.color]['CITY'])
                    trade_pending=False
                    while True:
                        act,_=policy_action(t,c,policy_rng);kind=decode(act,c.color).action_type.value
                        if len(actions)<16:actions.append(dict(turn=c.game.state.num_turns,index=act,type=kind))
                        local=task_flags(c) if kind=='MARITIME_TRADE' else {}
                        if kind=='MARITIME_TRADE' and any(action_index(x[0],c.color)==act for x in local.get('productive_trade',[])):trade_pending=True
                        _,r,term,trunc,_=c.step(act);steps+=1
                        for key in success:
                            if key=='knight' and kind=='PLAY_KNIGHT_CARD':success[key]=True
                            if key in ('expansion','expansion_route') and kind=='BUILD_SETTLEMENT':success[key]=True
                            if key=='largest_army' and c.game.state.player_state[k+'_HAS_ARMY']:success[key]=True
                            if key=='productive_trade' and trade_pending and kind in ('BUILD_SETTLEMENT','BUILD_CITY','BUY_DEVELOPMENT_CARD'):success[key]=True
                            if key=='immediate_win' and r==1:success[key]=True
                        if kind=='END_TURN':trade_pending=False
                        if term or trunc:break
                    case['continuations'].append(dict(task=task,repeat=repeat,mode=mode,win=r==1,loss=r==-1,truncated=trunc,points=c.points(),decisions=steps,task_success=success,first_actions=actions,action_counts=dict(c.action_counts['learner']-start_counts)))
        results.append(case);write(out/'historical-diagnostics.json',results)
    atomic_save(snapshots,out/'diagnostic-states.pt')

def paired_preflight(out):
    rows=[]
    for board in range(4):
        for opponent in ['weighted','heuristic']:
            for seat in [0,1]:
                seed=820000000000+board;e=ExperimentEnv(seed,seat,opponent);tick=time.perf_counter()
                while True:
                    c=e.color;a=e.planners[c].decide(e.game,e.knowledge[c]);_,r,term,trunc,_=e.step(action_index(a,c))
                    if term or trunc:break
                rows.append(dict(board_seed=seed,opponent=opponent,seat=seat,win=r==1,loss=r==-1,truncated=trunc,points=e.points(),seconds=time.perf_counter()-tick,**e.diagnostic()))
                write(out/'opponent-preflight.json',rows)

def generate_corpus(out):
    corpus={'snapshots':[],'games':[]};policies=[load_policy(checkpoint(s)) for s in SEEDS]
    # Split source games BEFORE inspecting or choosing any snapshot.
    specs=[('train',830000000000,60),('validation',840000000000,12),('test',850000000000,16)]
    for split,start,count in specs:
        for i in range(count):
            seed=start+i;opponent=['base','weighted','heuristic'][i%3];e=ExperimentEnv(seed,i%2,opponent);rng=np.random.default_rng(seed)
            selected=set();last_turn=-10;game_ids=[];tick=time.perf_counter()
            while True:
                pts=e.points();category='midgame' if 4<=pts<=6 else 'endgame' if 7<=pts<=9 else None
                if category and e.game.state.num_turns<=750 and e.actions<=15000:
                    flags=task_flags(e);tags=[k for k,v in flags.items() if v]
                    # One ordinary and one per task/category/source game; sibling
                    # snapshots remain in this source-game split.
                    eligible=[tag for tag in ['ordinary']+tags if (category,tag) not in selected]
                    if eligible and (e.game.state.num_turns-last_turn>=2 or any(t!='ordinary' for t in eligible)):
                        for tag in eligible:selected.add((category,tag))
                        sid=f'{split}-{seed}-{e.actions}';saved=copy.deepcopy(e)
                        assert saved.mask().any() and saved.game.winning_color() is None
                        other=next(c for c in e.game.state.colors if c!=e.color)
                        snapshot=dict(id=sid,source_game=seed,split=split,category=category,tags=tags,own_points=pts,opponent_public_points=e.game.state.player_state[player_key(e.game.state,other)+'_VICTORY_POINTS'],turn=e.game.state.num_turns,actions=e.actions,env=saved)
                        corpus['snapshots'].append(snapshot);game_ids.append(sid);last_turn=e.game.state.num_turns
                if i%4==0:
                    a=e.planners[e.color].decide(e.game,e.knowledge[e.color]);a=action_index(a,e.color)
                elif i%4==1:a=action_index(opponent_action(e.game,'heuristic',e.rng),e.color)
                else:a,_=policy_action(policies[i%3],e,rng if i%4==2 else None)
                _,r,term,trunc,_=e.step(a)
                if term or trunc:break
            corpus['games'].append(dict(split=split,board_seed=seed,opponent=opponent,seat=i%2,learner_behavior=['adapted-base','legacy-heuristic','sampled-PPO','greedy-PPO'][i%4],snapshot_ids=game_ids,win=r==1,loss=r==-1,truncated=trunc,points=e.points(),seconds=time.perf_counter()-tick))
            if (i+1)%10==0:print(split,i+1,len(corpus['snapshots']),flush=True)
        atomic_save(corpus,out/'corpus.pt')
    metadata=[{k:v for k,v in s.items() if k!='env'} for s in corpus['snapshots']]
    for split,_,_ in specs:
        subset=[s for s in metadata if s['split']==split]
        assert all(any(s['category']==cat for s in subset) for cat in ['midgame','endgame'])
        assert all(any(task in s['tags'] for s in subset) for task in ['expansion','productive_trade','knight','largest_army','immediate_win'])
    write(out/'corpus-manifest.json',dict(sha256=sha(out/'corpus.pt'),splits=specs,games=corpus['games'],snapshots=metadata,selection='one ordinary and first eligible task/category per source game; <=750 turns and <=15000 actions',future_randomness='new dice/steal RNG; physical cards, deck order, phase, ages, history and counters preserved'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['verify','diagnose','preflight','corpus','all'],default='all');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for phase,fn in [('verify',verify_inputs),('diagnose',diagnostic_cases),('preflight',paired_preflight),('corpus',generate_corpus)]:
        if a.phase in (phase,'all'):
            print('starting',phase,flush=True);fn(a.out);print('finished',phase,flush=True)

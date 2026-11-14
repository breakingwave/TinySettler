"""Pure pre-outcome scope/budget selection, explicitly simulated matched schedule."""
from sc_common import *

def ordered_runs(conditions,round_index):
    groups=[['A','B'],['C','D'],['E','F'],['G']];groups=[g for g in groups if g[0] in conditions]
    groups=groups[round_index%len(groups):]+groups[:round_index%len(groups)]
    result=[]
    for si,seed in enumerate(SEEDS):
        for group in groups:
            for c in (group if (si+round_index)%2==0 else group[::-1]):
                if si<2 or c in ['A','B']:result.append((c,seed))
    return result

def projection(conditions,target,workers,pilots):
    total=0.;teacher=0.
    for round_index in range(4):
        tasks=ordered_runs(conditions,round_index)
        for offset in range(0,len(tasks),workers):
            batch=tasks[offset:offset+workers];durations=[]
            for c,s in batch:
                key='dual' if workers==2 else 'single';rate=pilots[c][key]['aggregate_rate']/workers
                durations.append(target/4/rate)
            total+=max(durations)
    if 'G' in conditions:teacher=2*pilots['G']['single']['teacher_seconds']
    # End-to-end rates already include serialization/startup; an additional 20% margin.
    return dict(training_seconds=(total+teacher)*1.2,raw_training_seconds=total,teacher_seconds=teacher)

def freeze(pilots,timing,now,deadline,cpu_count):
    decisions=[];chosen=None
    for conditions in [list('ABCDEFG'),list('ABCDEF'),list('ABCD'),list('AB')]:
        modes=[1]
        if cpu_count>=2 and all(pilots[c]['dual']['aggregate_rate']>pilots[c]['single']['aggregate_rate']*1.02 and pilots[c]['dual']['recovery_passed'] for c in conditions):modes.append(2)
        options=[]
        for workers in modes:
            for target in range(786432,16383,-16384):
                p=projection(conditions,target,workers,pilots)
                # Preserve all planned later-phase allowances plus a recovery reserve.
                available=min(36000,deadline-now-2700-5400-1800-1800)
                if p['training_seconds']<=available:
                    options.append((target,workers,p));break
        best=max(options,key=lambda x:(x[0],-x[2]['training_seconds'])) if options else None
        decisions.append(dict(conditions=conditions,feasible_target=best[0] if best else 0,workers=best[1] if best else None,projection=best[2] if best else None))
        if best and (best[0]>=393216 or conditions==list('AB')):
            chosen=(conditions,*best);break
    if chosen is None:raise RuntimeError('preflight cannot fit a useful matched A/B study')
    conditions,target,workers,projected=chosen
    if target<65536:raise RuntimeError('preflight feasible target below one publication interval')
    kinds={CONDITIONS[c][0] for c in conditions};cost=max(timing[k]['seconds_per_game'] for k in kinds)*1.2
    nr=len(run_ids(conditions))
    # All counts frozen together, uniform across policies. Same validation prefix at milestones.
    validation=None
    for vi in range(5,0,-1):
        vf=2*vi;games=nr*(4*vi*6+vf*12)
        if games*cost<=2700:validation=(vi,vf,games);break
    if validation is None:raise RuntimeError('minimum validation coverage cannot fit')
    coverage=None
    for boards in range(20,1,-1):
        weak=5 if boards>=10 else 2;confirm=100
        games=nr*boards*12+nr*weak*6+3*boards*12+4*(confirm-boards)
        if games*cost<=5400:coverage=(boards,weak,confirm,games);break
    if coverage is None:raise RuntimeError('minimum final coverage cannot fit; no GPU sweep started')
    return dict(conditions=conditions,runs=run_ids(conditions),target=target,milestones=[target*i//4 for i in range(1,5)],workers=workers,
        scope_decisions=decisions,projection=projected,validation_intermediate_boards=validation[0],validation_final_boards=validation[1],validation_games=validation[2],
        final_boards=coverage[0],weak_boards=coverage[1],confirmation_boards=coverage[2],final_games=coverage[3],seconds_per_game_with_margin=cost,
        search=dict(included=False,reason='No repaired depth-two search passed the complete forced-chance/information/exhaustive-reference correctness gate before allocation; main panel prioritized.'),
        primary_mode='sampled',secondary_mode='greedy',selection_rule='final validation equal-weight base/contender capped wins, fewer truncations, parameter count, condition A..G, seed order, sampled before greedy',
        opponent_inference='CPU, one thread, immutable per-episode model, at most 25 cached models per process; learner CUDA',
        snapshot_retention='keep latest and oldest; when >8 remove interior snapshot with smallest neighbor time span, earlier step breaks ties',
        canonical_legal_order='installed actor-relative action ID; fixes cross-process tie enumeration',deadline=deadline,created=now)

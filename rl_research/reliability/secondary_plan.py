"""Freeze secondary coverage from validation timings only, without consulting test outcomes.

Executed once after selection commit. Also corrects the first run's stale in-memory
calibration estimate after recovery. This changes no policy or primary endpoint.
"""
import hashlib,json,os,time
from pathlib import Path
root=Path('/content/artifacts')
frozen=json.loads((root/'frozen.json').read_text());commit=json.loads((root/'selections.json').read_text());selections=commit['selections']
assert (root/'selection_verified').read_text()==hashlib.sha256((root/'selections.json').read_bytes()).hexdigest()
assert not (root/'secondary-plan.json').exists(),'secondary planning is immutable once written'
full=frozen['splits']['final']['boards'];validation_boards=frozen['splits']['validation']['boards']
def cost(sel,step):
    value=json.loads((root/'evaluations/validation'/sel['run']/f'{step}.json').read_text())
    assert len(value['rows'])==validation_boards*6
    return 1.30*sum(r['seconds'] for r in value['rows'])/validation_boards
primary_seconds=full*sum(cost(sel,434176) for sel in selections)
distinct=[sel for sel in selections if sel['step']!=434176]
secondary_seconds_per_board=sum(cost(sel,sel['step']) for sel in distinct)
available=frozen['deadline']-1200-time.time()-primary_seconds-180
boards=min(full,max(0,int(available/max(1e-9,secondary_seconds_per_board))))
plan=dict(boards=boards,policies=len(distinct),seed_start=frozen['splits']['final']['seed_start'],created_at=time.time(),source='expanded validation game durations, multiplied by 1.30; no test outcomes consulted',primary_seconds_reserved=primary_seconds,secondary_seconds_per_board=secondary_seconds_per_board,retrieval_reserve_seconds=1200,selection_sha256=hashlib.sha256((root/'selections.json').read_bytes()).hexdigest(),reason='explicit corrected timing plan after resumed calibration correction')
tmp=root/'secondary-plan.json.tmp';tmp.write_text(json.dumps(plan,indent=2));os.replace(tmp,root/'secondary-plan.json')
print(json.dumps(plan));print('SECONDARY_PLAN_FROZEN')

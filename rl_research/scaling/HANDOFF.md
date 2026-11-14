COMPLETED: authorized recovery succeeded;10/10 A–D runs425984transitions and all evaluation complete. All4runtime allocations stopped. Original elapsed40491.08s <50400s. Results: ../results/scaling-20261114-recovered/findings.md. Do not allocate or resume training. Persistent root /root/tinysettler-artifacts/tinysettler-scaling-20261114-a-recovery2.

ACTIVE RECOVERED EXPERIMENT: /root/tinysettler-artifacts/tinysettler-scaling-20261114-a-recovery2
Supervisor60106, session tinysettler-scaling-20261114-a-r2. Recovery and calibration succeeded; A–D10runs425984transitions,1worker. Next review original hour4 (then hourly). Original deadline unchanged. Do not tune settings. Prior failed reports are historical.

# USER-AUTHORIZED ADDITIONAL RECOVERY (2026-11-14)

Latest user: "try the recovery and get back on experiment plan if sucessful".
This overrides the prior closure for exactly one more fault attempt. Preserve
original started1794619143.8894265 and deadline1794669543.8894265 (15:19 UTC).
Prepare fresh source/artifact root recovery2, asynchronous idempotent installation,
retain latest pilot checkpoints, graph-first calibration E,F,A,B,C,D,G, cold/warm
cache exact recovery, graph external upload/recovery. Max60min renewed preflight
inside the original14h clock. No learning settings or experimental outcomes changed.
After launch use t0+3600*k sparse model reviews; only fatal/completion/user exceptions.

Earlier attempt history follows; closure statements below are historical.

# CLOSED: preflight failed; both runtimes terminated

See ../results/scaling-20261114/findings.md. No experimental runs or tests began.
Fault replacement allowance exhausted; do not resume or allocate another runtime.
Original deadline unchanged. Final termination independently verified via Colab CLI.

# Scaling experiment: one preflight repair launched

Active artifact root: `/root/tinysettler-artifacts/tinysettler-scaling-20261114-a-repair1`.
Preserved failed preflight: `/root/tinysettler-artifacts/tinysettler-scaling-20261114-a`.
Source SHA256: `0816def1c5eda0895563c8f10cfd87ebbafdc87c5956380ca7849ad0e05f9443`.
Dedicated session base: `tinysettler-scaling-20261114-a`.
Persistent supervisor/controller.json is authoritative; it enforces50400seconds
from first allocation request across all attempts. Never reset that deadline.
One planned hour10 handoff and one fault replacement are allowed, sequentially.
The independent watchdog survives supervisor failure.

User requires model progress reviews ONLY at t0+3600*k. Read one compact
hourly-status.json, latest live chart, and at most one bounded log tail. Do not
tune settings. Completion, fatal failure or required user action can interrupt.
`review_wait.py --artifacts ROOT --hour K` waits without reading outcomes and exits
only at the hourly boundary or completion/fatal event. No recurring automation.

Calibration automatically runs fresh8192-step single/dual pilots for A..G, with
own-architecture populated self-play archives, exact CUDA recovery and an external
copy/upload probe. Then it freezes scope (omit G, then E+F, then A/B if necessary),
common target and coverage before experimental outcomes. Search omitted at its
preallocation correctness gate. Do not inspect final tests until selection is
externally hash-committed. All code/inputs were checked before allocation.

Local checks:15 inherited contracts,7 control tests, A/B/E/G collection/update/
recovery, independent-process frozen-opponent recovery,12-game sampled/greedy
panel and idempotent resume. Teacher dataset200 sourcegames,9635/1334/1301 records
for train/validation/test; initial archive6 verified historical policies, held-out
lineage196613. New seeds400009/400031/400033. Compact879 normalized inputs;
parameters172109/884045/932098. Historical files and browser demo untouched.

On completion: independently verify all dedicated sessions absent; run archived
sc_report.py then sc_verify.py on CPU with SCALING_INPUTS=ROOT/inputs. New results
folder rl_research/results/scaling-20261114. Inspect all nine PNG/SVG chart families,
write self-contained findings and ranked recommendation, persist reports outside
sync, verify packages and checkpoint hashes. Report omissions/incomplete cells,
paired seed/board uncertainty, confirmation as one selected model, and runtime
ledger. Do not launch additional training or extend N after viewing results.

Original started1794619143.8894265; immutable deadline1794669543.8894265.
First T4 stopped and absence verified after E-single CUDA recovery divergence.
CPU-only model constructors inadvertently reset CUDA RNG via torch.manual_seed;
fixed to CPU default_generator.manual_seed and tested (9 control checks).
Replacement session tinysettler-scaling-20261114-a-r1, supervisor58973.
Fault allowance exhausted; fresh pilots, no prior sweep/outcomes/scope freeze.
Retry may consume up to1800s recovery reserve; original deadline unchanged.
Do not allocate another fault replacement. A planned hour10 handoff remains bounded.

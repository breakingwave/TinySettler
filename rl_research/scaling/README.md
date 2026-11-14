# Fourteen-hour scaling study

Execution status: **completed after explicitly authorized recovery**.
See [the recovered study report](../results/scaling-20261114-recovered/findings.md):
10 A–D runs completed425984transitions and every frozen evaluation panel finished.
All dedicated runtimes are confirmed stopped. Earlier failed preflights remain
preserved in [their report](../results/scaling-20261114/findings.md).
Do not relaunch the historical example artifact IDs below.

Implements [the frozen design brief](../fourteen-hour-scaling-plan.md). Historical
training sources, checkpoints, reports and browser demo are untouched. New models
use the strict `compact-normalized-879-v1` schema and source/configuration hashes.
`schema.json` lists every feature and divisor. No learned normalization or clipping.
The graph/candidate package has 932,098 parameters (width176 selected only by
parameter count from128/144/160/176/192); MLPs have172,109 and884,045.

Training starts from scratch. A/B, C/D, E/F initial weights match for each seed;
G has B's initialization before three teacher epochs and a fresh PPO optimizer.
All PPO episodes are full games. New seeds400009/400031/400033 and namespaces are
recorded before execution. Teacher split160/20/20sourcegames is assigned before
legal-play generation. Existing historical encoders remain2049-dimensional and
are dispatched by schema; seven exports passed provenance and exact logit checks.
The first six form the initial archive; seed196613 is held out by lineage/order.

The policy models receive identical permitted facts. The information-safe value
opponent retains the existing K=2 public-history sampler, independent planning
RNG and float64 weights. Neural opponents use their own actor-relative encoder,
mask and saved categorical RNG. Inference is CPU/one thread with immutable cached
models, fixed for each episode. Initializing a cached model cannot advance learner
RNG. Engine legal choices are canonicalized by the installed relative action ID
for deterministic tie enumeration and recovery across processes.

One `--gpu T4` at a time. The supervisor preserves50400seconds from the first
allocation request, including downtime/retries. A planned handoff around hour10
and one fault replacement are allowed; both require verified prior termination.
Transient CLI failures do not imply worker death. Three bounded command attempts
use10/30second backoff; persistent connection loss permits recovery after15minutes;
external copies stale for30minutes trigger failure. An independent watchdog stops
known experiment sessions before the hard deadline. Credentials are unchanged.

Workers emit progress each minute and atomic full-state checkpoints every five
minutes. Immutable content-addressed external generations are verified every ten
minutes; the last two are retained. Source/model/optimizer/environment/rollout,
all RNGs, active frozen opponents, retained snapshots and publication step survive
recovery. Completed chunks and evaluation games are skipped on resume. Health,
copying, retries and handoff are deterministic and do not invoke the model.

Calibration uses fresh8192-transition single/dual-process pilots for all seven
conditions, populated own-architecture archives, real PPO updates, recovery tests,
and fresh weak/pilot evaluation games. Pilot weights are discarded. Scope reduction
is G, then E+F, then matched A/B if required; seed3 remains A/B. A common target
multiple of16384 and uniform evaluation coverage are frozen before experimental
outcomes. Exact publication at65536-step boundaries follows complete PPO updates.
No promotion gate, adaptive hyperparameters or performance-based stopping.

Search is explicitly omitted: no repaired depth-two implementation passed the
forced-chance, hidden-information and exhaustive-reference gate before allocation.
Raw stock search is not used. This preserves priority for the main factorial study.

## Preparation and launch

```sh
export PYTHONHASHSEED=0
export PYTHONPATH=rl_research/scaling
export SCALING_INPUTS=/tmp/tinysettler-scaling-inputs
/tmp/tinysettler-rl-venv/bin/python rl_research/scaling/sc_prepare.py \
  --out "$SCALING_INPUTS"
/tmp/tinysettler-rl-venv/bin/python -m pytest -q \
  rl_research/scaling/test_scaling_control.py
/tmp/tinysettler-rl-venv/bin/python rl_research/scaling/sc_checks.py \
  --out /tmp/tinysettler-scaling-checks
/tmp/tinysettler-rl-venv/bin/python rl_research/scaling/sc_bundle.py \
  --inputs "$SCALING_INPUTS" --checks /tmp/tinysettler-scaling-checks \
  --artifacts /root/tinysettler-artifacts/tinysettler-scaling-20261114-a
/tmp/tinysettler-rl-venv/bin/python \
  /root/tinysettler-artifacts/tinysettler-scaling-20261114-a/source/sc_controller.py \
  --artifacts /root/tinysettler-artifacts/tinysettler-scaling-20261114-a \
  --session tinysettler-scaling-20261114-a
```

Run the controller detached with a log on persistent VPS storage. `--resume`
reattaches the same controller state; it must never reset its original deadline.
The controller manages permitted runtime replacement itself. Preparation refuses
to overwrite an existing artifact root. Dependencies remain pinned to the existing
lock and Catanatron revisionecf931181b9a65bb4116a2153fb78c16f1438e00; CLI0.7.5 is
not upgraded during the study. Existing balance before allocation274.95units.

After launch, model reviews occur only at first-request time+3600*k, reading
`hourly-status.json`, `live-charts/01-validation-transitions.png`, and at most one
bounded log tail. Completion/fatal/user-action events can interrupt the wait.
No recurring ChatGPT automation is created. Final-test outcomes are not read by
hourly summaries. Selection is hash-committed externally before tests begin.

## Evaluation and reporting

From the extracted source with `SCALING_INPUTS` set to its associated inputs:

```sh
python sc_eval.py --checkpoint CHECKPOINT.pt --output NEW_OUTPUT.json \
  --start FROZEN_BOARD_PREFIX --boards FROZEN_COUNT \
  --opponents base,contender,heldout --modes sampled,greedy \
  --deadline AUTHORIZED_UNIX_DEADLINE --condition A --seed 400009 --n COMMON_STEP
python sc_report.py --artifacts ARTIFACT_ROOT --output RESULTS_DIRECTORY
python sc_verify.py --artifacts ARTIFACT_ROOT --output RESULTS_DIRECTORY
```

Checkpoint exports include an explicit schema; loading always checks provenance.
The final verifier checks every remote checkpoint hash, initial pairs, CPU recovery,
matched evaluation coverage, source hashes and runtime termination. Reporting runs
on persistent CPU after releasing the T4. The nine PNG/SVG chart families preserve
individual seeds, omitted cells, undefined critic variance, weak/context separation,
and crossed paired-seed/board uncertainty. Three primary pairs and two secondary
pairs remain exploratory. Self-play parity is not external strength. Colab inference
latency is not browser latency; policy probabilities are not calibrated action values.

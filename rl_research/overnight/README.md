# Two-player Catan RL ablation

This experiment is isolated from the browser demo and historical results.
Simulator revision: `ecf931181b9a65bb4116a2153fb78c16f1438e00`.
All resolved Python dependencies are in `requirements-lock.txt`; settings and
seed partitions are in `config.json`. The graph option is omitted, so there
are six training runs. Actor-relative fixed-table observations include explicit
BASE topology and exact dice labels. No public-event history or belief model
is implemented in this initial observation version. Opponent random choices use
an independent per-episode stream. The heuristic ranks legal builds by type,
then settlement/city production, and uses no successor search or hidden state.

## Commands

From the repository root, create an isolated environment and install the lock:

```sh
uv venv /tmp/tinysettler-rl-venv
uv pip install --python /tmp/tinysettler-rl-venv/bin/python -r rl_research/overnight/requirements-lock.txt
/tmp/tinysettler-rl-venv/bin/python -m pytest -q rl_research/overnight/test_contract.py
python rl_research/overnight/controller.py --session UNIQUE --artifacts /root/tinysettler-artifacts/UNIQUE
```

Controller reattachment (same live session and original deadline):

```sh
python rl_research/overnight/controller.py --session UNIQUE --artifacts /root/tinysettler-artifacts/UNIQUE --resume
```

Standalone train/resume inside a compatible environment:

```sh
python rl_research/overnight/train.py --algo ppo --width 128 --seed 104729 --target TARGET --device cuda --checkpoint /persistent/run.pt
python rl_research/overnight/train.py --algo ppo --width 128 --seed 104729 --target TARGET --device cuda --checkpoint /persistent/run.pt --resume
python rl_research/overnight/evaluate.py /persistent/selected.pt --split final --boards 100 --output /persistent/final.json
```

Do not run final evaluation until validation selection is committed. Checkpoint
files pickle trusted experiment state; load only artifacts from this experiment.
The controller enforces a global wall-clock deadline across reattachment and
terminates its dedicated runtime in `finally`; it does not provision replacement
runtimes automatically. There are zero automatic provisioning retries and at most
two cleanup attempts. A host/container crash can interrupt the local watchdog;
reattach promptly using the original manifest. Colab itself does not guarantee
runtime lifetime. The controller downloads and hashes snapshots, then reloads
pilot checkpoints outside Colab before allowing the sweep to start.

PPO uses rollout-time masks, GAE with truncation bootstrap, and no continuation
across reset boundaries. DQN uses online selection and target evaluation,
masked exploration and masked next-action selection. Both use gamma one and
terminal rewards only. Validation evaluates two boards per opponent with swapped
seats at initialization, quarter, midpoint, three-quarter and target; ties favor the earlier checkpoint.
Final evaluation targets 100 paired boards per opponent, reduced uniformly based
on measured evaluation time. Intervals resample board clusters containing both
seats. Two training seeds provide preliminary evidence only. Inference timings
record the Colab network call and action extraction; they are not browser latency,
and PPO probabilities are neither Q-values nor calibrated win probabilities.

Official references: [Colab CLI](https://github.com/googlecolab/google-colab-cli)
and [runtime limits](https://research.google.com/colaboratory/faq.html).

## Replacement runtime after the requested pause

Use a fresh unique session and `--previous` pointing to the prior controller
manifest. This preserves the original deadline and subtracts prior allocation:

```sh
python rl_research/overnight/controller.py --session tinysettler-ablation-20261111-b --artifacts /root/tinysettler-artifacts/tinysettler-ablation-20261111-b --previous /root/tinysettler-artifacts/tinysettler-ablation-20261111-a/controller.json
```

The updated action contract uses actor-relative robber victim identities; old
pilot checkpoints remain historical validation artifacts and must not be used
as new training initialization. Checkpoints now enforce matching source hashes.
The controller retries one failed transfer at most and acknowledges incremental
snapshots only after hash verification and extraction. An independent local
watchdog releases the dedicated session at the original deadline if the main
controller dies; host failure remains outside its guarantees.

A full-buffer DQN pilot supplements the short pilots. Common transition targets
are computed from the slowest measured rate with 30-minute nominal training
allowance, capped at 500,000, leaving margin beneath each 45-minute hard limit.
Final metrics use the minimum completed board count across all six policies;
extra raw evaluation rows are retained separately. GPU utilization is sampled
every ten seconds. The worker probe measures one local engine and two synchronous
subprocess engines, including observation transport to the central policy.

For an interrupted runtime with retrieved sweep checkpoints, a fresh controller
can use `--restore /persistent/prior/retrieved --previous /persistent/prior/controller.json`.
It uploads the retrieved state, retains the frozen target, skips already selected
runs, resumes the latest atomic checkpoint, and preserves each recorded run's
original 45-minute deadline. Runs whose deadline expired are incomplete. This
replacement-runtime path was added locally after session b launched; its source
is distinct from the archived source actually running in b. It must pass a
recovery dry run before use. Standalone checkpoint recovery is separately tested.

The executed run's literal optimizer defaults, initialization, normalization and
sampling details are recorded in `implementation-details.json`. The external PPO
checkpoint upload/next-transition/update check is reproducible with
`verify_roundtrip_colab.py` (paths identify the actual tested checkpoint).
Direct CPU-percent/RSS sampling was added during the second PPO-128 run via
`cpu_monitor.py`; earlier CPU telemetry contains load averages and cumulative
process CPU time. GPU sampling spans preflight, training and evaluation.

An auxiliary 82 MiB DDQN checkpoint upload failed once with a TLS EOF. Training
continued unaffected. For large recovery uploads, the local replacement-runtime
controller now gzip-compresses the archive, sends 8 MiB parts through Colab CLI,
reassembles it remotely, and verifies its SHA256 before extraction. It also
requires an explicit success marker from remote Python because CLI exit status
alone is insufficient to distinguish a Python exception. Error URLs are redacted
before logging. These controller refinements do not change the running trainer.

A live controller can be reattached with `--resume` after a controller-only
interruption; the remote trainer continues and no training budget is reset.
The session-b controller was reattached this way after an unexpected SIGTERM.
For overnight use, run the controller in a persistent detached process with its
stdout/stderr redirected outside the synced workspace; its independent deadline
watchdog remains required. `status.py ARTIFACT_DIRECTORY` reads local status.

`train.py --seconds` is a per-invocation limit. Use the experiment controller for
cross-retry deadline enforcement. The recovery helper additionally accounts for
previous training and validation time and retains an existing run deadline.

After runtime termination, verify and report retrieved artifacts:

```sh
python rl_research/overnight/verify_artifacts.py --artifacts ARTIFACT_DIRECTORY --output recovery-checks.json
python rl_research/overnight/package_selected.py --artifacts ARTIFACT_DIRECTORY
uv pip install --python /tmp/tinysettler-rl-venv/bin/python -r rl_research/overnight/reporting-requirements-lock.txt
MPLCONFIGDIR=/tmp/tinysettler-mpl /tmp/tinysettler-rl-venv/bin/python rl_research/overnight/report.py --artifacts ARTIFACT_DIRECTORY --output REPORT_DIRECTORY
```

The report generator produces raw tables, manifests, aggregate and paired
board-cluster intervals, utilization summaries, and standalone SVG/PNG learning
curves. `metrics-report.md` is generated; `findings.md` contains the interpretation.
Boundary win rates use conservative exact board-cluster bounds because a plain
percentile bootstrap degenerates when every observed outcome is identical.
Intervals condition on the two trained seeds; they do not measure population
uncertainty over training seeds.

# PPO reliability experiment, 2026-11-11

Status: **completed**. [Findings, tables, curves and artifact hashes](../results/ppo-reliability-20261111/README.md). Both allocated sessions are terminated; the commands below document reproduction and recovery, not pending work.

The user's latest authorization sets a **five-hour (18,000 second)** allocation cap,
superseding the four-hour draft. The final 1,200 seconds are reserved for artifact
retrieval and shutdown. The absolute deadline persists across controller resume
and at most one runtime replacement. The optional stronger-heuristic challenge is
omitted to prioritize the primary experiment.

The exact archived trainer, environment, config and dependency lock are extracted
from the executed overnight source archive. They are not edited. Diagnostic
subclasses add counters only: local tests compare trajectories and optimizer
updates bitwise against the original implementation. The archived config's old
operational fields are overridden explicitly by the new experiment manifest;
learning settings and actual environment behavior remain unchanged.

Three fresh seeds (155921, 196613, 262147) train PPO-128 and PPO-256 from scratch,
rotating width order by seed. Each run targets 434,176 transitions using one worker.
Checkpoints are 106,496, 217,088, 323,584 and 434,176, plus initial and five-minute
recovery saves. A run has at most 2,100 training seconds; evaluation time is
accounted separately. Incomplete runs are recorded and are not silently retrained.

Fresh validation starts at 2,100,000,000 (20 boards); final testing starts at
3,100,000,000 (50 boards), each with two seats and three opponents. Before training,
the pilot may reduce these counts uniformly if conservative projection cannot fit.
The frozen manifest records the decision. All 40 nonzero checkpoints receive the
same validation coverage. Selection uses win fraction, then fewer truncations,
then earlier step. A hash-verified external selection commit precedes final tests.
The common final checkpoint is the primary endpoint. Distinct selected checkpoints
are secondary, with a separately reported uniform complete-board prefix.

Diagnostics record both actors' action-type counts, end-turn frequency, maritime
trades, scores and limits. Repetition means identical public buildings, roads,
robber position and own resource/development hand at a learner decision. It
intentionally ignores clock and phase: it is a repetition indicator, not proof
of a decision loop. Only state hashes are exported. Opponent hidden cards, exact
bank counts, deck order and RNG are excluded from diagnostic views. Training
checkpoints retain the full simulator state needed for exact recovery.

## Prepare and launch

Run from the repository root, with the existing isolated RL environment. Model
artifacts stay on the persistent VPS volume outside the synced workspace.

```sh
PYTHONPATH=rl_research/overnight:rl_research/reliability \
  /tmp/tinysettler-rl-venv/bin/python -m pytest -q \
  rl_research/overnight/test_contract.py rl_research/reliability/test_reliability.py
python rl_research/reliability/prepare.py \
  --artifacts /root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a
python rl_research/reliability/reliability_controller.py \
  --session tinysettler-ppo-reliability-20261111-a \
  --artifacts /root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a
```

For unattended operation launch the controller with `subprocess.Popen`,
`start_new_session=True`, stdin `/dev/null`, and stdout/stderr in the artifact
directory. Its independent watchdog terminates only the dedicated session before
the original deadline. Do not leave the controller attached to an expiring shell.

Reattach to a live session with the same command plus `--resume`. If the old
runtime was terminated and independently verified absent, use `--resume --replace`
(at most once). Replacement restores the externally verified snapshot and retains
the original absolute deadline. Never create a new artifact directory to evade
the allocation cap. The controller refuses a new allocation without the final
20-minute reserve. Snapshots are copied approximately every 40 seconds; transfers
have bounded retries and remote Python completion markers detect CLI false success.

`controller.json`, `retrieved/status.json`, and per-run `progress.json` give status.
The controller stops the runtime on success or unrecoverable failure and records
a fresh server session listing. Do not call completion until retrieval hashes and
runtime termination are verified.

## Reproduce evaluation and reports

On a compatible CUDA runtime, use the extracted `source/` directory so historical
provenance checks remain enabled:

```sh
cd /root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a/source
python reliability_evaluate.py CHECKPOINT.pt --seed-start 3100000000 \
  --boards 50 --deadline ABSOLUTE_UNIX_DEADLINE --output evaluation.json
```

Generate local reports after retrieval:

```sh
PYTHONPATH=rl_research/overnight:rl_research/reliability \
  /tmp/tinysettler-rl-venv/bin/python rl_research/reliability/reliability_report.py \
  --artifacts /root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a \
  --output rl_research/results/ppo-reliability-20261111
```

The dependency lock is the unmodified
[overnight lock](../overnight/requirements-lock.txt); plotting uses the existing
[reporting lock](../overnight/reporting-requirements-lock.txt). Every executed source
hash, simulator revision, effective split and seed is recorded in the manifest.
Intervals cluster both seats and opponents by shared board; a separate crossed
bootstrap also resamples paired training seeds. Five seeds are preliminary.

These are reliability measurements against the specified fixed opponents.
The heuristic's maritime-trade priority is weaker than ending the turn. Results
do not establish general Catan strength, negotiation quality or browser latency.
Inference is measured inside Colab; PPO probabilities are not Q-values or calibrated
win probabilities.

CLI syntax was checked against installed 0.7.5 help and the
[official CLI documentation](https://github.com/googlecolab/google-colab-cli).
Runtime lifetime and availability remain subject to
[Colab limits](https://research.google.com/colaboratory/faq.html).

The executed run required one preflight replacement and a timing correction before training. Its [operational record](../results/ppo-reliability-20261111/operational-events.json) and [calibration correction](../results/ppo-reliability-20261111/calibration-correction.json) are retained. Current orchestration prevents cached evaluation timing from being reused as a fresh measurement. The executed source archive remains unchanged.

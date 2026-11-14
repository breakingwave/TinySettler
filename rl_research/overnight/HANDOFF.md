# Paused at user request — 2026-11-11 03:51 UTC

User requested a pause to restart the agent with updated permissions. Do not
resume compute until the user resumes work.

## Current state

Implementation is in this directory. Browser demo and historical results were
not changed. Six local CPU contract tests passed; the same tests passed on Colab.
Standalone train/resume and local paired evaluation smoke tests passed.
A real Tesla T4 and CUDA optimizer update were verified remotely. All three
4096-transition pilots completed, evaluated six games each, and reloaded their
checkpoints remotely. No ablation training run has started; the common transition
target has NOT been frozen. No final-test seeds have been used.

Session `tinysettler-ablation-20261111-a` was stopped. The controller exited with
KeyboardInterrupt after successfully releasing it. A simultaneous second stop
returned Not Found because the first stop had already succeeded. The final
server listing reported **no active sessions**.

Persistent artifacts outside synced workspace:
`/root/tinysettler-artifacts/tinysettler-ablation-20261111-a/`

- `retrieved/`: preflight, tests, pilot statistics, three pilot checkpoints,
  telemetry, and status `awaiting_external_recovery`.
- `pause.tar`: retrieved pause snapshot, local SHA256
  `b4dddfa0e7ced5747818e8296566904f69d984abb7f5b34a5e49a95b5f61a251`.
- `controller.json`: allocation/deadline timestamps and uploaded source hashes.
- `pause-sessions-after.txt`, `sessions-after.txt`: no active sessions.
- Earlier incremental snapshots were remote/local SHA256 verified. The pause
  archive was downloaded and hashed locally; remote hash comparison and external
  checkpoint reload remain to be completed after resume.

Controller start: 1794368666.0740395 UTC epoch seconds.
T4 allocation ready: 1794368669.6337593. Cleanup finished: 1794369088.4465027.
Conservatively charge 423 seconds (7.05 minutes) against the eight-hour total.
Remaining allocation budget is at most 28,377 seconds (7h 52m 57s).
Do not reset to a fresh eight-hour budget on restart. The original controller
wall deadline was 1794397466.0740397; preserve it unless the resumed user explicitly
permits excluding paused time while still enforcing cumulative allocation <=8h.

## Pilot measurements (not learning comparisons)

| Model | learner decisions/s | engine actions/s | update seconds | parameters |
| --- | ---: | ---: | ---: | ---: |
| PPO 128 | 371.19 | 736.77 | 0.512 | 321869 |
| PPO 256 | 387.27 | 790.34 | 0.306 | 676173 |
| Double DQN 256 | 283.68 | 580.94 | 4.691 | 676173 |

Runtime exposes two CPUs. GPU snapshots were 4–7% utilization and 219–223 MiB
used. Full pilot raw data and telemetry are in persistent artifacts. These short
pilots include DQN warmup; measure steady-state replay/update throughput before
freezing the common target. The optional graph path is omitted, leaving six runs.

## Resume work

1. Read the user's original full specification and this handoff. Inspect and
   validate the implementation rather than assuming the initial code is complete.
2. Verify retrieved pilot checkpoints locally (CUDA tensors can load with
   `map_location='cpu'`; the serialized trainer still declares device cuda).
3. Improve controller lifecycle handling to reuse retrieved artifacts in a NEW
   runtime with cumulative budget accounting. Current `--resume` only reattaches
   to a live runtime, which no longer exists. Do not run it against the stopped
   session. No automatic reprovisioning is implemented.
4. Measure steady-state DQN and worker capacity, freeze target, then run six
   sequential training runs. Honor original settings, opponents and seeds.
5. Finish held-out validation selection, final evaluation, artifact hash/recovery
   checks, summary/curves and cleanup. `report.py` exists but has not been run.

Potential improvements to review before sweep: current checkpoint recovery
serializes the entire trusted Trainer object; resume CLI does not validate target
arguments against stored target. Action mapping uses absolute engine colors for
robber victim entries while observations are actor-relative; canonicalizing victim
semantics would improve consistency across seats and must be tested before freezing.
Worker count currently defaults to one based on pilot throughput and CPU count;
comparative worker measurements are outstanding. Final evaluation must enforce a
uniform completed board count across every run if any reaches its deadline.
Controller snapshot cutoff advances before successful transfer, so failed transfer
retries need careful handling. Host crash can interrupt the local cleanup watchdog.

Local environment: `/tmp/tinysettler-rl-venv`; all pinned dependencies are in
`requirements-lock.txt`. Colab CLI 0.7.5, existing OAuth authentication.
`monitor.py` was uploaded separately after the initial source bundle; `report.py`
was created locally afterward. Neither changes training. Do not claim the six-run
experiment or findings report is complete.

# Completed reliability experiment

No compute or follow-up action remains pending.

Results: `../results/ppo-reliability-20261111/findings.md`.
Artifact root: `/root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a`.
The dedicated session `tinysettler-ppo-reliability-20261111-a-r1` was released;
an independent server listing confirmed no active sessions. Both local watchdogs
and the detached controller are absent.

All six fresh PPO runs completed 434,176 transitions. All 40 checkpoints received
120 validation games; all ten common-final policies received 300 fresh test games.
Five distinct selected checkpoints received another 300 games each. Full board
coverage was retained. Total study evaluation games: 9,300; illegal actions: zero.

Conservative allocation accounting: 13,054.157 seconds, below the authorized
18,000-second cap. One failed preflight runtime was replaced under the same
absolute deadline. A cached-calibration timing bug was corrected before training;
the explicit secondary plan used real validation durations and no test outcomes.
Operational scripts and events are preserved beside the artifacts.

All 31 retrieved checkpoints passed hashes/provenance/finiteness checks. Six final
checkpoints passed CPU recovery updates; an external CUDA pilot round trip was
bitwise identical. The 15 common-final/selected policies are consolidated in
`common-final-and-selected.tar.gz`, with a verified package manifest.

The four archived core hashes remain unchanged. Current orchestration fixes are
separate from the archived executed source. No historical results or browser-demo
files were replaced.

Recommendation: retain PPO-128 provisionally; width differences remain inconclusive
across five seeds and the larger model did not reduce common-final stalls. Address
training and selection stability before broadening architectures or opponents.

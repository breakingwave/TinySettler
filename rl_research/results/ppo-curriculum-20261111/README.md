# PPO-128 curriculum continuation — completed

All six runs completed 176,128 additional transitions. Evaluation covered 1,200
fresh full games and 132 held-out exercise games. All checkpoints were retrieved
and verified; the dedicated runtimes are terminated and no Colab sessions remain.

**Curriculum did not demonstrate a fresh-game advantage:** 51.33% wins versus
52.50% for standard continuation; paired-policy/board 95% interval for the
−1.17-point difference: [−7.50, +5.67] percentage points. Three pairs provide
exploratory evidence. Both arms remained weak against adapted base/contender.

[Findings](findings.md) · [Frozen protocol](protocol.md) · [Manifest](frozen.json) ·
[Raw full games](full-game-outcomes.csv) · [Raw exercises](exercise-outcomes.csv) ·
[Per-pair table](per-pair.csv) · [Opponent table](per-opponent.csv) ·
[Training curves](training-return-curves.png) · [Timeouts](timeout-diagnostics.json) ·
[Checkpoint verification](completion-verification.json) ·
[Commands and recovery](../../curriculum/README.md)

Persistent artifacts:
`/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a/`.
See [checkpoint package](checkpoint-package.json) for the six-policy/source/corpus
archive and SHA256, and [artifact manifest](artifact-manifest.json) for individual
files. Historical artifacts and the browser demo remain unchanged.

The user-requested pause and one technical recovery are recorded. The original
four-hour deadline was retained: allocated interval upper bounds total 2 h 12 m
50 s; the conservative original clock including downtime totals 2 h 26 m 15 s.
[Resource accounting](resource-summary.json) records all three sequential T4
allocations and verified termination.

# Completed PPO reliability experiment

[Findings and recommendation](findings.md) · [Reproduction commands](../../reliability/README.md)

Six new runs completed at 434,176 transitions each. All five seeds per width received full 50-board common-final testing; five distinct validation-selected policies received an additional 50 boards each. Both seats and all three opponents were retained. No incomplete runs or illegal actions occurred.

- [Frozen experiment manifest](frozen.json), [original input manifest](input-manifest.json), [pre-test selection commit](selections.json), [secondary coverage plan](secondary-plan.json).
- [Common-final per-seed table](common-step-per-seed.csv), [selected-checkpoint table](selected-checkpoint-per-seed.csv), [opponent results](opponent-comparison.json).
- [Raw primary games](final-games.csv), [selected-policy games](selected-games.csv), [per-seed validation curves](learning-curves.csv), and complete remote JSON records in `raw/`.
- [Learning curves PNG](learning-curves.png) / [SVG](learning-curves.svg), [wall-clock curves PNG](wall-clock-curves.png) / [SVG](wall-clock-curves.svg), [seed comparison PNG](common-final-per-seed.png) / [SVG](common-final-per-seed.svg).
- [Primary intervals](paired-comparisons.json), [selected-policy intervals](selected-paired-comparisons.json), [per-opponent intervals](opponent-paired-comparisons.json), [observed seed variability](seed-variability.json).
- [Timeout diagnostics](timeout-diagnostics.json), [timeout summary](timeout-summary.json), [time accounting](time-accounting.json), [CPU/GPU utilization](utilization.json).
- [Artifact hashes](artifact-manifest.json), [checkpoint package manifest](checkpoint-package-manifest.json), [recovery checks](recovery-checks.json), [completion verification](completion-verification.json), [operational events](operational-events.json).

The selected-policy CSV contains 3,000 rows: it reuses primary rows where the selected checkpoint is the common final checkpoint. Only 1,500 additional test games were needed. The `split` column records the evaluation stage of each reused or newly evaluated row. No policy was selected from final-test outcomes.

Persistent artifact directory on the VPS, outside the synced repository:

`/root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a/`

Consolidated resumable checkpoint package (15 policies; 83,066,070 bytes):

`/root/tinysettler-artifacts/tinysettler-ppo-reliability-20261111-a/common-final-and-selected.tar.gz`

SHA-256: `0a09498c06cf38471876b94844b5e9ed85d1299a6959a9aa2b062aa2929f1b5e`.

All intermediate and initial new checkpoints remain under `retrieved/`; historical checkpoints remain in their original artifact directory and the verified historical archive. The executed source is preserved in `source.tar.gz`; operational repairs are stored separately and documented. The immutable [learning dependency lock](../../overnight/requirements-lock.txt) and [reporting lock](../../overnight/reporting-requirements-lock.txt) remain in the repository.

The runtime was released after verified retrieval. Conservative allocation accounting: 13,054 seconds, below the authorized 18,000-second cap, including the failed preflight attempt and replacement. The final 20-minute reserve was preserved.

In learning-curve plots, dashed lines denote historical training seeds and solid lines denote fresh seeds. The local controller and watchdog cleanup is recorded in [local-cleanup.json](local-cleanup.json).

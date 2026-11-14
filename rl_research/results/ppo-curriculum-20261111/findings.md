# PPO-128 curriculum continuation: completed 2026-11-11

Curriculum did **not demonstrate better fresh full-game performance**. Standard
training won 315/600 games (52.50%); curriculum won 308/600 (51.33%). The paired
curriculum-minus-standard difference was −1.17 percentage points, with a 95%
interval of **[−7.50, +5.67] points** when resampling both starting-policy pairs
and board clusters. The board-only interval, conditional on these fitted policies,
was [−4.00, +1.83]. Three pairs support exploratory conclusions, not equivalence.

Both arms remained weak against adapted ValueFunction-base and held-out contender.
Curriculum showed a small, uncertain improvement on the productive-trade exercise,
without consistent benefits across the other exercises or full games. Retain
ordinary full-game continuation as the reference; this result does not justify
adopting the tested curriculum for the demo.

## Common-budget comparison

Each pair started from the same historical PPO-128 weights **and Adam state** at
434,176 transitions. All six runs completed **176,128 additional transitions**,
1,056,768 total. No checkpoint selection was performed. Both arms faced the same
50% adapted base / 25% WeightedRandom / 25% legacy-heuristic mixture. Only reset
distribution differed. The full [frozen manifest](frozen.json) and
[prospective protocol](protocol.md) record unchanged rules, observations, masks,
rewards, PPO settings, opponent weights, seeds and budgets.

Each policy played 20 untouched boards, both seats, against five opponents.
Truncations remain in the denominator and are reported separately.

| Starting seed | Standard W / L / trunc. | Curriculum W / L / trunc. | Win difference | Training minutes, standard / curriculum |
| --- | ---: | ---: | ---: | ---: |
| 104729 | 111 / 89 / 0 | 97 / 102 / 1 | −7.0 pp | 16.94 / 15.25 |
| 130363 | 103 / 97 / 0 | 110 / 89 / 1 | +3.5 pp | 13.68 / 13.68 |
| 155921 | 101 / 98 / 1 | 101 / 98 / 1 | 0.0 pp | 14.13 / 14.04 |
| Pooled | **315 / 284 / 1** | **308 / 289 / 3** | **−1.17 pp** | |

| Opponent | Standard wins / 120 | Curriculum wins / 120 | Standard / curriculum truncations |
| --- | ---: | ---: | ---: |
| Adapted ValueFunction-base | 3 (2.50%) | 3 (2.50%) | 0 / 0 |
| Adapted contender | 6 (5.00%) | 4 (3.33%) | 0 / 0 |
| Random | 110 (91.67%) | 110 (91.67%) | 0 / 1 |
| WeightedRandom | 115 (95.83%) | 106 (88.33%) | 0 / 0 |
| Legacy heuristic | 81 (67.50%) | 85 (70.83%) | 1 / 2 |

Base and contender are configurations of the **same heuristic family**, not
independent algorithms. Their combined win difference was −0.83 pp, with paired
policy/board interval [−6.25, +4.58]. Mean points against that family increased
from 3.61 to 4.09, but its paired-policy interval also crossed zero. Low scores
suggest opening/economy development as a subsequent diagnostic target; this
experiment does not establish its causal role. Historical weak-opponent win
rates are not a matched before/after control for this stronger-opponent test.

There were 599 versus 597 terminal full games, and **zero illegal actions**.
All four evaluation truncations hit the original turn cap, not the action cap.
They involved two-settlement learners with substantial trading; the curriculum
timeouts ended at 8, 9 and 6 VP, and the standard timeout at 4 VP. Action counts,
end-turn frequency, public/own-hand repeat hashes and final scores are in
[timeout diagnostics](timeout-diagnostics.json). Repeated hashes alone do not
prove a decision loop. Training had one standard and two curriculum truncations,
recorded separately in [training diagnostics](training-timeout-diagnostics.json).

## Held-out exercises

Eleven cases from disjoint test source games received two future-randomness
continuations per policy: 132 games. Largest-army cases cover two source games;
the other tasks cover three each. The small clustered sample limits inference.

| Task | Standard successes | Curriculum successes | Decisions to success, standard / curriculum | Eventual wins, standard / curriculum |
| --- | ---: | ---: | ---: | ---: |
| Largest army | 12/12 | 11/12 | 2.0 / 3.7 | 11/12 / 9/12 |
| Expansion | 17/18 | 17/18 | 9.1 / 10.8 | 6/18 / 7/18 |
| Productive trade proxy | 6/18 | 9/18 | 18.8 / 43.4 | 0/18 / 2/18 |
| Immediate win | 6/18 | 6/18 | 1.0 / 1.0 | 12/18 / 16/18 |

Timing is conditional on success. The productive-trade proxy requires a trade
that creates a new legal purchase followed by a purchase before ending the turn;
it does **not** require buying that exact newly enabled item. Its +16.67 pp success
difference had a paired-policy/source-game interval of [−61.11, +83.33]. No robust
targeted-exercise advantage was established. More eventual wins on immediate-win
cases did not improve first-action conversion. Full
[exercise results](exercise-summary.csv) and [clustered intervals](exercise-comparisons.json)
preserve these distinctions.

## Historical strategic failures

The two recorded 25-development-card / zero-knight / 8-VP games are retained in
[historical diagnostics](historical-diagnostics.json). The seed-196613 game on
board 3100000019 reproduced both actors' action counts, counters, outcome and
recorded repeated-state hashes. The seed-155921 game on board 3100000010 could
only be reconstructed approximately: historical Python hash seed and action
trace were not archived, and CPU/CUDA arithmetic can differ. Its reconstruction
ended at 7 VP; it is explicitly **not** represented as the original 8-VP state.

In the exact case, knight play was legal at 1,097 decisions. At turn 401 the
policy assigned 24.14% to a playable knight but 55.51% to rolling, so greedy
selection rolled. An earlier road-expansion opportunity assigned 25.41% combined
probability to legal roads and 19.00% to a knight, while ending the turn had
36.31%. There was no legal settlement at that earlier state. This supports a
policy/greedy-selection failure rather than claiming a useful action was masked
out. Permitted observations, installed action indices, masks, probabilities,
own development-card availability/age and subsequent actions are archived.

Seven selected positions received eight repeated continuations per selection
mode, with independent policy RNG and paired future randomness. Greedy won 9/56
and truncated 43/56; sampling won 55/56 and truncated 1/56. In the exact case's
three positions, greedy truncated all 24 continuations while sampling won all
24. Task success and action traces are retained, not inferred from timeout
reduction. These deliberately selected failures, including approximate
reconstructions, are **diagnostic-only**; they do not show sampling is stronger
in general. A separate untouched full-game greedy/sampling comparison is needed
before changing demo action selection. PPO probabilities are neither Q-values
nor calibrated win probabilities.

## Corpus, opponent audit and efficiency

The legal-play corpus contains 421 snapshots: 307 train, 57 validation and 57 test,
split by source game before snapshot selection. Training has 162 midgame and
145 endgame positions; 36 are behind the opponent's public VP. Complete state,
development-card age, awards, history and original counters/horizon are retained.
Repeated resets change future randomness, not established physical facts.
Historical diagnostics never enter the corpus. See [corpus provenance](corpus-manifest.json).

The explicitly labeled opponent adapter retains upstream base/contender weights,
float64 scores and K=2 common sampled worlds. It replaces hidden opponent cards,
hidden VP and unseen deck order, uses conservative public resource bounds and
observed legality, and isolates checkpointed planning RNG from live RNG. This is
an approximate information-state sampler, not an exact Bayesian belief. Local
and T4 contract checks passed; configured opponent probabilities were tested
through actual environment creation. The small paired preflight won 8/8 against
WeightedRandom and 8/8 against the legacy heuristic. Recorded training and primary
evaluation sampling fallbacks were **zero**, across 425,757 planning calls;
single-legal-action calls bypass world sampling. No search agents were used.

Curriculum's 528,384 transitions comprised 372,213 fresh, 106,727 midgame and
49,444 endgame decisions. The 50/25/25 mixture applies to **episode resets**, not
transition shares. Standard completed 2,919 episodes (2,918 terminal); curriculum
completed 4,073 (4,071 terminal). Terminal rewards occurred on 0.552% versus
0.770% of learner transitions: the curriculum increased reward frequency without
demonstrating full-game transfer. Per-seed/category counts are in [runs.json](runs.json).

Training achieved 173–215 learner decisions/s and 359–447 engine actions/s.
Recorded trainer CPU use averaged 95.8% of one core; GPU utilization averaged
6.4%, with 221 MiB peak reported GPU memory and 1.71 GiB peak trainer RSS. These
telemetry summaries include training and evaluation across allocations. The
single-worker path was CPU dominated. Primary evaluation took 714.6 seconds;
exercises took 54.9 seconds. Mean policy inference was 0.430 ms standard and
0.433 ms curriculum, **measured inside Colab, not browser latency**.

## Recovery, artifacts and recommendation

All six final checkpoints and 33 total checkpoint files passed external SHA256,
source/configuration and finite-state checks. Each final policy passed a CPU
collect/update recovery check. Initial paired weights and Adam states matched.
CUDA external-copy recovery had already passed before the sweep. There are no
incomplete final runs and all 1,332 evaluation games are present.

One allocation was stopped at the user's request. The first resume then suffered
a Colab kernel connection read timeout and was shut down automatically. One
additional bounded technical recovery resumed preserved work. Only controller
retry/backoff/copy cadence changed; the executed training bundle stayed immutable.
Rolled-back work was not counted toward the retained transition target. The
original absolute deadline was never extended. Sequential allocation intervals
total at most **7,970 seconds (2 h 12 m 50 s)**; the more conservative original
clock, including pause/downtime, was **8,775 seconds (2 h 26 m 15 s)**, below four
hours. All dedicated runtimes were terminated, and an independent final CLI
listing found no active sessions. [Verification](completion-verification.json),
[resource ledger](resource-summary.json) and [recovery amendment](technical-recovery.json)
record the evidence.

Persistent artifact directory:
`/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a/`.
The self-contained `final-policies-and-corpus.tar.gz` contains all six final
policies, corpus, frozen manifests and exact executed source; SHA256:
`2bcfacc77a101807544aecf9bdc625f6f1d51b34dd86b6d2276ced55b28e9802`.
Individual hashes and locations are in [artifact manifest](artifact-manifest.json)
and [checkpoint package](checkpoint-package.json). Dependencies remain pinned in
the archived `requirements-lock.txt`; installed packages are recorded in
[installed freeze](installed-freeze.txt). Historical artifacts and browser demo
were preserved.

Use [launch/resume/evaluation commands](../../curriculum/README.md) and the
archived implementation for checkpoint loading. [Raw full games](full-game-outcomes.csv),
[raw exercises](exercise-outcomes.csv), [per-pair comparisons](per-pair.csv),
[training-return curves](training-return-curves.png), [loss curves](learning-curves.png)
and [paired result plot](paired-full-game-results.png) accompany this report.
Training-return curves are descriptive, not held-out learning curves.

The tested curriculum should remain an experimental tool rather than the new
default. Prioritize diagnosing early full-game losses to the stronger opponent
and testing greedy versus sampled action selection on untouched full games.
The current trade-exercise signal is too small and uncertain to justify a
strength claim. This experiment measures reliability and transfer against the
specified opponents; it measures neither negotiation quality nor browser latency.

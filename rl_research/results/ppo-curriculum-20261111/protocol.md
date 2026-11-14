# Frozen curriculum comparison protocol

The only arm difference is starting-state distribution. Standard resets to fresh
games; curriculum samples 50% fresh, 25% midgame (4–6 own VP), 25% endgame (7–9).
Both face 50% adapted ValueFunction-base, 25% WeightedRandom, 25% legacy heuristic.
No negotiation, reward shaping, architecture change, checkpoint selection or
training-opponent change is permitted during the comparison.

The three paired starting policies are PPO-128 seeds 104729, 130363 and 155921 at
434,176 historical transitions. Each pair starts with identical model and Adam
state. Fresh RNG namespaces and episodes replace historical pending rollouts.
The calibrated target is **176,128 additional transitions per policy**. Run order
rotates: standard/curriculum, curriculum/standard, standard/curriculum.

Primary testing uses **20 fresh boards**, each in both seats against adapted
base, held-out contender, Random, WeightedRandom and the legacy heuristic: 200
games per policy, 1,200 total. Base and contender are configurations of one
heuristic family. Report them separately and never count them as independent
algorithms. No model selection uses these test outcomes.

The frozen legal-play corpus contains 421 snapshots: 307 training, 57 validation,
57 test, partitioned by source game before snapshot selection. Training snapshots
include 162 midgame and 145 endgame positions; 36 have fewer own VP than the
opponent's public VP. This is not a uniform sample of all possible Catan states.
Each source game contributes at most one ordinary and first eligible task state
per category, with some states satisfying multiple tasks. Source games were
assigned to splits before any snapshot was chosen. Future dice/steal randomness
is reseeded; physical state, deck order, history and original caps are retained.

Held-out exercise cases are selected by source-game order, at most three distinct
source games per task; two independent continuations per case. There are 11
cases, 22 games per policy. Success definitions, fixed before evaluation:

- Largest army: acquire the largest-army award that was absent at reset.
- Expansion: build a settlement after reset.
- Productive trade: take a maritime trade that makes a new purchase legal, then
  execute a purchase before ending that turn. This is a conversion proxy; it
  does not require the purchased item to be that exact newly enabled action.
- Immediate win: win on the first learner action.

Record decisions/turns/actions until first success and eventual win/loss/truncation.
Cases can share a source game across tasks; source-game clusters retain this
relationship. Success timing is conditional on success and can be misleading if
reported without success rates. Full-game transfer remains the primary endpoint.

Uncertainty uses 10,000 percentile bootstrap draws. Full-game board clusters
retain swapped seats and opponent configurations. A separate crossed bootstrap
resamples the three paired starting policies and shared board clusters. Exercise
intervals cluster source games and resample paired policies separately. Three
starting pairs and few exercise games support exploratory conclusions only.

Diagnostic historical games are excluded from both training and evaluation
corpora. One requested 25-card/zero-knight/8-VP case reproduces both actors' action
counts, counters and all recorded repeated-state hashes. The other recorded case
cannot be exactly reconstructed from archived data: Python hash seed and action
trace were not retained, and CPU/CUDA arithmetic may differ. Its original record
and explicitly approximate same-board continuation are kept, without presenting
the latter as the original state. Selected failures cannot establish the overall
strength of sampled versus greedy play.

Runtime cap: 14,400 seconds across all attempts, including setup, pilots,
evaluation, transfer and shutdown; the final 1,200 seconds are reserved. Both
pilots used actual stronger opponents. Full evaluation coverage fit the measured
budget. Model/action latency is measured inside Colab, not in the browser.

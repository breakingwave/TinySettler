# PPO and curriculum evaluation strategy

Use fresh full games to decide whether the midgame/endgame curriculum improves PPO-128. Use targeted continuation tests to explain the result. Stronger opponent performance and reliable game completion matter more than a high average against weak bots.

Originally prepared before the curriculum results; updated after completion on 2026-11-11. The original frozen protocol and results remain unchanged. The curriculum did not demonstrate a full-game advantage. See the [complete synthesis](rl-summary-20261111.md) and [next short evaluation plan](short-strong-opponent-evaluation.md). The screening criteria below record proposed decision rules; they do not retroactively redefine the completed experiment's endpoints.

## Opponents and performance targets

These targets are proposed project milestones, not predictions or published Catan skill ratings. Current rates below are standard PPO-128 continuation, pooled over three starting policies with \(120\) games per opponent. They are not the performance of a single selected deployment checkpoint. The earlier weak-opponent figures used five policies and different boards.

| Opponent | Evaluation role | Latest standard continuation wins | Demo milestone | Stronger milestone |
| --- | --- | ---: | ---: | ---: |
| Catanatron RandomPlayer | Basic competence and regression check | \(91.67\%\) | \(\geq95\%\) | \(\geq98\%\) |
| Catanatron WeightedRandomPlayer | Regression against basic build preferences | \(95.83\%\) | \(\geq90\%\) | \(\geq95\%\) |
| Existing fixed-priority heuristic | Historical continuity; it never voluntarily bank-trades | \(67.50\%\) | \(\geq80\%\) | \(\geq90\%\) |
| Adapted ValueFunctionPlayer, base weights | Main heuristic challenge; included in training | \(2.50\%\) | \(\geq50\%\) | \(\geq60\%\) |
| Adapted ValueFunctionPlayer, contender weights | Held-out heuristic variant | \(5.00\%\) | \(\geq50\%\) | \(\geq60\%\) |
| Audited information-safe AlphaBetaPlayer, depth \(2\) | Advanced held-out search challenge, after validation | Unmeasured | \(\geq50\%\) | \(\geq60\%\) |

Base and contender share the same evaluator family. Neither the name “contender” nor its different weights establishes that it is stronger. It counts as held out only if absent from training opponents, curriculum generation and tuning. Benchmark both against WeightedRandom and the legacy heuristic on separate calibration boards, and against each other if the time budget permits. Record results and time per decision; do not remove an inconvenient opponent after seeing learner results.

The measured win rates against the two ValueFunction variants are far below parity. The \(50\%\) and \(60\%\) figures are longer-term goals, not reasonable forecasts for one more short run. Winning less than half can still be promising if it is a substantial, repeatable improvement over a matched baseline. Beating both at the stronger milestone establishes strength against these bots under this ruleset; it does not establish expert-human strength.

Retain the three exact pre-continuation PPO-128 parent checkpoints as reference policies. Evaluate them on the same boards, especially against the two new opponents, to measure progress from the starting policies. A later paired-seat head-to-head against these frozen policies adds an opponent family independent of the heuristic. Do not select the reference using final-test performance. Self-play at approximately \(50\%\) is not a strength test.

Do not use raw AlphaBetaPlayer, SameTurnAlphaBetaPlayer or MCTS as a rated benchmark yet. The pinned engine audit found invalid chance enumeration and shared simulation RNG, and the information boundary requires an adapter. Repairing and validating search is a separate experiment.

AlphaBetaPlayer is the next advanced benchmark: it applies a hand-coded value function at the leaves of a search that averages chance outcomes and maximizes or minimizes according to the actor. In the installed implementation, depth \(2\) means two engine actions, not two complete turns. It can therefore test combinations such as a bank trade followed by a build, while not necessarily reaching an opponent reply. Keep its leaf weights matched to the base ValueFunction opponent to measure the contribution of search.

Before rating it, correct the dice/development-card outcome API, isolate all simulation RNG, and replace inaccessible hidden state with information-consistent sampled worlds. Validate chance probabilities and search values against small exhaustive reference trees, including chance nodes; deterministic alpha-beta cutoff bounds cannot simply be treated as exact values inside a weighted expectation. Record this as an adapted approximate-belief search opponent. Use a frozen depth and explicit node/decision-time budget; report achieved depth, cutoff/fallback rates and latency. The stock implementation has a nominal \(20\)-second search deadline per decision and expands children before recursive deadline checks, so that constant alone is not a strict cost bound.

Evaluate the frozen PPO parents against this search opponent first. Parity is a useful initial goal, and \(60\%\) wins a stronger milestone, but neither is a forecast. Keep AlphaBeta excluded from training for this evaluation. Add it as a separately budgeted supplementary tier after correctness and timing checks; it is not included in the current five-opponent game count or the two-ValueFunction comparison. Do not silently extend the running allocation to accommodate it. [Pinned search implementation](../.venv/lib/python3.14/site-packages/catanatron/players/minimax.py).

Sources: [latest opponent results](results/ppo-curriculum-20261111/per-opponent.csv), [latest findings](results/ppo-curriculum-20261111/findings.md), [earlier opponent results](results/ppo-reliability-20261111/opponent-comparison.json), [engine audit](results/audit.json), [installed value evaluator](../.venv/lib/python3.14/site-packages/catanatron/players/value.py).

## Comparable games and policies

Keep two players, BASE board, \(10\) victory points, ordinary robber/discard behavior, bank/port trades, no domestic negotiation, and the existing turn/action limits. Both PPO arms must face the same opponent mixture during training. Changing opponents only for the curriculum arm would confound the comparison.

Use the same additional learner-transition budget and common final checkpoint for all six continuations, paired by the three parent checkpoints. Preserve the inherited optimizer state consistently. Report each parent pair as well as the aggregate. Do not compare a selected curriculum checkpoint with the standard arm's final checkpoint.

The primary full-game deployment policy uses the historical masked argmax action rule. Evaluate stochastic action sampling separately as a diagnostic, with independent recorded policy RNG. If sampling is adopted for deployment, choose it on validation data and evaluate that choice on a fresh test set.

Use a new reserved test-board namespace disjoint from training, curriculum source games, validation, historical failure examples and earlier reported tests. Rotate both seats on every board. Keep board generation, environment chance, opponent planning and policy sampling RNG streams independent. Record all seeds and source hashes. Matching a game seed pairs initial conditions; it does not guarantee identical realized dice or draws after policies diverge.

Target \(20\) fresh boards per opponent per policy for the current screening run: \(6\times5\times20\times2=1200\) full games. Respect the actual frozen coverage if the running pilot chose another count. Additional parent-reference evaluations must fit the existing budget or be deferred. Reduce coverage uniformly before testing, never by dropping bad games. Interleave arms and parent pairs by board so interruptions do not systematically deprive one arm of coverage. Retain partial records but compare only common completed paired blocks and disclose exclusions.

For a later confirmation of the selected frozen policy, target at least \(100\) fresh boards per opponent, both seats. This is a planning scale, not a guarantee of precision. At \(200\) independent games, the simple worst-case binomial \(95\%\) half-width is about \(7\) percentage points; pairing and shared boards change the actual uncertainty. Approximately \(400\) independent games would reduce that nominal half-width to \(5\) points. A small screening run cannot resolve a few percentage points reliably. More evaluation games also cannot replace independent training seeds. For demo acceptance, select one deployment checkpoint on validation data and test it once on fresh boards; the five-policy historical average is not the performance of one deployable model.

## Metrics and decision rules

For each opponent, report wins \(W\), losses \(L\), administrative truncations \(T\), and

\[
p_{\mathrm{win}}=\frac{W}{W+L+T},\qquad
p_{\mathrm{timeout}}=\frac{T}{W+L+T}.
\]

A timeout is an unresolved game and contributes no win. Report it separately; never discard it, infer a winner from points, or silently award half a win. Aborted or crashed evaluations are separate integrity failures. Show completed-game win rate only as a labelled secondary measure.

Report every opponent separately. Keep the legacy three-opponent average separate from the stronger two-opponent average; increasing the number of easy opponents must not inflate the main strength claim. For the supplementary stronger-opponent comparison, equally weight base and contender and all three parent pairs:

\[
\Delta_{\mathrm{strong}}=
\frac{1}{3}\sum_{s=1}^{3}\frac{1}{2}
\sum_{o\in\{\mathrm{base},\mathrm{contender}\}}
\left(p_{\mathrm{curriculum},s,o}-p_{\mathrm{standard},s,o}\right).
\]

Use paired board-cluster intervals retaining both seats and all opponents together. Also resample the paired training seeds and report this wider uncertainty separately. With only three parent pairs, both the interval and any apparent consistency remain preliminary. This follows the emphasis on uncertainty across runs in [Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264).

Proposed screening criteria, to freeze before reading the comparison:

- **Worth extending:** stronger-opponent gain of at least \(5\) percentage points, positive in at least two of the three parent pairs; no legacy-suite average drop greater than \(3\) percentage points; no observed increase in full-game timeout rate. These are practical screening margins, not significance tests or demonstrated noninferiority.
- **Evidence of superiority:** the paired seed-and-board interval for the prespecified endpoint excludes zero in the favorable direction, with per-opponent results and failure rates consistent with the claim. Replication across more independent training seeds remains necessary for a robust general conclusion.
- **Specialized benefit only:** better held-out midgame/endgame tasks without a clear full-game gain. Keep the curriculum as a candidate targeted finetuning method, without declaring it the stronger general player.
- **Completion quality:** target full-game timeouts below \(1\%\), with zero illegal actions and zero information-boundary violations. Report uncertainty: observing zero timeouts in a small sample does not establish a sub-\(1\%\) rate. Historical PPO-128 truncation was \(19/1500\approx1.27\%\).

For a performance milestone, distinguish “point estimate reached target” from “lower confidence bound exceeds target.” Avoid declaring \(51\%\) against a heuristic a reliable win over it when the interval spans parity.

## Failure-case evaluation

Keep the two known stalled games as named regression cases, not independent generalization evidence: parent \(155921\), board \(3100000010\), seat \(0\), Random; and parent \(196613\), board \(3100000019\), seat \(1\), legacy heuristic. Reconstruct earlier decision states from replay, preserving development-card age, phase and elapsed turn/action counters. Final timeout states alone cannot test whether the agent could have escaped earlier.

Create a separate held-out continuation suite from source games excluded from the training corpus. Split by source game before choosing snapshots. Test the following only where the required opportunity is verified from permitted information and actual legality:

| Case | Measure |
| --- | --- |
| Deterministic immediate winning move | Fraction taking an available winning action; aim for essentially all cases |
| Knight/army opportunity | Legal playable-knight opportunities taken, turns to obtaining an available army award, subsequent wins |
| Expansion bottleneck | New settlement or city within a prespecified own-turn horizon, subsequent wins |
| Productive bank/port trade | Completion of an available trade-and-build sequence, not merely trade frequency |
| Repeated end-turn behavior | Full observation including phase/mask/availability, action probabilities, progress and eventual timeout |

Freeze horizons and opportunity definitions before testing. Report task success and full-game wins separately, with favorable and unfavorable states separated. Do not reset the remaining game horizon or grant extra turns to curriculum snapshots. For stochastic continuation repeats, cluster by source game; repeated trials of one snapshot are not independent problems.

Compare greedy and sampled actions on the diagnostic suite using the same initial states and separately controlled future randomness. A sampled-policy rescue suggests a deployment/entropy issue; failure under both policies supports further investigation of learned preferences, credit assignment or missing inputs. First verify that intended actions appear in the mask and map to the correct engine actions. Repeated coarse public-state hashes alone do not prove a loop.

## Opponent validity and reporting

The rated ValueFunction opponents must use independent planning RNG, a fixed number of common information-consistent sampled worlds across candidates, mean float64 scores, and preserved upstream weights. They must not read the true hidden opponent hand, private victory points, deck order or live future RNG. Log sampling failures, fallback counts and opponent latency. A broken adapter does not count as a weak opponent.

At a curriculum reset, restore the opponent's saved permitted-history knowledge or reconstruct it from permitted history. Its public resource bounds cannot start at zero for a midgame hand. Validate this integration as well as ordinary full-game initialization.

The current adapter contains both the Params-based constructor and corrected tile comparison. Its SHA-256 `c79f7b15a0bdadb009bfd5442ef761f2efadf8a41ba3c57e5cbfd8fc160451e7` matches the frozen manifest and executed preflight. Earlier compatibility concerns and their association with this hash are superseded by the current source and completed contract checks. No search agent was included in the completed experiment.

The final report should contain the paired per-seed/per-opponent outcome table, timeout and diagnostic tables, uncertainty intervals, frozen evaluation manifest, source/checkpoint hashes, and complete-game coverage. Separate model-only inference from end-to-end decision latency; measure browser latency in the eventual deployment runtime. This no-negotiation evaluation does not assess natural trading or the combined LLM/tool agent.

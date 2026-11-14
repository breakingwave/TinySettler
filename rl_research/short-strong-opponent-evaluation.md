# Short evaluation of PPO against stronger heuristics

Proposed evaluation only; no runtime has been allocated for this plan. Use at most \(60\) minutes of allocated compute, with no training. Complete evaluator/search implementation and local smoke checks before starting that clock.

The questions are whether sampled action selection improves fresh games, whether the recent stronger-opponent continuation improved on its parents, and how one frozen deployment candidate performs against value heuristics and bounded search. The latest strong-family win rate is \(3.75\%\) for standard continuation; parity remains a longer-term milestone, not an expected result of this evaluation.

## Policies and source

Use the three standard PPO-128 final checkpoints for seeds `104729`, `130363`, `155921`, each with \(176128\) additional transitions, plus their unchanged \(434176\)-transition parents as diagnostic references. Use common final checkpoints, not a retrospective choice of the best training step. The tested curriculum did not demonstrate an advantage and is not needed in this short comparison.

The latest artifacts are on the persistent VPS at `/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a/`. Current checkpoints are `retrieved/standard-SEED/176128.pt`; the source/corpus/policy archive is `final-policies-and-corpus.tar.gz`, SHA-256 `2bcfacc77a101807544aecf9bdc625f6f1d51b34dd86b6d2276ced55b28e9802`. Resolve parent paths/hashes from the frozen `starting_checkpoints` manifest. These paths are on the VPS, not the Mac workspace.

Load with the exact archived source and dependency lock. Verify every hash and action mapping before execution. The existing full-game evaluator is greedy and tests all five opponents; it needs an evaluation-only extension for selected opponent subsets, sampled inference and the reference matrix. Keep that extension separate from the frozen training bundle. A T4 Colab runtime is the simplest compatible loader; the workload is CPU dominated. CPU-only extraction is acceptable only after checking logits/masks/actions against the archived loader. Do not disable provenance assertions to make a checkpoint load.

## Before allocation

Implement and locally check masked argmax and categorical sampling at temperature \(1\). Sampling uses an independent recorded RNG per game; it must not advance simulator or opponent-planning RNG. Do not tune temperatures on final-test outcomes. Record source, model and sampler hashes/configurations.

Use three disjoint new board namespaces for calibration, candidate selection and final testing. Check disjointness against all prior training, corpus, validation, test and diagnostic namespaces. The previous `880000000000` test range is already consumed. Keep both seats for each board; the same board IDs and recorded chance seeds are used across models/modes. Policy divergence can still change the realized random-event sequence.

The rules, observation, mask and episode caps stay identical. Administrative timeouts are not losses and not draws: retain them in the win-rate denominator and report them separately. Infrastructure interruptions are a different status.

### Search opponent prerequisite

Use adapted ValueFunction-base and contender unchanged. For the advanced challenge, create an information-safe bounded search opponent using the same base leaf weights. Do not run unmodified AlphaBetaPlayer as a rated opponent.

Correct forced dice/development-card transitions through `ActionRecord.result`; use sampled-hand frequencies for robber outcomes; isolate simulation RNG; sample hidden inventories/deck only from permitted information. Use the existing fixed \(K=2\) common-world sampler and record its failures. Validate probability sums, card conservation, legal actions, hidden-state invariance and live-state/RNG nonmutation.

For the first reference, disable unsafe ordinary alpha-beta cutoffs through chance expectations and compare with small exhaustive expectiminimax trees. The `prunning=False` parameter only disables heuristic action filtering, not these search cutoffs. Keep actor turns distinct from individual engine actions. Depth \(2\) means two actions, not two complete turns.

Enforce limits inside successor generation, not only at recursive entry. Complete a depth-\(1\) fallback before deepening. A proposed starting cap is \(2000\) generated successors and \(100\)–\(250\) ms per decision; choose and freeze the actual values from calibration, then report completed depth, node counts, time-cap hits and fallback rate. Label it “budgeted search targeting depth \(2\)” if full depth is not consistently achieved. The stock nominal \(20\)-second deadline is unsuitable for this short run.

Check the search opponent against its matched one-step value baseline on separate calibration boards. Until that comparison supports a strength advantage, describe it as a search challenge rather than assuming it is stronger. If the search correctness checks fail, omit the search stage with the reason recorded; complete the value-heuristic evaluation within the same cap.

## Allocated run

| Stage | Work | Maximum time |
| --- | --- | ---: |
| Integrity and timing | Verify runtime/checkpoints; fresh timing pilot for value opponents and search, including a slow-game case | \(5\) min |
| Candidate selection and parent references | Three standard finals, greedy and sampled, versus base/contender; also three parents under greedy inference | \(12\) min |
| Fresh confirmation and regression | Freeze one standard checkpoint; evaluate both action modes against base/contender and a small WeightedRandom check | \(13\) min |
| Search challenge | Evaluate the validation-selected checkpoint and mode against the validated bounded search opponent | \(20\) min |
| Retrieval and report | Save raw games, hashes and summaries; verify retrieval and terminate runtime | \(10\) min |

The cap includes all allocations/retries. Do not extend it to obtain a desired result. Freeze coverage after timing and before candidate-selection outcomes are read.

For selection, target \(20\) boards per opponent, both seats. The current-policy matrix has \(3\times2\times2\times20\times2=480\) games; greedy parent references add \(240\). Reduce uniformly to \(10\) boards if calibration cannot fit the budget. Compare parent versus continuation using their paired greedy results; do not compare a sampled final policy with a greedy parent and call the difference a training effect.

Select one standard checkpoint and preferred inference mode using equal-weight base/contender win fraction; break ties by fewer timeouts, then the predeclared seed order, then greedy mode. Save and hash-freeze that decision before final testing. This selects a deployment candidate; it does not estimate average algorithm performance. The parent references diagnose continuation and are not additional candidates in this plan.

For confirmation, target \(100\) new boards per value opponent, both seats, for BOTH greedy and sampled inference on the selected checkpoint: \(800\) games. Testing both modes gives a paired held-out inference comparison, while the deployment choice remains frozen from validation. Add \(20\) new boards against WeightedRandom under both modes: \(80\) games. If timing requires fewer boards, reduce all matched cells uniformly before looking at outcomes.

For search, target \(20\)–\(40\) new boards, both seats, using the selected checkpoint/mode: \(40\)–\(80\) games. Freeze the count and search budget from calibration. Search speed has not yet been measured. Retain partial records on interruption and disclose missing coverage; never drop slow or unfavorable games silently.

## Measurements and interpretation

For every opponent and policy mode, report wins, losses, timeouts, illegal actions, sampler failures and the observed capped win fraction

\[
p_{\mathrm{win}}=\frac{W}{W+L+T}.
\]

This is not an estimate of eventual uncapped win probability when some games remain unresolved.

Keep base and contender separate and also show their equally weighted average. They share one heuristic family. Report the paired sampled-minus-greedy difference with board-cluster intervals retaining both seats. For the parent-reference comparison, additionally show each parent pair and paired policy/board uncertainty; its small development sample is exploratory. Final confirmation concerns one selected model, not generalization across training seeds.

Record production/resource coverage after setup, new settlements/cities, time to first city/third settlement, final VP, development-card usage, immediate-win opportunities taken and timeout traces. Record end-to-end decision latency separately from network inference, and search nodes/depth/fallbacks. These explain whether defeats arise before the previous endgame failure cases can matter.

The fixed-policy heuristic target is improvement over its measured baseline, not an assumed \(50\%\). A substantial observed gain remains provisional if its paired interval crosses zero. Parity and \(60\%\) against a validated search opponent are longer-term milestones. Do not infer expert-human strength or negotiation quality from this suite.

Deliver one compact findings table, raw per-game records, parent/reference and inference comparisons, diagnostic traces, the frozen selection/evaluation manifest, source/checkpoint hashes, actual runtime and verified termination. Preserve the original completed experiment reports.

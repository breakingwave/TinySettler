# Fourteen hour PPO scaling experiment

Prepared 2026-11-13. This is an implementation and execution brief for the experiment orchestrator. No runtime has been allocated while preparing it.

Run one controlled study on Google Colab T4 through the Colab CLI. The hard limit is \(14\) hours, or \(50400\) seconds, from the first allocation request, including setup, evaluations, downtime, replacement runtimes and retrieval. Use at most one T4 runtime at a time. The orchestrator model reviews progress once every \(60\) minutes; deterministic infrastructure handles collection, checkpoints and ordinary recovery between those reviews. Deliver the report, charts, raw results and verified checkpoints at the end.

## Questions and design

Answer four questions: does continuously refreshed self-play improve external playing strength; does increasing model capacity help; does an architecture that represents board relationships help at comparable parameter count; and does teaching opening/economy decisions before PPO improve learning? Save intermediate checkpoints to measure training-duration scaling without creating separate short-run and long-run populations.

Use a balanced architecture-by-opponent-schedule matrix. Do not eliminate architectures because of noisy early win rates. All retained conditions receive the same learner-transition budget, evaluation boards and PPO settings. Current strong-opponent wins are near \(3\%\)–\(4\%\), and seed variability has been large; a broad single-seed sweep would be misleading.

| ID | Model | Opponent schedule | Training seeds | Main contrast |
| --- | --- | --- | ---: | --- |
| A | Compact normalized MLP-128 | Static pool | \(3\) | Reference |
| B | Compact normalized MLP-128 | Refreshed self-play | \(3\) | B versus A: adaptive opponents |
| C | Compact normalized MLP-512 | Static pool | \(2\) | C versus A: capacity under fixed opponents |
| D | Compact normalized MLP-512 | Refreshed self-play | \(2\) | D versus B: capacity under self-play |
| E | Structured graph/candidate model, parameter-matched to MLP-512 | Static pool | \(2\) | E versus C: architecture package |
| F | Same structured model | Refreshed self-play | \(2\) | F versus D: architecture under self-play |
| G | Same model as B, teacher warm start | Refreshed self-play | \(2\) | G versus B: opening/economy teaching |

This is \(16\) training runs. The first two seeds are paired across every condition; the third seed is reserved for the primary A/B comparison, independent of screening performance. Pick fresh seed IDs and record them before running. Within an architecture, clone identical initial weights across opponent schedules for each seed. G starts from the same random initialization as B before its teacher phase. All other models start from scratch with fresh optimizers. Do not warm-start MLPs from old checkpoints while training the graph model from scratch.

The static and adaptive conditions start with identical opponent pools. This isolates refreshing the neural opponents rather than confusing self-play with simply adding different opponents. The historical heuristic-only protocol remains a contextual reference, not a newly randomized control in this matrix.

Do not repeat the old late-state reset curriculum in this study. It already increased terminal-reward frequency without demonstrated fresh-game benefit. All PPO episodes here start from full games. G tests a different, explicit curriculum: learn initial placements and early economy from the stronger heuristic, then use ordinary full-game PPO. Do not add goal conditioning, reward shaping, recurrent memory or an optimizer sweep to this matrix.

## Read the existing evidence first

Read [the implementation and results summary](rl-summary-20261111.md), [latest findings](results/ppo-curriculum-20261111/findings.md), [resource accounting](results/ppo-curriculum-20261111/resource-summary.json), [current evaluator](curriculum/curriculum_evaluate.py), [trainer](overnight/train.py), [environment](overnight/env.py), [opponent adapter](curriculum/opponent_adapter.py), and [existing controller](curriculum/curriculum_controller.py).

The current path is CPU dominated. Recorded throughput was \(173\)–\(215\) learner decisions/s with one worker, and GPU utilization averaged \(6.4\%\). Do not project this unchanged onto a graph model or a two-worker setup. The previous approximately \(13478\) engine-actions/s microbenchmark was not training throughput.

The local working tree is `/Users/alexanderwang/projects/TinySettler`; the synced VPS container uses `/workspace/TinySettler`. Persistent large artifacts belong outside the synced checkout under `/root/tinysettler-artifacts/`. Verify the live paths and CLI on the execution host before use. Create a new dedicated source/artifact directory; do not edit historical bundles or results.

## Models and shared information

Keep the same two-player BASE rules, \(10\)-VP goal, no domestic negotiation, \(332\)-action mapping, legal masks, turn/action limits and learner-to-learner transition semantics. Every model receives the same permitted facts. Exclude true hidden opponent inventories/VP, deck order, exact bank composition and RNG state. No model receives extra public-history beliefs in this experiment.

Create one versioned common observation schema from the existing \(2049\)-vector: remove the \(1170\) constant topology coordinates to give \(879\) dynamic/context coordinates, and apply a documented deterministic per-feature scaling table. Preserve binary/categorical fields; scale counts using known physical/rule units. Do not fit scaling on validation or final games, accidentally scale an already-normalized field twice, or clip distinct valid states to one value. Board adjacency remains available as a fixed structural constant to the graph encoder. The common preprocessing change is not itself an isolated ablation against the old model.

The MLPs keep two Tanh hidden layers, a linear \(332\)-logit head and a scalar value head. With \(879\) inputs, MLP-128 has \(172109\) parameters; MLP-512 has \(884045\). This is approximately \(5.1\times\) parameter scaling, a larger contrast than the previously inconclusive width-\(256\) test.

For the structured model, use typed tile, intersection and edge features with fixed-adjacency message passing, a global own/public-player context, shared semantic candidate scoring and a scalar value head. Use three message-passing rounds and ordinary vectorized PyTorch operations, not Python loops over candidates or a new graph framework. Choose the hidden width from a small predefined grid solely to place the total trainable parameter count within \(20\%\) of MLP-512. Freeze the architecture before measuring wins. Candidate descriptors must map back to the identical \(332\) action IDs. Keep legality handling identical. The model tests a graph-plus-shared-action-head package; do not attribute a gain to message passing alone.

Make the structured encoder consume only the common permitted observation and fixed topology, not unrestricted Game objects. Test the observation/action roundtrip, seat swapping, legal masks, and invariance to hidden-state changes that preserve permitted information. Spatially structured Catan networks have prior support, but this particular model is untested: [Gendre and Kaneko](https://arxiv.org/abs/2008.07079).

Do not call topology folding or widening an exact training resume: preserving initial predictions does not preserve Adam's future updates. Historical checkpoints remain unmodified inference references. New models use a new strict schema/source/configuration manifest rather than disabling the old loader's checks.

## Opponent schedules and self-play

Construct an immutable initial neural archive from the three completed standard PPO-128 policies and their three parents. Verify their hashes and exclude them from any later claim of unseen-opponent generalization. Set aside a historical PPO policy from a different training seed/lineage, chosen by a fixed seed/order rule, as an evaluation-only neural opponent; another checkpoint from an archive policy's lineage is not unseen. Do not choose it by testing the new models. Historical archive/reference policies require their original \(2049\)-feature encoder; new policies require their own \(879\)-feature or structured encoder. Dispatch by frozen checkpoint schema, not by the learner's current architecture. An inference-only export must first match the provenance-checked archived loader.

Define the heuristic anchor mixture H as \(50\%\) adapted ValueFunction-base, \(25\%\) WeightedRandom and \(25\%\) legacy heuristic. Every condition spends \(50\%\) of episode resets against H, so its absolute probabilities are \(25\%\), \(12.5\%\), \(12.5\%\). The remaining \(50\%\) is neural opponents:

| Schedule | Neural-opponent half of episode resets |
| --- | --- |
| Static | \(25\%\) total probability on one predeclared reference checkpoint, \(25\%\) total uniformly over the immutable initial archive |
| Refreshed | Identical to static until the first publication; thereafter \(25\%\) total on the latest own snapshot and \(25\%\) total uniformly over the initial archive plus retained own snapshots |

Publish an immutable own-policy snapshot every \(65536\) learner transitions, after a complete PPO update. Retain the initial archive and up to eight own snapshots spread over training time using a deterministic retention rule; never remove an opponent because the learner loses to it. Newly generated snapshots are not shared between experimental runs, architectures or seeds; only the immutable initial archive is common. Keep contender and search out of both training and teacher data.

Select the opponent at episode reset and keep its checkpoint fixed for the whole episode. Use the acting opponent's own observation and actor-relative mask, `eval()`/`no_grad()`, and its own saved action-sampling RNG. All neural opponents use masked categorical sampling at temperature \(1\). They do not update during an episode. Isolate learner, opponent, environment, snapshot-selection and evaluation RNG streams.

Log both episode proportions and learner-transition proportions: stronger opponents can finish games faster, so \(50\%\) of resets is not \(50\%\) of the training data. Self-play win rate is a training diagnostic, not the primary strength measure. Preserve the refresh schedule in absolute steps, including through recovery. Early own snapshots may be weaker than the historical references they replace; this is a valid adaptive-opponent intervention, but improvement would not prove that monotonically stronger opponents caused it. Describe snapshot strength rather than adding a new promotion gate.

## Teacher curriculum G

Prepare a bounded dataset with the audited base adapter, using legal games and only permitted observations/action labels. Cap generation at \(200\) source games and \(25000\) stored non-forced decisions. Before generation assign \(160/20/20\) source games to train/validation/test, with fresh seeds and rotated teacher seats. Opponents follow a fixed four-game cycle: base, base, WeightedRandom, legacy heuristic. Record only the designated teacher's decisions. Never reuse existing evaluation or diagnostic source games. Use a seeded reservoir within each split if the decision cap is reached, reserving \(20000/2500/2500\) records for train/validation/test rather than filling the cap with training games first.

Define three exclusive strata: initial setup; non-setup decisions with own actual VP at most \(4\); and remaining decisions. During teacher optimization sample these with probabilities \(0.2/0.4/0.4\), respectively. If a stratum is empty, renormalize over the available strata once and record that fact before fitting. Report unique source games/states separately from repeated training presentations. Do not adjust the dataset composition after inspecting PPO outcomes.

Use three fixed epochs of masked action cross-entropy, batch \(256\), Adam learning rate \(3\times10^{-4}\), on the actor/body. Do not regress onto the heuristic's enormous raw scores, invent value targets, or change PPO rewards. The critic head starts normally. At PPO start, create a fresh PPO optimizer, as for every other new model. No teacher loss continues during PPO; no DAgger loop is added during this run.

Record teacher-generation decisions, elapsed CPU time, supervised examples/updates and test action agreement by phase. Charge these costs when comparing total data and wall time, even if data preparation occurred before GPU allocation. G versus B isolates the benefit of this extra teaching procedure, not a gain at equal total training data. The teacher has public-history information that the memoryless learner may not fully reconstruct, so report imitation errors rather than assuming perfect realizability.

## PPO settings and instrumentation

Retain the established PPO update: rollout \(4096\), four epochs, minibatch \(256\), Adam \(3\times10^{-4}\), clip \(0.2\), GAE \(\lambda=0.95\), \(\gamma=1\), entropy coefficient \(0.01\), value coefficient \(0.5\), gradient norm cap \(0.5\), terminal \(+1/-1\) rewards and otherwise zero. Preserve terminal/truncation bootstrapping and rollout masks. Do not change these at hourly reviews.

Log policy loss, value loss, entropy, approximate KL, clip fraction, pre-clipping gradient norm, critic explained variance, advantage/return statistics and non-finite counts. Undefined explained variance must remain undefined, not zero. Include per-phase/action-type probabilities, legal opportunity counts, actual knight/build/trade use, first city/third settlement timing, final points, truncation causes, opponent identities, snapshot ages, transitions/s and CPU/GPU utilization. Store bounded failure traces, not unlimited full-state dumps.

## Budget and throughput calibration

| Work | Maximum planned allocation |
| --- | ---: |
| Runtime setup, timing and exact-recovery preflight | \(45\) min |
| PPO and teacher optimization across the matrix | \(600\) min |
| Intermediate/final-validation evaluations | \(45\) min |
| Untouched final evaluation and search challenge | \(90\) min |
| Retrieval, verification and initial report | \(30\) min |
| Planned runtime handoff or recovery reserve | \(30\) min |
| Total | \(840\) min |

Finish implementation, small correctness tests and teacher-data preparation before requesting the T4. A graph model that has not passed a tiny collection/update/checkpoint smoke test is not ready for allocation.

On the actual T4, measure every model/opponent schedule, including a populated self-play archive with snapshots of that condition's own architecture. Timing only the old small MLP opponents understates later graph/MLP-512 costs. Freeze and record opponent inference device, model caching and transfer behavior. Use fresh \(8192\)-transition pilots with actual optimization and end-to-end timing. Measure fresh evaluation games, including a weak/slow policy; never reuse cached evaluation timing. Discard pilot weights before the study. Benchmark one worker versus two independent one-thread training processes sharing the single T4. Use two only if aggregate throughput improves and both pass isolated-RNG/recovery checks. Do not simultaneously rewrite the rollout/GAE implementation for vectorized environments.

Freeze one common target N for every retained run, a multiple of \(16384\), after calibration and before any experimental results. Aim for \(524288\), allow up to \(786432\) if measured throughput supports it, and seek at least \(393216\). Simulate the complete fixed schedule with per-condition single/dual-worker costs, teacher optimization, checkpoint overhead and evaluation costs. Add a \(20\%\) duration margin and require projected training to fit its \(600\)-minute allocation. Do not assume a two-worker speedup or use one cheap opponent's rate for the whole matrix.

For scale: \(16\times393216=6291456\) transitions already take about \(9.7\) hours at an aggregate \(180\) transitions/s, before a duration margin. Thus the full matrix at that budget needs either better measured aggregate throughput or a declared reduction in scope. These are planning calculations, not throughput promises.

If the full matrix cannot reach the minimum target within the conservative projection, apply this order BEFORE experimental outcomes: omit G; then, if necessary, omit both E and F together. Recompute N after each change. Never silently omit self-play, remove only the slower losing member of a comparison, or sacrifice the third A/B seed to preserve a larger model. If even A–D cannot fit useful training and the reserved evaluation, run the largest feasible matched A/B experiment and mark capacity/architecture unanswered. Record every preflight scope decision in the manifest.

Save and evaluate at common checkpoints \(N/4,N/2,3N/4,N\). Round-robin complete seed-pair chunks across conditions, rotating order each round, rather than giving one model all the early hardware time. Keep unfinished episodes and rollout/RNG state across chunks. No performance-based early stopping; only predeclared numerical/integrity stops. If a later interruption prevents full completion, show missing cells and compare the largest common checkpoint rather than ranking unequal budgets.

Unused reserves may expand a predeclared final-evaluation board prefix before outcomes are inspected. Do not add new training conditions, selectively extend a winning run or increase N after seeing curves.

## Evaluation and selection

Primary strength is fresh full-game performance against three frozen opponents: adapted base, held-out contender and the reserved historical neural policy. Keep each opponent's result separate; base and contender belong to one heuristic family. Report the equal-weight base/contender aggregate as the main heuristic-strength endpoint, with the unseen neural opponent as a separate generalization endpoint. Random, WeightedRandom and the legacy heuristic are regression checks, not ways to inflate the headline average.

Training action selection is stochastic, so use masked categorical sampling at temperature \(1\) as the predeclared primary evaluation mode. Greedy is a separately reported secondary mode. Never switch the headline mode after inspecting tests. Both modes use identical game rules and separate policy RNG. Match initial boards and seats across conditions; do not claim that divergent policies experience identical realized random-event sequences.

Use entirely new namespaces for teacher source games, pilots, training, validation and final testing; verify disjointness from every earlier experiment and corpus. Use both seats on every board. Do not change a policy using validation actions or expose final-test outcomes during training.

For intermediate validation, target \(5\) boards, both seats, against the three main opponents under sampled inference at N=0 and at each of the first three milestones: \(30\) games per checkpoint. For G, N=0 means after teacher initialization but before PPO. At N, target \(10\) validation boards against the same opponents under both modes. For \(16\) runs that is at most \(3840\) validation games; identical initial policies may share explicitly identified cached rows, but cached evaluations cannot be used as timing pilots or counted as independent games. Freeze a smaller uniform count before the sweep if the pilot cannot fit the \(45\)-minute allowance. These small panels track direction, not precise rankings.

All common-final policies receive the primary test, not only a selected winner. Target \(20\) untouched boards, both seats, three main opponents, both modes: \(16\times20\times2\times3\times2=3840\) games. Add \(5\) boards per weak regression opponent under sampled inference for every policy: \(480\) games. Evaluate the three historical standard policies as context on the same main panel, \(720\) further games. Historical references use their original encoder and are not a controlled preprocessing ablation.

Before opening final tests, choose one deployment candidate and mode using final-validation base/contender win fraction, then fewer timeouts, then a fixed simplicity/seed-order tie-breaker. Hash-freeze the choice. Within the final budget, extend that candidate's untouched board prefix to \(100\) boards against base and contender in the selected mode; the first \(20\) boards already exist, so this adds \(320\) games. If the common prefix is B boards after calibration, the extra count is \(4(100-B)\). Candidate selection does not replace reporting every trained policy, and its confirmation describes one model rather than an average algorithm.

Reserve up to \(20\) of the final \(90\) minutes for the selected candidate against a repaired, information-safe search opponent targeting depth \(2\), ideally \(20\)–\(40\) boards with both seats. Complete search correctness checks before allocation: forced chance outcomes via ActionRecord, correct sampled-hand robber probabilities, isolated RNG, no hidden-state access, and no invalid deterministic alpha-beta bounds across chance expectations. Enforce node/time limits during expansion and report achieved depth and fallback frequency. See [search evaluation details](short-strong-opponent-evaluation.md). If search is not validated or calibrated to fit, omit it transparently and preserve the full main evaluation. Do not call raw stock AlphaBeta a valid strong benchmark.

The pilot freezes all board counts, search limits and final-stage priorities. The target is \(5360\) non-search final games plus any search challenge; recompute counts from retained conditions and board counts if the scope changes. Do not assume these fit merely because old small-network games were fast. Priority order is sampled primary tests for all policies, selected-candidate confirmation, paired greedy tests, weak/context checks, then search. Reserve search time only when the calibrated core coverage fits. Reduce planned counts uniformly before the sweep if needed; do not reduce coverage after seeing adverse results. Run complete matched board blocks; retain partial records and disclose any exclusions caused by the deadline. Report wins, losses, administrative truncations and infrastructure failures separately. The observed capped win fraction is \(W/(W+L+T)\), not eventual uncapped win probability. For city/settlement timing, also report the fraction that never reaches the event; conditional timing alone can hide more failures.

Report paired seed-level contrasts, board-cluster \(95\%\) intervals retaining both seats, and a separate crossed seed/board analysis. Use only the first two common seeds for capacity, architecture, teacher and factorial interaction contrasts; use all three pairs for the primary B-versus-A comparison. Two-seed architecture/curriculum effects remain exploratory; three A/B pairs are still a small study. Show individual seeds and avoid an unsupported winner claim when intervals are broad. For factorial interactions, compare self-play-minus-static within each architecture, then the differences of those contrasts. Do not treat thousands of games as thousands of independent training runs. [Evaluation methodology](https://arxiv.org/abs/2108.13264).

## Hourly orchestration and unattended recovery

After launch/preflight, the LLM orchestrator performs scheduled reviews only at \(t_0+60k\) minutes. Each review reads one compact machine-generated status file, the latest chart bundle and at most one bounded log tail. Do not continuously poll terminals or request model commentary every few minutes. Notify immediately only for completion, a fatal integrity failure, a recovery condition beyond the frozen allowance, or information requiring user action.

Each hourly note should fit a short paragraph or table: completed/pending runs and common step; validation strength by opponent; any timeout/expansion trend; measured throughput and projection; age of last externally verified checkpoint; remaining budget; and any predeclared recovery action. Do not tune learning rates, rewards, pools, temperatures or architectures based on that review. Final-test results remain unavailable to these reviews until the selection manifest is committed.

This sparse MODEL cadence does not apply to deterministic safeguards. The training process writes health/progress every minute, atomic recovery checkpoints every five minutes, and immutable external hash-verified generations every ten minutes. Preserve the last two verified generations. A background controller can check liveness and transfers without invoking the LLM. Keep a separate deadline watchdog. Do not create a recurring ChatGPT automation to watch this one experiment.

Checkpoint models, optimizers, current environments, partial rollouts, all RNG states, completed evaluation blocks, active opponent hashes, archive contents, snapshot-publication counters, teacher provenance and original deadlines. Never publish a generation manifest before its files are complete. Test an external copy/upload/resume round trip including frozen-opponent state before the sweep. A recovery must not reset the learning schedule or silently count rolled-back work as retained training.

Distinguish a CLI connection timeout from worker failure. Retry transient read/transfer failures with bounded backoff; do not stop a healthy runtime merely because one status request failed. Permit at most one planned lifetime handoff and one fault replacement, always after confirming the previous runtime is terminated. All attempts share the original clock and remaining budget. Recovery affecting learning semantics invalidates that comparison; preserve the evidence and mark it rather than silently changing code.

## Colab runtime lifetime

Check installed CLI commands and existing account compute-unit balance before allocation. Use a unique session name and explicit `--gpu T4`; do not upgrade the CLI or dependencies mid-run. Confirm the real GPU and CUDA updates. Standard Colab runtimes generally cannot be assumed to last beyond \(12\) hours; Pro+ documents up to \(24\) hours with sufficient compute units. Use an eligible existing account, or plan a durable stop-and-resume to another available paid runtime around hour \(10\), within the original \(14\)-hour clock. Do not work around a denied quota with another account or an indefinite retry loop. Availability remains a constraint. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html), [official CLI](https://github.com/googlecolab/google-colab-cli).

Build the detached supervisor on persistent VPS storage. Model review can occur hourly while the active job and deterministic supervisor continue normally. Do not depend on frequent LLM interaction to keep the runtime alive, and do not fabricate activity to evade platform limits. Stop only this experiment's sessions; verify `colab sessions` after termination. Preserve credentials and unrelated work.

## Final report, charts and decision

Write a self-contained Markdown report and machine-readable CSV/JSON outcomes in a new results directory. Render publication-usable PNG and SVG charts from saved data, inspect their labels/legends, and include the exact configuration, seeds, split manifests, source/model hashes, archive locations, failed/incomplete cells and runtime ledger. Finish chart rendering on the persistent CPU host after releasing the T4 if needed; the compute deadline must not be extended for presentation work.

Required charts:

1. Strong-opponent validation wins versus learner transitions, with each seed visible and common checkpoints aligned.
2. The same outcomes versus actual elapsed training time, plus teacher-inclusive total compute time. Shared-GPU worker times must not be mistaken for additional GPU-hours.
3. Final architecture-by-opponent-schedule heatmaps, separately for base, contender and held-out PPO; distinguish sampled and greedy modes.
4. Paired self-play, width, architecture and teacher contrasts with uncertainty and raw per-seed points.
5. Win rate versus parameters and end-to-end inference latency, showing the strength/cost tradeoff rather than only largest-model wins.
6. First-city/expansion progress, immediate-win conversion, knight/trade opportunities and timeout rates.
7. PPO entropy/KL/clipping and critic-quality curves; undefined metrics and failed runs remain visible.
8. Opponent-pool composition and snapshot ages over training, with actual episode and transition shares.
9. CPU/GPU utilization, transitions/s, checkpoint-copy age and allocation/recovery timeline.

Answer explicitly: did refreshing opponents help at fixed initialization and budget; did more parameters help at equal information/steps; did structure help at matched parameter scale; did teacher initialization help enough to justify its extra data/cost; were larger models merely slower to learn; which results persist across seeds; and what evidence remains inconclusive? Report whether the earlier selected-failure sampling advantage transfers to ordinary games. Self-play parity alone is not a positive result.

Recommend one next direction with its strongest counterevidence. If all models still lose overwhelmingly to the value heuristics, say that this budget did not establish a scalable learner; use the opening/critic/teacher diagnostics to choose the next targeted test rather than recommending a larger GPU by default. Preserve the best validation-selected policy for later deployment, but do not present PPO logits as calibrated action values or claim that negotiation has been solved.

Completion means artifacts were retrieved and hash-verified, results and charts are present, incomplete work is disclosed, the final analysis is delivered, and all dedicated runtimes are confirmed stopped.

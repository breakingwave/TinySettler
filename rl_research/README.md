# TinySettler reinforcement learning design and local measurements

Completed study, 2026-11-14: [recovered scaling results, charts and raw data](results/scaling-20261114-recovered/findings.md), [independent interpretation and next test](results/scaling-20261114-recovered/analysis.md), and [implementation and reproduction](scaling/README.md). Ten matched-budget MLP runs completed; refreshed self-play and larger capacity did not establish reliable gains against the strong heuristic family. Graph and teacher conditions were omitted before outcomes.

Artifacts for all ten retained November 14 A–D runs are in the [model release](https://github.com/breakingwave/TinySettler/releases/tag/rl-scaling-20261114): [portable policy exports](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/final-delivery.tar.gz), [full resumable checkpoints](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/full-checkpoints.tar.gz), and [SHA256 checksums](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/SHA256SUMS.txt). Download these release assets separately from the repository.

The same release includes earlier experiments: [overnight selected checkpoints](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/overnight-selected-checkpoints.tar.gz), [reliability common-final and selected checkpoints](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/reliability-common-final-and-selected.tar.gz), and [curriculum final policies and corpus](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/curriculum-final-policies-and-corpus.tar.gz).

Original study plan, 2026-11-13: [fourteen-hour scaling experiment and orchestrator instructions](fourteen-hour-scaling-plan.md), comparing capacity, structured architecture, refreshed self-play and teacher initialization with hourly model reviews.

Updated synthesis, 2026-11-11: [implementation, all three experiments and next priorities](rl-summary-20261111.md), with a [short strong-opponent evaluation plan](short-strong-opponent-evaluation.md). The completed curriculum comparison found no demonstrated full-game benefit; both PPO arms remain weak against the adapted value-function heuristics.

Follow-up, 2026-11-11: the [completed five-seed PPO reliability experiment](results/ppo-reliability-20261111/findings.md) compares common-final and validation-selected policies on fresh held-out boards. [Reproduction and recovery commands](reliability/README.md).

Follow-up, 2026-11-11: the [completed T4 ablation](results/overnight-20261111/findings.md) reports six PPO/Double DQN runs and 3,600 held-out evaluation games. [Implementation and reproduction commands](overnight/README.md) are available separately from this original design report.

Research date: 2026-11-04. Scope: the reinforcement learning component of a browser agent for two-player Catan, with an LLM choosing strategies and calling an action evaluator.

Start with masked PPO and a small multilayer perceptron to establish a playing baseline. Compare masked Double DQN when implementing the planner's action-value tool. Keep Catanatron, add an explicit information-safe observation and action adapter, and postpone full tree search until a learned evaluator works. These are engineering recommendations, not results of an algorithm tournament. The local work measured simulator costs and audited interfaces; it trained no neural networks.

## Rules and information contract

The existing [browser bridge](../backend/catan_server.py) runs four players, with RED facing three WeightedRandom bots. It does not expose player trade offers. The local benchmark instead instantiated two active players on the BASE board, with a target of \(10\) victory points, discard threshold \(7\), ordinary robber rules and no player offers. This preserves the demo while measuring the intended player count.

Use that simple two-player configuration for the first RL baseline, then add bounded bilateral trades as an explicitly versioned custom variant. Do not call it official CATAN for Two: those rules include neutral players, additional production rolls and trade tokens. They are materially different. [Official rules, printed pages 10–11](https://www.catan.com/sites/default/files/2021-07/catan_traveler_english_rules_2021_201130.pdf).

Freeze victory target, discard threshold, robber rules, bank visibility, trade protocol and episode limits before comparing policies. A policy trained without player trades must be evaluated again once trades are enabled.

## Board and hand encoding

Use an actor-relative numeric observation. The installed base map has \(19\) land tiles, \(54\) intersections and \(72\) edges. Flatten these for the first MLP; retain the topology so a graph encoder can replace it later.

| Component | Recommended contents |
| --- | --- |
| Tiles | Resource category, exact dice-token category, production probability, robber indicator |
| Intersections | Empty/self/opponent ownership, settlement/city, adjacent tiles, port type |
| Edges | Empty/self/opponent road ownership and endpoint indices |
| Own player | Exact resource and development-card counts; development-card availability this turn; true own points; remaining pieces; knights, road length, port exchange rates |
| Opponent | Public points, total resource/development-card counts, played cards, pieces, road and army status |
| Phase | Acting player and turn owner separately; setup, roll, build, discard, robber and trade states; remaining discards/free roads; turn count and rule configuration |
| History and belief | Public production, builds, trades, revealed cards and robber events; permitted private events; estimated opponent resource counts and uncertainty |
| Trade context | Current offer, previous response, proposal budget, give/receive vectors and counterparty |

Normalize counts by known ranges; use categorical features for identities. Never feed the policy the simulator seed, RNG state, deck order, opponent hidden victory points or inaccessible card identities. Preserve exact dice labels: \(6\) and \(8\) have equal probability but different simultaneous-production correlations. The stock encoding stores only probability.

The stock two-player vector has \(614\) features, but it includes exact per-resource bank counts. In a two-player game,

\[
H_{\mathrm{opponent},r}=19-B_r-H_{\mathrm{self},r}.
\]

Those fields reveal the entire opponent resource hand. The [audit](results/audit.json) reconstructed it exactly and verified a synthetic card-conserving hand/bank swap that changed only bank features. That swap is an encoder sensitivity check, not proof of two reachable states with identical observable histories; initial hands at the sampled setup state are already inferable from placements. The algebra establishes the general exact-reconstruction risk. Exclude the five resource-bank counts if matching the current browser's hidden-bank interface. The remaining \(609\) stock features are a timing baseline, not a complete recommended observation: phase, exact dice labels, development-card age and history still need improvement. Legal masks may reveal legitimate bank shortages; do not conceal information that is genuinely available through permitted actions.

Start opponent inference with a ledger of visible gains/spending and uncertainty introduced by hidden discards or unobserved transfers. Add bounded history or belief means/ranges before a recurrent network. In two-player play, much resource information can be inferred legitimately; preserve that information instead of replacing every opponent hand with an arbitrary uniform guess. A later particle belief should satisfy card conservation, observed events and hand totals. Never initialize it from the true hidden hand. Stock SB3 MaskablePPO currently lacks recurrent policies, so masked recurrence requires deliberate additional implementation. [Official documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## Legal actions and trading

The rules engine must determine legality during training and inference. Mask impossible actions before sampling or taking an argmax; do not spend training effort learning legality through penalties. In PPO, reuse the rollout-time mask in the update. In DQN, mask both exploratory choices and the next-action selection in the target. This follows the established policy-gradient masking approach. [Masking study](https://arxiv.org/abs/2006.14171).

The installed fixed action table contains \(332\) actions for two-player BASE, not the \(290\) or \(328\) found in other versions or stale documentation. All mappings roundtripped in the audit. It omits domestic offer/response actions. Use the fixed table for the initial no-trade control; use a shared candidate scorer for the eventual trading model:

\[
z=f_\theta(b),\qquad q_a=q_\theta(z,e(a),e(g)).
\]

Here \(b\) is the permitted observation/history summary, \(a\) is a structured action and \(g\) is an optional goal. Encode type, target node/edge/tile, give/receive quantities and counterparty. Compute \(z\) once, then batch all candidate scores. Never treat the current menu index as a stable action meaning in replay. Store the descriptor, candidate set or stable semantic mapping, mask and observation version.

Begin trade offers with bounded templates such as \(1{:}1\), \(2{:}1\) and \(1{:}2\), with a small proposal budget per turn. Retain reject, cancel and end-turn choices. This restricts the action space and therefore defines a restricted trading variant. Measure candidate coverage before broadening it. Offer generation may check our own inventory; it must not filter using the opponent's hidden inventory. Acceptance legality belongs to the responding player's view.

Add validation for nonnegative integer quantities, exact vector length, phase, affordability and final execution. The installed offer helper accepts negative and fractional counts in the audit; generated ordinary moves do not establish safety for unrestricted offers. Continue using the bridge's version check for stale selections.

Do not reward making or accepting trades. In a win/loss two-player game, both players gaining resources does not imply both increase their winning probability. Strong agents may trade infrequently; frequent natural trading is a separate behavioral target, and any changed utility should be explicit. Repeated poor offers can exploit an occasionally mistaken acceptance policy. Train against several acceptance styles and evaluate repetition, rejection, acceptance and eventual wins separately. Natural language can realize an approved offer; the RL interface should consume its structured meaning. The Settlers-RL project documents unreasonable trade behavior despite extensive training. [Author's account](https://settlers-rl.github.io/).

## Training baseline and the planner tool

Use masked PPO first because it provides an established policy baseline with straightforward legal-action handling. Start with two hidden layers of width \(256\), public/history features, terminal win/loss reward and a fixed mixture of weak and audited heuristic opponents. This choice is not a demonstrated Catan optimum. PPO has relevant Catan precedents, while Double DQN provides the more direct action-value interface. [PPO](https://arxiv.org/abs/1707.06347), [Double DQN](https://arxiv.org/abs/1509.06461).

| Method | Role | Main issue |
| --- | --- | --- |
| Masked PPO with policy and state-value heads | First playing baseline | Policy logits are preferences, not action values |
| Masked Double DQN with shared action scorer | Direct \(Q(b,a)\) tool and algorithm comparison | Hidden-state uncertainty and stale replay as opponents change |
| PPO plus action-value regression | Distill candidate evaluation for cheap browser inference | Unchosen actions need coverage and values need calibration |
| Policy plus belief search | Later strength/latency experiment | Requires correct chance handling and a simulator at inference |

For a concrete first PPO configuration, use learning rate \(3\times10^{-4}\), clip \(0.2\), GAE \(\lambda=0.95\), four update epochs, batch size \(256\) and \(4096\) collected learner decisions per update. These are starting settings, not measured recommendations. Check entropy by phase and completion rate before increasing the training budget. Keep the LLM outside the training loop; train an inexpensive structured strategy selector or scripted goal sampler instead.

Define the tool's meaning explicitly:

\[
Q^{\pi,\mu}(b,a)=\mathbb E[G\mid b,a,\text{our continuation policy }\pi,\text{opponent mixture }\mu].
\]

A scalar \(V(b)\) cannot rank actions without transitions. Initially, use the PPO policy to shortlist candidates and estimate action values through short belief-sampled rollouts ending at learner decision states, using \(V\) at leaves. Later regress those targets into a shared \(Q\) head, or use Double DQN's direct scorer. A PPO action probability must never be labeled a win probability. For unshaped terminal rewards \(\{-1,+1\}\) and \(\gamma=1\), a calibrated expected return can be transformed as \(p_{\mathrm{win}}=(Q+1)/2\); raw learned scores are not automatically calibrated.

The LLM can request candidate rankings, goal-success estimates and estimated game value. Return the model version, continuation-policy assumption and uncertainty/coverage status. Evaluate the combined LLM/tool agent separately: repeated LLM overrides change the continuation assumed by the evaluator. Do not combine arbitrary numerical scores with arbitrary strategy weights.

One learner transition should advance from its decision to its next decision, including intervening opponent/chance actions. A Catan turn can contain several actions, and discards/trade replies can occur out of turn. Never negate a value merely because one primitive action elapsed. For Double DQN:

\[
y=r+\gamma(1-d)Q_{\bar\theta}\!\left(b',\arg\max_{a'\in A_{\mathrm{legal}}(b')}Q_\theta(b',a')\right).
\]

Here \(d\) marks a genuine terminal state. Bootstrap through administrative truncation using the final observation where appropriate; if a move limit is an actual game rule, encode it and define its terminal payoff. Store elapsed turns as well as decisions. With terminal win/loss objectives, \(\gamma=1\) avoids unintentionally preferring quick games; arbitrary per-action discounting can penalize useful action sequences. Bound automatic advancement and negotiation loops.

## Search and MCTS

Do not make MCTS a prerequisite. First compare greedy action values with a small amount of candidate rollout evaluation. If that improves decisions, implement belief sampling or information-set search. Standard perfect-information AlphaZero search is not directly applicable to hidden hands, dice and negotiation. [ISMCTS](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf), [Catan POMCP work](https://ojs.aaai.org/index.php/AIIDE/article/view/13014).

A minimal implementation samples hidden hands/deck order consistent with the root's information, selects actions with policy priors, samples chance outcomes, runs an opponent policy using only its permitted view, and backs up root-player returns. Key search statistics must correspond to information available at that node, rather than revealing sampled hidden identities. Naive independent determinization can suffer strategy fusion; it is only a cheap baseline. At opponent nodes, use the opponent model or properly signed opponent utility, not cooperative maximization of our score. Final root selection should use visits or estimated value, not the exploration bonus.

The installed search implementations require repair before use as benchmarks:

| Finding | Consequence |
| --- | --- |
| `Game.copy()` and `State.copy()` share RNG objects | Search advances the live game's random stream |
| `execute_spectrum` passes desired dice through `Action.value`, but execution reads `ActionRecord.result` | The claimed chance branches have incorrect realized outcomes and weights |
| Development-card spectrum uses a similar outdated override | It does not enumerate the intended card outcomes |
| Search copies the true hidden state | Opponent moves, Monopoly and card draws can exploit inaccessible information |
| Stock MCTS maximizes root scores at opponent nodes | It is not an audited adversarial reference |

The roll probe requested \(2,\ldots,12\) but obtained \([8,9,12,6,8,11,4,8,9,11,8]\). The scripts isolate their probe roots, so they do not mutate the demo. New search needs separate simulation RNG streams, explicit chance outcomes, belief-sampled states and regression checks. Cloning the real RNG state is useful for controlled tests, but deployed search must not use it to predict actual future randomness.

A later search-cost sweep can compare \(0\), \(32\) and \(128\) simulations on held-out positions, followed by complete-game evaluation. Fix both a simulation budget and a deadline; record actual transitions and neural evaluations. This sweep has not been run.

## Curriculum and goal conditioning

First validate complete legal games, information boundaries, rewards and turn handling. Start training against Random plus WeightedRandom, then include an information-safe production/expansion heuristic. Optional imitation of that heuristic is a separate ablation. Short tactical tasks and reduced victory targets can help debugging, but select final models on the full target rules.

Introduce historical self-play when the learner finishes games reliably and improves against fixed opponents. Keep heuristics and older checkpoints in the opponent pool, freeze the sampled opponent for each episode and maintain a held-out cross-play matrix. Latest-policy-only play can forget older strategies or cycle. For off-policy methods, store opponent/version metadata and restrict replay age or freeze the opponent pool during a training stage. Gendre and Kaneko's Catan study found overfitting when training against a single fixed opponent. [Study](https://arxiv.org/pdf/2008.07079.pdf).

Optimize terminal wins. If sparse rewards stall learning, ablate potential-based shaping using public progress/production features:

\[
r'_t=r_t+\gamma\Phi(b_{t+1})-\Phi(b_t).
\]

Use consistent discounting and zero terminal potential; do not bootstrap off an artificial zero at a mere timeout. The invariance result requires an appropriate state formulation and discount treatment; a lossy history summary does not automatically meet all assumptions. [Shaping theorem](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

Goal conditioning is a useful second-stage interface for the LLM. Start with typed goals: settlement at an eligible vertex, city upgrade, or acquiring a specified resource bundle within a horizon. Encode target, remaining horizon and completion status. Train a shared \(Q(b,a,g)\) or goal-success head while keeping a separate game-win value. A goal that is easy to complete can still harm winning chances. Do not assume the model generalizes to arbitrary natural-language goals. [Universal value functions](https://proceedings.mlr.press/v37/schaul15.html).

Sample goals from reachable contexts and test unseen goal/location combinations. Hindsight relabeling is optional: Catan's stochastic outcomes can bias naive hindsight training, so compare it with ordinary goal sampling before relying on it. [HER](https://arxiv.org/abs/1707.01495), [USHER](https://proceedings.mlr.press/v205/schramm23a.html).

## Ablations and benchmarks

Use staged comparisons, with a fixed observation/action contract and reward. For the initial PPO–DQN comparison, retain the fixed no-trade action table and comparable encoder/head sizes; introducing structured candidate scoring is a separate representation change. Do not run a full Cartesian sweep.

| Stage | Comparison | What it answers |
| --- | --- | --- |
| Algorithm | Masked PPO versus masked Double DQN, same observation and small architecture | Stability, sample efficiency and practical evaluator quality |
| Capacity | Two-layer MLP widths \(128,256,512\) | Whewtional capacity pays for itself |
| Structure | Best MLP versus small tile/node/edge graph encoder | Whether spatial structure improves generalization at similar parameter and latency budgets |
| Information | Current observation versus public-event ledger/belief | Value of remembering legitimately observed information |
| Opponents | Fixed pool versus historical self-play | Robustness and forgetting |
| Planning | Greedy scorer versus bounded belief search | Strength gained per inference millisecond |
| Goals/trading | Goal conditioning; restricted offers; acceptance-model mixtures | Strategic benefit and behavioral failure modes |

Start graph models with a few small message-passing layers; include exact public road/connectivity summaries because shallow local propagation can miss long-road structure. Match parameters and measured inference time where possible. Evaluate browser batch size equal to a realistic candidate list, not only one action.

A future integration run can be short and single-seed, but algorithm claims require multiple independent training seeds. Use at least three for preliminary comparisons, with equal interaction budgets and separate wall-clock reporting; larger budgets follow only if learning curves justify them. Training, model-selection and final-test boards/seeds must be distinct. Run paired seat swaps and bootstrap uncertainty by board/seed pair. Repeated timing runs on identical seeds are not additional independent gameplay evidence. [Empirical RL design](https://www.jmlr.org/papers/v25/23-0183.html).

Benchmark against Random, WeightedRandom, an audited public-information heuristic, frozen learned checkpoints and eventually a belief-search agent. The installed `ValueFunctionPlayer` is a useful heuristic starting point, but its copied stochastic successor evaluation needs the same RNG/information audit. Unmodified AlphaBeta/MCTS should not serve as fair references. Evaluate each opponent separately before aggregating. Same initial seed does not ensure the same later dice: bot choices currently consume the game's RNG.

Report wins, losses and truncations with all attempted games in the denominator; also show completion rate, points, turn count, illegal actions, per-opponent results and browser median/tail latency. A stalled game is not a loss or draw unless the benchmark rules explicitly assign that payoff. With wins \(W\), truncations \(T\) and attempts \(N\), \(W/N\) is the observed win fraction; unresolved outcomes place the completed sample's eventual win fraction between \(W/N\) and \((W+T)/N\), before sampling uncertainty. Do not calibrate an uncapped win-value model against timeout-as-nonwin labels; finish those games or handle censored outcomes explicitly. For the planner, measure action ranking agreement/regret against additional rollouts and value calibration on held-out games. For trading, report accepted/rejected/repeated offers and changes in win rate. Compare policy-only with LLM-plus-tool end to end.

| Prior work | Relevant evidence | Limit |
| --- | --- | --- |
| [Gendre and Kaneko, 2020](https://arxiv.org/pdf/2008.07079.pdf) | Two-player actor-critic with spatial/scalar features and historical self-play; reported \(56.5\%\) versus JSettlers across \(10{,}000\) games | No player trades; approximately five weeks of training; different environment/information assumptions |
| [Deep Catan, 2022](https://www.lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf) | MCTS with learned evaluation and Expert Iteration; later networks improved over earlier networks | Four-player internal comparison, no player trading; no transferable one-versus-one win rate |
| [Charlesworth's Settlers-RL](https://settlers-rl.github.io/) | Four-player masked PPO, trading and hidden-state sampled search; approximately \(450\) million training decisions | Author project; persistent poor trade offers; costly search, not a browser latency result |

These support design choices, not a promise of strong play at a small training budget. More recent [BenjaminL1/catan_rl](https://github.com/BenjaminL1/catan_rl) and [AlgoCatan/RLCatan](https://github.com/AlgoCatan/RLCatan) are implementation references; their rules, checkpoints and performance claims need independent reproduction before using them as benchmarks.

## Local simulator measurements

Measured the installed Catanatron 3.3.0 package pinned to `ecf931181b9a65bb4116a2153fb78c16f1438e00`, using Python 3.14.7 on this arm64 Mac. [Raw timings](results/benchmark.json) retain environment metadata, episode results, trajectory hashes, timings and a small profile.

There were six unique seeds per player count, five timing repetitions and three variants: \(180\) short rollouts representing only \(12\) distinct trajectories, plus one profiled rollout. The final benchmark took approximately \(7.4\) seconds. All paired variants produced identical action/outcome traces and no rollout hit a cap. There was no training, neural inference, HTTP or browser execution. Timings include forced actions, both/all players, winner checks and measurement overhead; they exclude game initialization. Variant ordering was rotated.

| Configuration | Two-player engine actions/s | Four-player engine actions/s |
| --- | ---: | ---: |
| Stock WeightedRandom | \(62{,}717\) | \(44{,}417\) |
| Same policy distribution without repeated-list allocation | \(70{,}419\) | \(51{,}169\) |
| Compact weighted sampling plus numeric observation every action | \(13{,}478\) | \(8{,}745\) |

The optimized sampler makes the same integer random draw and returns the same action, without constructing a list containing thousands of repeated actions. It still allocates a small weights list despite its internal `AllocationFreeWeightedPlayer` name. It exists only inside the benchmark. Its paired trajectory checks prevent confusing a different policy with a faster implementation.

In the two-player numeric-observation condition, feature extraction took approximately \(77\%\) of measured loop time; `game.execute` took approximately \(17\%\). The corresponding RED decision rate was approximately \(6{,}655\)/s, including forced moves. Neither number is a neural-training throughput estimate.

| Sampled two-player operation | Median | Sample p95 |
| --- | ---: | ---: |
| Generate legal actions | \(6.2\,\mu s\) | \(14.0\,\mu s\) |
| Copy with independent RNG | \(39.0\,\mu s\) | \(48.7\,\mu s\) |
| Independent copy plus one action | \(47.1\,\mu s\) | \(64.8\,\mu s\) |
| Stock numeric vector minus resource-bank fields | \(57.2\,\mu s\) | \(66.8\,\mu s\) |
| Bridge snapshot including text prompt | \(322.3\,\mu s\) | \(485.5\,\mu s\) |
| Snapshot plus JSON serialization | \(416.9\,\mu s\) | \(591.5\,\mu s\) |

These kernel measurements use \(64\) saved states per player count and five repetitions each, with warm caches. They are not stratified by game phase; copy-and-execute uses the first legal action, and the synthetic bridge has empty event history. Its four-player prompt wording is not used to play the two-player game. These are component costs, not MCTS throughput or deployment latency percentiles. The numeric representation still lacks the proposed additional features.

**Do not rewrite the simulator now.** The measurements show no need for a new engine to begin training. They cannot establish the bottleneck of a learner that has not been implemented. Train directly through the in-process engine, cache static board features, use arrays instead of repeated dictionaries, avoid text/JSON in rollout collection and batch policy inference. Profile the actual learner before deciding between more worker processes, faster feature extraction or a native transition core.

If a later profile shows transition execution dominates after those changes, replace only the measured hot path and compare recorded action/outcome traces against the pinned engine. Cover card conservation, bank shortages, development-card timing, longest-road changes, discard sequences and trade responses. A full rewrite adds rules risk; it is not justified by these timings. Search helper repairs are a smaller, separate task.

## Browser inference and next implementation

Export the small policy/value or candidate-scoring network to ONNX, with a versioned feature order and identical normalization. Start with ONNX Runtime Web's WebAssembly backend in a worker; measure WebGPU as an alternative. Small networks may not amortize GPU dispatch, and the browser LLM already uses GPU resources. [Official ONNX Runtime Web guidance](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).

Benchmark realistic legal-action batches, warm/cold latency, model bytes, peak memory and numerical parity with Python. None of these neural/browser measurements were performed here. Quantize only after evaluating both ranking and game outcomes. A direct neural scorer can be hosted as static assets; browser-side search also needs the rules engine or a search service. The current Python bridge does not become browser-only merely by exporting a model. Hosting implementation is outside this RL investigation.

The next implementation should be a thin adapter exposing `observe(actor)`, `legal_actions()`, `step(action)` and `clone_for_search(seed)`, followed by a masked PPO trainer and a common evaluation harness. Validate hidden-state invariance, action-map roundtrips, terminal/truncation handling and RNG isolation before training. Add the Double DQN scorer, goals and negotiation only after that baseline is measurable. Use a separate pinned training environment; the current environment has the simulator but lacks NumPy, Gymnasium, PyTorch and ONNX.

Reproduce the completed local checks from the repository root:

```sh
PYTHONHASHSEED=0 .venv/bin/python rl_research/audit_engine.py --output rl_research/results/audit.json
PYTHONHASHSEED=0 .venv/bin/python rl_research/benchmark.py --output rl_research/results/benchmark.json
```

Only the research scripts and results were added. The demo, installed engine and existing experiment results were not changed.

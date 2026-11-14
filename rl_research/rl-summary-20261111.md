# TinySettler RL implementation and experimental conclusions

PPO-128 is a working compact baseline, but it is not yet a strong Catan player. It beats weak bots, loses almost every game against the adapted Catanatron value-function heuristics, and sometimes misses legal winning or progress-making actions under greedy inference. Doubling network width did not establish an advantage. The tested midgame/endgame curriculum increased reward frequency without demonstrating improved fresh full-game performance.

The next useful work is a fresh greedy-versus-sampling evaluation, opening/economy diagnosis, and a controlled test of learning from the stronger heuristic. Historical-policy self-play is worth comparing after establishing these controls. Keep Catanatron and PPO-128 while improving the training signal and evaluation.

## What is implemented

The environment is two-player BASE Catan to \(10\) victory points, with a discard threshold of \(7\), bank/port trades, ordinary robber behavior, and administrative limits of \(1000\) turns or \(20000\) engine actions. There are no domestic trade offers or natural-language negotiations. This is the experimental two-player variant, not the official special CATAN for Two rules.

The simulator is pinned Catanatron revision `ecf931181b9a65bb4116a2153fb78c16f1438e00`. The RL path directly calls the engine. It does not use the browser, HTTP, or an LLM during training.

### Board and player representation

The policy receives an actor-relative float32 observation \(o\in\mathbb{R}^{2049}\). “Self” and “opponent” retain their meanings when seats change. The board has \(19\) land tiles, \(54\) intersections and \(72\) edges.

| Input block | Features | Contents |
| --- | ---: | --- |
| Existing board/player features | \(608\) | Road/building occupancy, tile resources and production, ports, own cards and statistics, opponent public totals/statistics, phase flags |
| Exact dice-token identities | \(247=19\times13\) | One-hot identities distinguish equal-probability tokens such as \(6\) and \(8\) |
| BASE topology | \(1170=54\times19+72\times2\) | Node–tile incidence and edge endpoint indices |
| Current prompt | \(7\) | Setup, play and forced-action context |
| Global/phase/rule fields | \(11\) | Acting player, turn owner, setup/discard/knight/free-road/trade flags, free roads, turn counter and rule constants |
| Pending discards | \(2\) | Actor-relative counts |
| Own development-card age | \(4\) | Whether each playable development-card type was owned at the start of the turn |

The \(608\)-feature block consists of \(144\) road, \(216\) building, \(152\) tile, \(54\) port, \(26\) own-player, \(14\) opponent and \(2\) phase features. All stock `BANK_` features are removed. In particular, exact resource-bank counts would reveal the opponent's hand in two-player play through

\[
H_{\mathrm{opponent},r}=19-B_r-H_{\mathrm{self},r}.
\]

The policy sees its own cards, but not the opponent's hidden resource composition, hidden development-card identities/VP, deck order or RNG state. It has no recurrent memory or public-event belief tracker. The stronger heuristic adapter does maintain public-history resource bounds; the PPO model does not receive those bounds.

The topology block is constant for this BASE encoding, approximately \(57\%\) of the input. Inputs are otherwise mostly unnormalized: edge endpoints are divided by \(53\), and turns by \(1000\). This is a flat feature model, not a graph network. [Observation implementation](overnight/env.py)

### Actions and legality

There are \(332\) global discrete action IDs. This is the action vocabulary, not the number of legal choices in every state or the number of game states.

| Action family | Entries |
| --- | ---: |
| Road / settlement / city placements | \(72+54+54\) |
| Bank/port trade combinations | \(60\) |
| Robber destinations/victims | \(57\) |
| Year of Plenty / Monopoly / discard choices | \(20+5+5\) |
| Roll / end turn / buy development / play knight / play Road Building | \(5\) |

The engine supplies a Boolean legal-action mask. Illegal logits are suppressed before sampling or argmax; rollout masks are reused during PPO optimization. Robber victims are encoded relative to the actor. Execution checks legality again. Development cards that require additional choices lead to subsequent decision states. Domestic trade offers are absent from this vocabulary. This is the standard policy-gradient masking approach described by [Huang and Ontañón](https://arxiv.org/abs/2006.14171).

### Network and training

The current network is

\[
2049\rightarrow128\;\mathrm{Tanh}\rightarrow128\;\mathrm{Tanh}
\rightarrow\{332\text{ policy logits},\;1\text{ state value}\}.
\]

The actor and critic share both hidden layers. PPO-128 has \(321869\) parameters; PPO-256 has \(676173\). There is no goal input, graph message passing, action-descriptor encoder or explicit opponent identity. The legal mask changes the action distribution but is not an input to the shared representation or value head.

| PPO setting | Implemented value |
| --- | --- |
| Learner decisions per rollout | \(4096\) |
| Update epochs / minibatch size | \(4\;/\;256\) |
| Optimizer / learning rate | Adam, \(3\times10^{-4}\), constant |
| Policy clipping / GAE | \(0.2\;/\;\lambda=0.95\) |
| Discount / reward | \(\gamma=1\); terminal win \(+1\), loss \(-1\), otherwise \(0\) |
| Entropy / value-loss coefficients | \(0.01\;/\;0.5\) |
| Gradient norm clipping | \(0.5\) |

Training samples from the masked categorical policy. Reported full-game evaluation uses masked argmax. One learner transition includes its chosen action plus any intervening opponent/chance actions until that same player next decides. True terminal states do not bootstrap; administrative truncations bootstrap the last state value but stop GAE chaining into the reset episode. Advantages are normalized per rollout. There is no reward shaping, learning-rate schedule, value clipping or KL-based early stopping. [Trainer](overnight/train.py), [configuration](overnight/config.json), [PPO paper](https://arxiv.org/abs/1707.06347).

All completed experiments used fixed opponents, not self-play. The original mixture was equal Random, WeightedRandom and a simple priority heuristic. The latest run used \(50\%\) adapted ValueFunction-base, \(25\%\) WeightedRandom and \(25\%\) priority heuristic, sampled per episode. The priority heuristic never voluntarily bank-trades. The adapted base/contender bots retain upstream heuristic weights, use two common sampled hidden worlds per decision, float64 scoring and independent planning RNG. Their resource beliefs use permitted public history. They are two configurations of one heuristic family. No AlphaBeta opponent has yet been evaluated.

## What the experiments establish

Three training studies retained \(6266880\) learner transitions across twelve from-scratch runs and six continuations. Outcome rates below have different protocols and must not be interpreted as one learning curve.

| Study | Design and result | Supported conclusion |
| --- | --- | --- |
| Initial algorithm/width ablation | Two seeds each; PPO-128 \(83.33\%\), PPO-256 \(82.25\%\), Double DQN-256 \(55.92\%\), using validation-selected checkpoints against three weak opponents | PPO is the better tested configuration; neither a universal PPO-versus-DQN result nor evidence that width \(128\) is optimal |
| PPO reliability | Five seeds per width, common \(434176\) transitions; PPO-128 \(80.47\%\), PPO-256 \(79.53\%\) | No demonstrated width advantage: paired seed/board interval for the difference is \([-5.53,+7.27]\) percentage points; substantial seed variation remains |
| Stronger opponents and reset curriculum | Three paired PPO-128 parents, each continued for \(176128\) transitions; standard \(52.50\%\), curriculum \(51.33\%\) over five opponents | The tested curriculum did not demonstrate transfer to fresh full games; both arms remain weak against the stronger bots |

The first two studies used \(434176\) transitions per newly trained policy. Latest policies therefore have \(610304\) transitions along their training lineage. Double DQN used a masked double target, uniform replay and an epsilon schedule, but approximately sixteen times as many optimizer steps as PPO per collected transition. Its poorer outcome is specific to that replay/update configuration. The initial validation set was also small; the later reliability study explicitly addressed selection uncertainty.

Sources: [initial ablation](results/overnight-20261111/findings.md), [reliability study](results/ppo-reliability-20261111/findings.md), [curriculum study](results/ppo-curriculum-20261111/findings.md).

### Latest opponent results

Each cell contains \(120\) games across three parent policies, \(20\) boards and both seats. Timeouts stay in the denominator.

| Opponent | Standard continuation | Curriculum |
| --- | ---: | ---: |
| Adapted ValueFunction-base | \(2.50\%\) | \(2.50\%\) |
| Adapted contender | \(5.00\%\) | \(3.33\%\) |
| Random | \(91.67\%\) | \(91.67\%\) |
| WeightedRandom | \(95.83\%\) | \(88.33\%\) |
| Legacy heuristic | \(67.50\%\) | \(70.83\%\) |

Standard won \(315/600\), curriculum \(308/600\). The curriculum-minus-standard difference is \(-1.17\) percentage points, with paired-policy/board interval \([-7.50,+5.67]\). This is an inconclusive difference, not proof of equivalence. There were one and three timeouts respectively, and zero illegal actions. [Raw outcomes](results/ppo-curriculum-20261111/full-game-outcomes.csv), [intervals](results/ppo-curriculum-20261111/paired-comparisons.json).

The strong-opponent average is only \(3.75\%\) standard and \(2.92\%\) curriculum. The weak-opponent averages are \(85.00\%\) and \(83.61\%\). The drop from the earlier approximately \(80\%\) overall score to approximately \(52\%\) largely reflects adding much stronger opponents; it is not a matched regression estimate. The unchanged parent checkpoints were not evaluated on this new suite, so the causal benefit of stronger-opponent continuation itself is still unknown.

### Curriculum and failure diagnosis

Both arms started from identical weights and Adam state for each parent. Standard always reset to a fresh game; curriculum reset to \(50\%\) fresh, \(25\%\) midgame with \(4\)–\(6\) own VP, and \(25\%\) endgame with \(7\)–\(9\) own VP. The corpus had \(307\) training snapshots, split from validation/test by source game. Reset states retained history, development-card age, physical cards and the remaining episode horizon.

These are reset proportions, not decision proportions. Curriculum's actual transitions were approximately \(70.4\%\) fresh, \(20.2\%\) midgame and \(9.4\%\) endgame. Terminal-reward frequency increased from \(0.552\%\) to \(0.770\%\), approximately \(1.40\times\), without demonstrated strength improvement.

Against the strong family, \(175/240\) standard games and \(178/240\) curriculum games built no settlement beyond setup; \(181/240\) and \(182/240\) built no city. Mean final points were only \(3.61\) and \(4.09\). This motivates opening, production and expansion diagnostics; it does not prove which decisions caused the losses. During stochastic training, fresh-start win rates against base were also only \(34/1459=2.33\%\) and \(26/1037=2.51\%\). Those changing-policy training figures are not final-policy tests, but they show that greedy inference alone does not explain the strong-opponent gap.

The held-out exercises contained eleven task cases from only eight distinct source games. Productive-trade success rose from \(6/18\) to \(9/18\), with an extremely wide interval; expansion stayed \(17/18\), and immediate first-action wins stayed \(6/18\). No robust task advantage was established. High eventual win rates on an exercise do not establish immediate tactical competence. [Exercise outcomes](results/ppo-curriculum-20261111/exercise-outcomes.csv).

One historical development-card stall was exactly reconstructed. Knights were legal at \(1097\) decisions; at one inspected state the policy preferred rolling with probability \(55.51\%\) over a legal knight with \(24.14\%\). Across deliberately selected historical failure positions, greedy play won \(9/56\) continuations and sampling won \(55/56\). The exact reconstruction alone gave greedy \(24/24\) timeouts versus sampled \(24/24\) wins. This is a strong diagnostic of those selected failures, not a general sampling win rate. [Historical diagnostics](results/ppo-curriculum-20261111/historical-diagnostics.json).

## Most promising next work

| Priority | Experiment or development | Why it is useful now |
| --- | --- | --- |
| First | Fresh full-game greedy versus categorical-sampling evaluation, with frozen parents as references | Cheaply separates inference failures from training limitations and supplies the missing before/after control |
| Next | Opening/economy diagnostics, followed by heuristic imitation warm start plus PPO versus matched PPO alone | The teacher already wins decisively; the learner rarely expands or builds cities against it |
| Next | Input-scale and critic diagnostics, then one compact representation or PPO-stability ablation | Width has not solved the problem; current Tanh inputs are mostly unscaled and training logs contain only a combined loss |
| Then | Historical-policy self-play mixed with fixed heuristic anchors | Tests broader opponent coverage while preserving an external strength yardstick |
| Product development | Calibrated action evaluator and browser export | Needed for the original LLM-tool design; current logits are not action values |

For the teacher experiment, collect legal, information-safe demonstrations for initial placements and early economy, then include teacher labels on learner-visited states if simple cloning fails. Compare the same PPO-128 architecture, downstream training budget and evaluation suite; account for teacher-generation time separately. The motivation for labeling learner-visited states is the distribution-shift problem addressed by [DAgger](https://proceedings.mlr.press/v15/ross11a.html). This is a proposed experiment, not a demonstrated improvement here.

Add policy entropy by phase/action family, approximate KL, clipping fraction, critic explained variance and value error by stage/opponent, gradient norms, input scales and Tanh activation saturation. Log production/resource coverage after setup, opportunity to expand, time to third settlement/first city, and opponent/reset shares measured in learner transitions. Then test one change at a time: normalized counts and explicit production/purchase/expansion features; or a learning-rate schedule; or \(\lambda=0.99\) versus \(0.95\). Larger GAE traces change the bias/variance tradeoff, not necessarily performance; bootstrapped values still carry longer-horizon information. [GAE](https://arxiv.org/abs/1506.02438).

For self-play, compare the fixed-pool reference with a predeclared mixture such as \(50\%\) fixed heuristic pool and \(50\%\) frozen historical PPO opponents. Sample several snapshots and seeds; add new snapshots only at fixed intervals and retain old ones. Keep training and held-out evaluation pools distinct. Competitive training can supply a curriculum, but this remains untested in this project and does not guarantee escape from shared weak strategies. [Self-play precedent](https://arxiv.org/abs/1710.03748).

A verified deterministic immediate-win override is also a useful demo safeguard: use only legal actions and permitted information, and report it as a hybrid policy separately from raw PPO. Do not add a blanket knight-playing rule from selected failures. Goal-conditioned training and a graph encoder remain later ablations; the tested reset curriculum did not condition the network on goals. A goal input and goal-specific value targets would be a different method. [Goal-conditioned value precedent](https://proceedings.mlr.press/v37/schaul15.html).

### Efficiency and the LLM tool

The latest path was CPU dominated: approximately one core occupied, \(6.4\%\) average GPU utilization, and \(173\)–\(215\) learner decisions/s. Full-game evaluation took \(714.6\) seconds for \(1200\) games. Measured policy inference was approximately \(0.43\) ms in Colab, not browser latency. Profile observation creation, opponent planning and batching before rewriting the simulator. The original approximately \(13478\) engine-actions/s benchmark excluded neural training and strong-opponent planning, so it is not comparable training throughput.

For fixed BASE topology, fold its constant input contribution into the first-layer bias before deployment:

\[
W_dx_d+W_cx_c+b=W_dx_d+\widetilde b,
\qquad\widetilde b=b+W_cx_c.
\]

Removing those \(1170\) input columns reduces PPO-128 to \(172109\) parameters without changing the mathematical function. Verify numerical action/logit equivalence after export. This is an efficiency improvement, not evidence of better strategy. Measure CPU versus batched GPU inference and a small number of environment workers before scaling compute.

The current outputs are \(\pi(a\mid o)\) and \(V(o)\), not \(Q(o,a)\). The scalar critic is uncalibrated and reflects its training opponent mixture. To make an LLM planning tool, first evaluate candidate actions using information-consistent simulated continuations, then consider distilling a candidate-conditioned value head with a defined continuation policy/opponent distribution. Validate ranking and calibration on held-out actions and eventual outcomes. Do not label softmax probabilities as win probabilities. Negotiation and the combined LLM-plus-tool policy require separate evaluation once implemented.

## Integrity and immediate evaluation plan

All six continuation policies and all \(1332\) primary/exercise games were retrieved. Source/configuration hashes, paired initial weights/optimizer state and checkpoint recovery checks passed. The current adapter contains the constructor and tile-comparison fixes; its hash matches the executed preflight and frozen manifest. Earlier local compatibility warnings are superseded. The experiment used \(2\) h \(12\) m \(50\) s of allocated-session upper bounds, or \(2\) h \(26\) m \(15\) s on the conservative original clock including downtime; the original four-hour deadline was preserved. [Completion checks](results/ppo-curriculum-20261111/completion-verification.json), [resource accounting](results/ppo-curriculum-20261111/resource-summary.json).

The short next evaluation is specified in [the strong-opponent evaluation plan](short-strong-opponent-evaluation.md). It uses the current standard checkpoints, unchanged parent references, fresh boards, greedy and sampled inference, the two audited value heuristics, and a separately validated bounded search opponent. It launches no training and is capped at one hour of allocated compute.

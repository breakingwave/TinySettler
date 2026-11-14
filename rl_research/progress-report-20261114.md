# TinySettler reinforcement learning progress report

Status as of November 14, 2026. Scope: reinforcement learning and readiness for the LLM/browser demo.

**A reproducible learning system is working, but strong Catan play has not been achieved.** Masked PPO learns to beat weak bots. The tested larger networks, late-game reset curriculum and refreshed self-play have not established reliable improvements against stronger opponents. The latest validation-selected policy won $18/400=4.5\%$ of its confirmation games against the value heuristics.

Across four studies, we completed 22 fresh training runs and six continuations, collecting $10{,}526{,}720$ retained learner transitions. This excludes pilots and retried work and counts reused historical runs only once. It is aggregate effort across models, not the experience of one agent.

## Environment and information

Training uses PyTorch on Google Colab T4 GPUs through the Colab CLI, with CPU simulation and opponent decisions. Catanatron 3.3.0 is pinned to revision `ecf931181b9a65bb4116a2153fb78c16f1438e00` and called directly from Python. Games use the BASE board, two players, a target of ten victory points, discard threshold seven, ordinary robber behavior and bank/port trading. Administrative limits are $1000$ turns or $20000$ engine actions. This is our two-player experimental variant, not the special official CATAN for Two rules. Player-to-player offers and negotiation are absent.

The observation is **actor-relative**: “self” always means the player making the decision, regardless of seat. It contains:

| Component | Information supplied |
| --- | --- |
| Board | Resource and exact dice-token identities for 19 tiles; roads on 72 edges; settlements/cities on 54 intersections; ports and robber location |
| Own player | Exact resource and development cards, development-card availability/age, points, pieces and awards |
| Opponent | Public points, total hand sizes, played cards, pieces and public board position |
| Context | Current decision phase, turn ownership, pending discards, free roads, turn count and rule constants |

The original encoder produced $2049$ numbers, including $1170$ constant topology features. Scaling removed that constant block and applied fixed feature scaling, giving **879 inputs without dropping changing game information**. Normalization's effect on strength was not separately tested.

Exact bank counts are excluded because they can reveal the opponent's resource hand. Hidden opponent cards/points, deck order and simulator random-number state are also excluded. PPO is memoryless: it has no recurrent network, event history or belief state. The stronger heuristic tracks public history without receiving true hidden cards. [Encoder and schema](scaling/sc_models.py)

## Actions and learning

The **332-entry discrete action space** covers building locations, bank/port trades, robber and development-card choices, discards, rolling and ending the turn. Only a subset is legal in each position. Catanatron's legal-action mask excludes illegal choices during training and inference; execution checks again. Recorded evaluations had zero illegal actions.

The main algorithm is **Proximal Policy Optimization (PPO)**, an actor–critic method: the actor selects actions and the critic estimates future reward. Both share two fully connected layers with tanh activations. Their outputs are 332 action scores and one state value.

| Input representation | Hidden layers | Parameters |
| --- | --- | ---: |
| Original | $128\times128$ | $321{,}869$ |
| Original | $256\times256$ | $676{,}173$ |
| Compact normalized | $128\times128$ | $172{,}109$ |
| Compact normalized | $512\times512$ | $884{,}045$ |

Rewards are sparse: $+1$ for winning, $-1$ for losing and zero otherwise, including building and trading. One transition spans two consecutive learner decisions, including intervening opponent/chance actions. For administrative timeouts, training estimates unfinished returns using the critic, called **bootstrapping**. True endings do not bootstrap, and targets never connect across reset games.

PPO uses $4096$-transition rollouts, four epochs, minibatches of $256$, Adam learning rate $3\times10^{-4}$, clipping $0.2$, discount $\gamma=1$, generalized advantage estimation (GAE) with $\lambda=0.95$, and entropy coefficient $0.01$. These settings stayed fixed. Training samples legal actions. The first three studies evaluated primarily with greedy selection; scaling made sampling primary and also tested greedy selection. [Trainer](scaling/sc_train.py)

**Double DQN**, which learns action values using replay and a target network, was also tested. Its implementation used roughly sixteen times as many optimizer steps per collected transition as PPO. Its results therefore compare particular configurations, not the algorithms under every reasonable training schedule.

## Opponents and evaluation

Initial opponents were Random, WeightedRandom and a simple priority heuristic that never voluntarily bank-trades. Later studies introduced adapted Catanatron **ValueFunction base and contender**: two settings of one heuristic family, not independent algorithms or alpha-beta search.

The adapter evaluates actions using two sampled hidden worlds consistent with permitted information, public resource bounds and separate planning randomness. This is an approximate belief model. It avoids directly reading the opponent's true hidden hand.

Tests use held-out boards, both seats and repeated training with different random seeds. Timeouts remain in win-rate denominators. The paired intervals below are exploratory $95\%$ bootstrap intervals accounting for board and training-seed variation; board-only intervals can be misleadingly narrow. Checkpoint choices are frozen before final testing. **Different opponent panels, encoders and inference modes prevent treating study averages as one learning curve.**

## Completed experiments

| Study | What was compared | Result and interpretation |
| --- | --- | --- |
| [Initial overnight ablation](results/overnight-20261111/findings.md) | PPO-128, PPO-256 and Double DQN-256; two seeds each; $434{,}176$ transitions per run | Selected checkpoints won $83.33\%$, $82.25\%$ and $55.92\%$ against the weak pool. PPO was the better tested configuration. Width was inconclusive; DQN took about $1.56\times$ longer. |
| [PPO reliability](results/ppo-reliability-20261111/findings.md) | Six additional runs, combined with four historical PPO runs to give five seeds per width | At the common final step, PPO-128 won $80.47\%$ versus PPO-256's $79.53\%$. The 128-minus-256 difference was $+0.93$ points, interval $[-5.53,+7.27]$. Seed variation was substantial; the earlier larger model's lower stall count did not persist. |
| [Midgame/endgame curriculum](results/ppo-curriculum-20261111/findings.md) | Three paired PPO-128 parents, each continued for $176{,}128$ steps with matched weights and optimizer state | Ordinary training won $52.50\%$ versus curriculum's $51.33\%$ over five opponents. Curriculum-minus-ordinary was $-1.17$ points, interval $[-7.50,+5.67]$. Against the strong family alone: $3.75\%$ versus $2.92\%$. No demonstrated full-game benefit. |
| [Scaling and self-play](results/scaling-20261114-recovered/findings.md) | Ten fresh compact-model runs at $425{,}984$ steps each; width 128/512 crossed with frozen/refreshed opponent pools | No reliable refresh benefit or established capacity winner. Static width 512 produced a promising but uncertain signal. Graph and teacher conditions were omitted before outcomes under the throughput rule. |

The curriculum reset to fresh, midgame and endgame positions with probabilities $50/25/25\%$, using reachable snapshots split by source game. Terminal rewards became approximately $1.4\times$ more frequent without demonstrated fresh-game improvement. This was a **reset curriculum**, not a policy conditioned on an explicit goal. Small trading, expansion, army and immediate-win exercises also showed no robust advantage.

The low later scores are not evidence that performance collapsed from $80\%$: stronger opponents changed the benchmark. The curriculum study also lacked a matched evaluation of unchanged parents on its new panel, so it did not isolate the benefit of introducing stronger opponents.

## What the self-play study established

Both schedules mixed heuristic opponents with historical neural policies. Half the initial game-start probability went to neural opponents; the rest was base $25\%$, WeightedRandom $12.5\%$ and the simple heuristic $12.5\%$. Refreshed runs added their own frozen snapshots every $65{,}536$ transitions. Opponents stayed fixed within each game. This tests **refreshing a neural opponent pool**, not self-play versus heuristic-only training.

| Compact model and pool | Training seeds | Strong-family wins | Held-out PPO wins |
| --- | ---: | ---: | ---: |
| MLP-128, frozen | 3 | $5/240=2.08\%$ | $53.33\%$ |
| MLP-128, refreshed | 3 | $7/240=2.92\%$ | $58.33\%$ |
| MLP-512, frozen | 2 | $7/160=4.38\%$ | $62.50\%$ |
| MLP-512, refreshed | 2 | $5/160=3.12\%$ | $56.25\%$ |

Refreshing MLP-128 improved the strong-family point estimate by $0.83$ percentage points, with a seed/board interval of $[-3.75,+5.83]$. Static MLP-512 improved both matched seed pairs, averaging $3.75$ points, but its interval $[-1.25,+8.75]$ includes zero. Capacity contrasts use the common two seeds, not subtraction of the full table averages.

The selected MLP-128 refreshed policy's $4.5\%$ confirmation rate had a board-cluster $95\%$ interval of $[2.25\%,7.0\%]$; its $12.5\%$ validation score came from only 32 games. This measures one selected model, not average self-play benefit. Scaling validation/final games had no timeouts or illegal actions.

## Strategic failures and useful diagnostics

**The main losses happen before substantial expansion.** In sampled games against strong heuristics, $66$–$81\%$ ended without a city and $78$–$89\%$ without a third settlement. Possible causes include placement, resource allocation, blocked expansion or early opponent wins. Trade and knight counts alone do not establish good strategy.

**Greedy selection explains some stalls, not the overall strength gap.** In deliberately selected historical failure positions, sampling won $55/56$ continuations versus greedy's $9/56$. The later untouched full-game comparison did not establish a universal sampling advantage. Strong-opponent losses also occurred during sampled training.

**Opponent shares differ from learning exposure.** Base received a quarter of game starts but only $9.6$–$11.9\%$ of transitions because its games ended faster. Own snapshots consumed roughly $36$–$40\%$ in refreshed runs. This motivates an exposure experiment without establishing causation.

**Critic fit does not establish planning quality.** Explained variance reached $0.85$–$0.97$ against bootstrapped GAE targets, not independent eventual-win labels. Neither win calibration nor candidate-action ranking was validated. Noisy validation and few seeds also prevent claims of a clear plateau or overtraining.

## Efficiency and experiment reliability

The simulator audit found hidden-hand leakage through bank counts, shared randomness in stock game copies, and problems in stock chance/search helpers. These motivated the information boundary and independent planning randomness. No repaired alpha-beta/MCTS benchmark was validated, so search was not evaluated. Strong heuristic results must not be presented as search-bot results.

The original lightweight engine/observation benchmark reached about $13{,}478$ actions/s. The latest complete training path reached $171$–$174$ learner decisions/s; these are different workloads. GPU utilization averaged about $4\%$, and PPO updates consumed only about $2\%$ of run time. Larger MLPs cost almost no additional elapsed time. The bottleneck is the CPU-side pipeline; this does not identify engine execution alone as the cause. A simulator rewrite or larger GPU is not currently justified.

The graph/candidate model was implemented and passed preflight, but collected only about $49$–$55$ decisions/s versus MLP pilots' $164$–$178$. It was excluded before strength testing. The 14-hour study finished within $11.25$ hours including recovery, with approximately 41 training minutes per model. It did **not** test training one model for 14 hours.

Frozen configurations, checkpoint hashes, full training-state recovery, external backups and runtime deadlines worked. A graph recovery failure exposed a CPU initializer that reseeded CUDA; it was fixed and recovery reverified. All retained runs completed and dedicated allocations were stopped.

## Most useful next work

Next, separate **teacher initialization** from **strong-opponent exposure**. The prepared teacher corpus has 200 games, including 160 training games and $9635$ training decisions, covering setup, early and later play. First check imitation accuracy and standalone strength: some teacher decisions depend on public history absent from PPO's observation.

Compare four PPO-128 conditions: ordinary versus teacher-initialized weights, crossed with existing versus increased strong-opponent exposure. Keep the archive/refresh rule fixed. Use three paired seeds, common budgets, complete games, unchanged rewards and fresh test boards. Track actual exposure and expansion milestones alongside wins. Neither intervention's benefit is established.

Replicate MLP-512 if capacity remains a promising signal, and test longer per-policy training directly. Defer another broad architecture sweep. Goal-conditioned RL, recurrent/belief-based policies, teacher-initialized strength, graph strength and validated search remain open experiments.

For the intended LLM tool, develop and validate a candidate-action evaluator with defined continuation policies and opponents; do not expose PPO logits as win probabilities. Browser export, numerical parity, latency, negotiation and the combined LLM–RL policy still need implementation and evaluation. The existing browser bridge uses four players and LLM inference; it is not the tested two-player PPO agent or a permanently hosted RL demo.

## Reports and checkpoints

The linked study reports preserve raw results, charts, configurations and qualifications. The [research index](README.md) and [scaling implementation](scaling/README.md) provide the technical details. Trained policies, full recovery checkpoints, historical models and checksums are published in the [GitHub release](https://github.com/breakingwave/TinySettler/releases/tag/rl-scaling-20261114); model archives are downloaded separately from the source repository.

# TinySettler overnight RL ablation — 2026-11-11

**Masked PPO merits the next training budget. The capacity comparison is inconclusive.** PPO-128 is a compact baseline; PPO-256 had similar wins and fewer truncations. Double DQN learned, but this fixed configuration was slower, weaker and more sensitive to checkpoint selection.

All **six runs completed 434,176 learner transitions**, with two independent training seeds per configuration: **2,605,056 sweep transitions** in **139.1 training minutes**. The optional graph model was omitted before allocation. Final evaluation used **100 held-out board seeds per opponent with swapped seats**, giving **600 games per selected policy / 3,600 total**. There were **3,543 completed games, 57 administrative truncations and zero illegal actions**. No training run hit its 45-minute limit.

Total allocated runtime was conservatively **3 h 10 m 39 s**, including the first session paused at the user's request, below the eight-hour cap. Both T4 sessions were released; the final server listing reported no active sessions. Local deadline watchdogs were also cleaned up.

| Configuration | Test wins / losses / truncations | Win rate, 95% board-cluster interval | Mean points | Training minutes/run |
| --- | ---: | ---: | ---: | ---: |
| PPO, 128×128 | 1,000 / 173 / 27 | 83.33% [80.75, 85.75] | 9.42 | 19.56 |
| PPO, 256×256 | 987 / 208 / 5 | 82.25% [79.50, 84.92] | 9.26 | 19.53 |
| Double DQN, 256×256 | 671 / 504 / 25 | 55.92% [52.17, 59.59] | 7.76 | 30.48 |

Rates include every attempted game, including truncations. Intervals resample whole board clusters, retaining both seats, opponents and both trained seeds. They measure test-board uncertainty **conditional on these two seeds**, not population uncertainty over training seeds.

Individual seed win rates were **91.50% / 75.17%** for PPO-128, **87.00% / 77.50%** for PPO-256, and **68.83% / 43.00%** for Double DQN. PPO-128's paired advantage over PPO-256 was only **1.08 percentage points**, with interval **[−1.58, +3.92]**. There is no demonstrated win-rate benefit from doubling width. PPO-256's lower truncation count deserves follow-up before choosing a deployment architecture.

| Fixed opponent | PPO-128 | PPO-256 | Double DQN-256 |
| --- | ---: | ---: | ---: |
| Random | 92.25% | 88.75% | 62.00% |
| WeightedRandom | 88.75% | 88.25% | 53.25% |
| Audited public-information heuristic | 69.00% | 69.75% | 52.50% |

Each entry uses 400 games. Per-opponent counts and intervals are in [aggregate.json](aggregate.json); individual-seed results are in [opponent-comparison.csv](opponent-comparison.csv).

Sparse rewards did permit learning. PPO validation improved substantially from initialization, but longer training was not uniformly better. Both PPO-128 seeds selected earlier checkpoints. The first DQN seed's validation wins moved **4 → 8 → 1 → 6 out of 12**, and its midpoint was selected. The second selected its first-quarter checkpoint. DQN used about **1.56×** the training time at the same transition budget. Its replay/update schedule performs substantially more optimizer updates than PPO; this is evidence about the tested implementations/settings, not an inherent algorithm ranking.

Prioritize more PPO training seeds and a larger validation set before simply extending these runs. Retain PPO-128 as the compact baseline and PPO-256 as a completion-rate comparison. For direct action values, improve and retest DQN's replay/update settings and value calibration before committing a longer run. Rewards, opponents and budgets were held fixed throughout this comparison.

| Configuration | Parameters | Learner decisions/s | Engine actions/s | Colab inference, mean ms | Terminal rewards/transition |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO-128 | 321,869 | 370.0 | 734.8 | 0.403 | 0.348% |
| PPO-256 | 676,173 | 370.6 | 731.9 | 0.409 | 0.347% |
| Double DQN-256 | 676,173* | 237.4 | 499.7 | 0.407 | 0.233% |

*DQN's shared architecture includes an unused 257-parameter value head. Throughput covers collection and learning, excluding evaluation. Inference measures the network and masked action extraction inside Colab, excluding observation encoding and host-to-device transfer. **These are not browser latency measurements. PPO probabilities are not Q-values or calibrated win probabilities; DQN calibration was not evaluated.**

The runtime exposed two CPUs. Whole-session GPU sampling averaged **14.3% utilization**, with **24% p95** and **439 MiB peak GPU memory**. Later direct CPU sampling averaged **96.9% of one core** for the trainer, with **3.91 GiB peak RSS**; CPU-percent sampling began during the second PPO-128 run. The two-worker engine probe was about 20% faster, but the validated single-worker training path retained a CPU for artifact work. Batched collection/inference is a useful next efficiency experiment. See [utilization.json](utilization.json).

Validation used only **two boards per opponent**, so checkpoint selection is noisy. Final-test seeds were reserved until all selections were committed. These results cover two-player BASE, target 10, discard 7, bank/port trading and the fixed equal opponent mixture. They do not establish performance with negotiation, stronger opponents, self-play, graph encoders or browser deployment. No history/belief model was included.

The recoverability checks passed: real T4/CUDA updates, privacy and action-mapping tests, ten final local tests, **30 retrieved checkpoint integrity checks**, and ten selected/final checkpoint recovery checks. External PPO and DQN checkpoints were uploaded back to Colab; DQN reproduced a bitwise-identical resumed CUDA update with replay, target network and exploration state intact. One auxiliary uncompressed upload failed with TLS EOF and succeeded on the single compressed retry. A local controller received an unexplained SIGTERM; a detached controller reattached to the same still-running runtime without restarting training. These events and the requested pause are recorded in [operational-events.json](operational-events.json). There were no incomplete sweep runs.

- [Launch, resume, evaluation and reporting commands](../../overnight/README.md), [locked dependencies](../../overnight/requirements-lock.txt), [configuration](../../overnight/config.json), [implementation details](implementation-details.json).
- [Validation learning curves](learning-curves.svg), [training learning curves](training-curves.svg), [wall-clock curves](wall-clock-curves.svg), [six-run comparison](comparison.csv), [raw final games](final-games.csv), [paired comparisons](paired-comparisons.json).
- [Artifact hashes and exact executed source provenance](artifact-manifest.json), [checkpoint recovery evidence](recovery-checks.json), [local cleanup evidence](local-cleanup.json).
- [Six selected resumable checkpoints plus exact executed source, 79.2 MB](/root/tinysettler-artifacts/tinysettler-ablation-20261111-b/selected-checkpoints.tar.gz).

All common checkpoints, full optimizer/replay/RNG state and large raw training metrics remain outside the synced workspace at `/root/tinysettler-artifacts/tinysettler-ablation-20261111-b/`. The selected archive verifies every checkpoint hash after decompression. The browser demo and historical results were preserved.

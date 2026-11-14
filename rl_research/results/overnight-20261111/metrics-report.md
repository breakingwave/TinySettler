# Overnight ablation findings

Session: `tinysettler-ablation-20261111-b`. Persistent artifacts: `/root/tinysettler-artifacts/tinysettler-ablation-20261111-b`.
Controller elapsed: 183.6 minutes. Stop succeeded: True.

Outcome: **complete**.

| run | transitions | complete | selected_transitions | train_seconds | decisions_s | training_episodes | training_wins | training_losses | training_truncations | terminal_training_episodes | terminal_reward_frequency | test_games | wins | losses | truncations | mean_points |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ddqn-256-104729 | 434176 | True | 217088 | 1816.58 | 239.01 | 1019 | 558 | 452 | 9 | 1010 | 0.002326245577830189 | 600 | 413 | 183 | 4 | 8.55 |
| ddqn-256-130363 | 434176 | True | 106496 | 1841.53 | 235.77 | 1033 | 537 | 480 | 16 | 1017 | 0.0023423680719339623 | 600 | 258 | 321 | 21 | 6.97 |
| ppo-128-104729 | 434176 | True | 323584 | 1170.43 | 370.95 | 1559 | 1298 | 259 | 2 | 1557 | 0.0035861033313679246 | 600 | 549 | 45 | 6 | 9.78 |
| ppo-128-130363 | 434176 | True | 106496 | 1176.51 | 369.04 | 1465 | 1206 | 257 | 2 | 1463 | 0.0033696012676886792 | 600 | 451 | 128 | 21 | 9.05 |
| ppo-256-104729 | 434176 | True | 434176 | 1175.79 | 369.26 | 1582 | 1354 | 226 | 2 | 1580 | 0.0036390772405660377 | 600 | 522 | 76 | 2 | 9.55 |
| ppo-256-130363 | 434176 | True | 434176 | 1167.51 | 371.88 | 1436 | 1202 | 227 | 7 | 1429 | 0.0032912920106132077 | 600 | 465 | 132 | 3 | 8.97 |

Two training seeds are preliminary evidence. Validation has only two board clusters per opponent. Colab inference timing is not browser latency. PPO probabilities are not Q-values or calibrated win probabilities.

# PPO reliability results

Primary endpoint: 50 fresh boards × both seats × three fixed opponents, at 434,176 transitions. Paired training seeds: 5.

| Width | Wins | Losses | Truncations | Win fraction | Colab inference ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 1207 | 274 | 19 | 80.47% | 0.399 |
| 256 | 1193 | 283 | 24 | 79.53% | 0.408 |

Intervals retain each board’s two seats and three opponents together. Board-only intervals condition on the fitted policies. The separate crossed bootstrap resamples paired training seeds as well as shared board clusters. Five seeds give preliminary evidence about training variability.

Secondary selection uses win fraction, fewer truncations, then earlier step; 50 common test boards per policy. Selections were frozen before final testing. This differs from the original 12-game mean-reward selection.

The fixed heuristic prioritizes ending the turn over every maritime trade. These results measure reliability against the specified opponents, not general Catan strength, negotiation quality, or browser latency. PPO probabilities are not Q-values or calibrated win probabilities.

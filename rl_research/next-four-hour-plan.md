# Four hour PPO reliability experiment

Prepared 2026-11-11 after reviewing the completed overnight results. Status: planned; no new compute launched.

Use one Colab T4 for at most four hours, including setup, retries, evaluation and artifact retrieval. The next question is whether the compact PPO policy learns reliably across seeds, and whether the larger policy actually reduces stalled games. Keep the learning problem unchanged so the answer is interpretable.

## What the completed experiment establishes

The raw final-game rows reproduce the reported totals. The current trainer, environment, configuration and dependency-lock hashes match the executed source manifest. Inspection found no fatal error in Double DQN target selection, legality masking or PPO terminal/truncation bootstrapping.

| Configuration | Test wins | Training minutes per run | Truncations |
| --- | ---: | ---: | ---: |
| PPO with two layers of width \(128\) | \(83.33\%\) | \(19.56\) | \(27/1200\) |
| PPO with two layers of width \(256\) | \(82.25\%\) | \(19.53\) | \(5/1200\) |
| Double DQN with width \(256\) | \(55.92\%\) | \(30.48\) | \(25/1200\) |

These are validation-selected checkpoints, not necessarily the common final checkpoint. The PPO width difference is only \(1.08\) percentage points, with a board-cluster interval of \([-1.58,3.92]\). That interval conditions on two training seeds. PPO-128's individual win fractions were \(91.50\%\) and \(75.17\%\), so training variability matters more than the observed width difference.

Checkpoint selection used only \(12\) games. PPO-128's second seed selected the \(106{,}496\)-transition checkpoint after winning all twelve; its final checkpoint won eleven. This is insufficient evidence of regression. That early-selected policy also accounts for \(21\) of PPO-128's \(27\) truncations. Compare common final checkpoints before attributing the timeout difference to network size.

DDQN performed approximately \(16\) times as many optimizer steps as PPO, with a small sparse-reward replay buffer. Its result justifies deprioritizing this configuration, not concluding that action-value learning is unsuitable.

Sources: [raw final games](results/overnight-20261111/final-games.csv), [paired comparisons](results/overnight-20261111/paired-comparisons.json), [validation curves](results/overnight-20261111/learning-curves.csv), [implementation details](results/overnight-20261111/implementation-details.json), [executed manifest](results/overnight-20261111/artifact-manifest.json).

## Experiment matrix

| Arm | New training seeds | Transitions per run | Runs |
| --- | --- | ---: | ---: |
| Existing masked PPO, width \(128\) | \(155921,196613,262147\) | \(434{,}176\) | \(3\) |
| Existing masked PPO, width \(256\) | Same seeds | \(434{,}176\) | \(3\) |

Train from scratch. Together with the four historical PPO runs, this gives five independent training seeds per width. Additional training totals \(2{,}605{,}056\) learner transitions. Rotate width order across seed pairs. Pairing training seeds helps organize the comparison; it does not create identical policy trajectories or random initial weights across widths.

Retain the exact observation/action mapping, architecture, initialization, optimizer, PPO settings, terminal reward, \(\gamma=1\), opponent mixture and game rules. Retain the validated single-worker collection path. Changing representation, reward, batching or opponent strength here would add a second explanation for any improvement. Use these unchanged conditions to establish a reliable baseline before a separate self-play or action-value experiment.

Use the existing four nonzero checkpoint positions: \(106{,}496\), \(217{,}088\), \(323{,}584\) and \(434{,}176\). The primary endpoint is always the last checkpoint at the common transition budget. Validation-selected checkpoints are a secondary practical comparison.

## Validation and final testing

Use fresh seed ranges, after checking them against all recorded training/evaluation manifests:

| Split | Board seed start | Boards | Games per policy/checkpoint |
| --- | ---: | ---: | ---: |
| Validation | \(2{,}100{,}000{,}000\) | \(20\) | \(120\) |
| Final test | \(3{,}100{,}000{,}000\) | \(50\) | \(300\) |

Each board is played in both seats against each of the three existing opponents. Freeze both manifests before training. Previous final-test seeds are now historical development evidence; do not reuse them as the new untouched test set.

Reevaluate the sixteen historical nonzero PPO checkpoints on the expanded validation set. Evaluate the same checkpoint positions for the six new runs. Select by observed win fraction, then fewer truncations, then earlier checkpoint; freeze this rule in advance. This is a deliberately revised selection procedure and must be labeled separately from the original mean-reward selection.

Commit selections before final testing. First evaluate all ten common-final-step policies on the fresh test set. Then evaluate distinct validation-selected checkpoints where time permits. If the secondary evaluation cannot finish, reduce its board count uniformly across policies, retain complete board blocks, and report it separately. Never select models from final-test outcomes.

Report per-seed results, pooled wins/losses/truncations, per-opponent outcomes, and paired width differences. Give board-cluster intervals conditional on the fitted training seeds and a separate estimate that resamples paired training seeds as well as board clusters. Five seeds still permit only preliminary statements about training variability. Do not present a narrow board-only interval as evidence that training is reliable.

## Stalled game diagnostics and benchmark limits

The final-game records indicate the observed truncations reached the turn limit rather than the engine-action cap. Record action-type counts, end-turn frequency, maritime trades, repeated public board/own-hand states and final score for every new truncation. Exclude hidden opponent cards from diagnostic views intended for policy analysis.

The existing heuristic is a useful regression opponent but a weak economy baseline: it assigns every maritime trade a lower priority than ending the turn. Thus these win rates do not establish strong trading or general Catan strength.

Before allocating the GPU, optionally prepare and test one frozen public-information heuristic that accepts a bank trade when it immediately enables a useful legal purchase. Keep it out of training and checkpoint selection. If validated and time remains, challenge all ten common-final-step policies on the same \(25\) fresh boards with both seats. Report those results separately. Omit this optional challenge before reducing primary evaluation coverage.

## Four hour allocation

| Budget | Work |
| --- | --- |
| \(20\) minutes | T4/source/checkpoint verification and expanded validation of historical checkpoints |
| \(120\) minutes | Six fresh PPO training runs; approximately \(20\) minutes each at measured throughput |
| \(45\) minutes | New-run validation, primary final tests and secondary tests as time permits |
| \(20\) minutes | Reports, checkpoint retrieval, integrity checks and runtime release |
| \(35\) minutes | Recovery/throughput reserve and optional challenge evaluation |

Training and validation can interleave; their cumulative times must be accounted for separately. Existing final evaluation took approximately \(3.5\)–\(5.0\) minutes per \(600\) PPO games. These allocations are estimates, not guaranteed runtime. Historical-checkpoint validation is approximately \(12\)–\(16\) minutes of inference at those rates.

Calibrate one short PPO pilot and evaluation batches from both an early, weaker checkpoint and a final model; earlier policies may produce longer games than the selected models used for the timing estimate. If the projected complete workload cannot fit, reduce validation/test board counts uniformly before starting the sweep and record the revised manifest. Preserve the three fresh seed pairs and common transition target when feasible. If a run still misses its budget, mark it incomplete and exclude it from the completed fixed-budget comparison rather than silently shortening only that arm.

Reserve the last \(20\) minutes for retrieval and shutdown. Enforce \(14{,}400\) seconds of total allocated runtime across retries. Stop optional diagnostics first, then secondary checkpoint evaluation. Never spend the artifact/shutdown reserve on another training run.

## Implementation and recovery requirements

Prepare the new orchestration locally before allocating the T4. Use a new experiment manifest and output directory. Preserve the executed trainer/configuration files so historical checkpoints continue to pass their source/configuration checks. A separate runner can pass new seeds and evaluation ranges to the existing Trainer and evaluate functions; the old CLI's fixed validation/final ranges are insufficient for this experiment.

Do not reuse the original controller's eight-hour assumptions or hardcoded ninety-minute reserve unchanged. Make the four-hour deadline, phase reserves, split manifests and common target explicit. Verify that resume retains the original deadline and does not retrain completed runs.

Some existing configuration fields, including opponent probabilities and rule parameters, are descriptive rather than wired into the environment. This does not invalidate the completed run because the actual behavior matches the declared values. Keep those values unchanged here; test configuration-to-behavior wiring before any later curriculum or rules ablation.

Historical checkpoints are recorded at `/root/tinysettler-artifacts/tinysettler-ablation-20261111-b/` on the VPS, outside the synced repository. Verify the necessary checkpoints against the existing manifest before provisioning. This review checked the local results and source hashes; it did not independently reread those remote checkpoint bytes. Use the archived implementation for historical loading rather than disabling provenance checks.

Retain atomic checkpoints, external verified copies, independent evaluation RNG, bounded transfer retries and the deadline watchdog. Log actual GPU allocation, learner and engine throughput, training/evaluation time, model/source hashes and every incomplete run. Confirm that the dedicated runtime is released after retrieval.

## Decision after the experiment

Choose a larger model only if common-budget results show a repeatable advantage across training seeds or a persistent reduction in stalled games that matters for the demo. If the width comparison remains inconclusive, retain PPO-128 provisionally for its smaller model, without claiming the architectures are equivalent.

If expanded validation substantially changes which checkpoints appear strongest, treat checkpoint selection as the explanation to investigate before claiming overtraining. If performance remains highly variable across five seeds, address training stability before broadening architectures. A subsequent strength experiment should introduce stronger frozen opponents or historical self-play; a direct action-value experiment should separately retest DDQN update/replay settings or fit an evaluator from PPO rollouts.

Deliver the fresh manifest, common-step and selected-checkpoint tables, per-seed learning curves, timeout diagnostics, checkpoint locations/hashes, and a short recommendation. State explicitly that this experiment measures reliability against the specified opponents, not negotiation quality or browser latency.

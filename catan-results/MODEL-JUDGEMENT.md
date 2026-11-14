# Model judgement for this browser Catan prototype

**Prefer Qwen3-1.7B with bounded thinking for the current slower, quality-oriented mode. Keep no-thinking play as the fast baseline. Do not use Qwen3.5-2B thinking as the default in this configuration.** This is a judgement about these INT4/WebLLM game-agent configurations, not a global ranking of the base models.

Qwen3.5 was stopped at the user's request. It is an interrupted game, not a loss or a turn-cap result. No further inference was run for this diagnosis.

## Measured results

| Final thinking run, seed 42 | Outcome | Completed choices | Median / p95 / maximum decision | Reasoning closed naturally | Budget terminations |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | Won: 10 VP, turn 142 | 55 | 14.01 / 22.17 / 24.43 s | 44/55 | 10/55 |
| Qwen3.5-2B | Stopped: 6 VP, turn 154 | 54 | 24.98 / 25.88 / 27.48 s | 24/54 | 30/54 |

All completed choices were below the requested \(30\)-second threshold, not merely the medians. Qwen3.5's additional attempt 55 was stopped after \(0.321\) seconds. It is preserved in the raw report and listed separately in `diagnosis-summary.json`; including it as a completed decision would artificially improve latency. There were no model/API failures among the completed choices. The 12 earlier calibration attempts are not mixed into these final-run statistics.

Qwen3 took 13.93 minutes to win. Qwen3.5 had run for 21.02 minutes when stopped. Qwen3 finished with four cities and longest road, worth \(4\times2+2=10\) VP. Qwen3.5 had upgraded its two starting settlements to cities and earned longest road, worth \(2\times2+2=6\), but had built no additional settlements.

Both used a 256-token reasoning ceiling, 12-second decode ceiling and 18-second reasoning-stage ceiling, followed by a separate greedy constrained action-selection pass. No adaptive budget reduction occurred. Qwen3 reasoning had median 110 generated tokens, with 44 natural closures, ten token-limit cutoffs and one model stop without a closing tag. Qwen3.5 had median 218.5 generated tokens, with 24 natural closures, 25 stage-deadline cutoffs and five decode-deadline cutoffs. Its median reasoning stage was 17.98 seconds, close to the wall limit; the selection stage added a median 7.26 seconds. Qwen3's corresponding stage medians were 8.83 and 5.38 seconds. Stage medians need not sum to the overall median.

Each decision processes the board twice. Median reasoning/selection prompt lengths were 2,289/2,345 tokens for Qwen3 and 2,271/2,455 for Qwen3.5. Thus the longer latency is not just the length of its visible reasoning; repeated prompt processing and runtime overhead also matter.

## What went wrong in Qwen3.5

Trace numbers below are **one-based positions in `decisions`**, matching `decisions[number-1]` in the archived JSON. Evidence is from [the stopped Qwen3.5 trace](1793577953996-42-Qwen3_5-2B-q4f16_1-MLC.json).

| Decisions / player-turn | Observed issue |
| --- | --- |
| 1 and 3 / setup | Starting sites N0 and N22 produce brick, ore and wood, but neither wheat nor sheep. That leaves both city upgrades and new settlements dependent on costly bank trades. |
| 18–19 / turn 50 | Traded four brick for one wood, then four wood for one ore. Direct four-brick-for-one-ore was already legal at decision 18, action 14. The two-step route wastes three extra wood and another model call. This is not caught by the simpler adjacent inverse-trade heuristic. |
| 24 / turn 58 | Moved the robber to T6/ORE9, blocking both its own settlements N0 and N22, and stole nothing. The draft incorrectly described this as a free road and claimed eight VP when the input said two. Six of its seven robber moves selected “steal nobody.” |
| 27 / turn 58 | The draft recommended road N0–N20, but the final pass selected N9–N10. This is a draft/selection mismatch, not evidence that the displayed draft faithfully explains the executed move. |
| 30 / turn 70 | Described an ordinary road purchase as two free roads worth two VP, with invented costs, despite having no development cards. |
| 47 / turn 122 | Ended the turn with two wheat and four ore even though both city upgrades were listed as legal. The draft said no new building path existed. It upgraded one city at turn 126 instead. |
| 49 / turn 130 | Identified action 1 as road N4–N5, while action 1 actually meant N10–N11, which the engine executed. |

These are state-reading, rule, action-mapping and planning errors, not JSON/legality failures. The grammar prevents invalid IDs, but cannot prevent an unhelpful legal move. It made 13 post-setup roads, two city upgrades, 13 trades and one development-card purchase/play. Those actions show basic game operation, but their explanations and sequencing were unreliable.

The separate final pass can change or repair a draft; it can also receive misleading analysis. This two-pass mechanism is a plausible contributor, not an isolated proven cause. Long free-form reasoning is not automatically useful reasoning.

A read-only runtime audit found no confirmed template bug or context overflow: the thinking opening and corrected model-specific stop tokens are preserved, both stages reset their caches, and the longest recorded prompts are below the 8,192-token window. The system explicitly permits thinking, although the observation still says “No explanations”; that conflicting wording is a possible burden for a small model. Qwen3.5's semantic errors also appear early in drafts, before the deadline, so truncation alone cannot explain them. Sampling differs between the models, and the runtime does not expose top-k; these remain comparison limitations.

## Qwen3 was better here, but is not reliable reasoning

The [winning Qwen3 trace](1793577116699-42-Qwen3-1_7B-q4f16_1-MLC.json) added two settlements, upgraded four cities and made 15 trades, with no detected adjacent inverse-trade cycle. Its opening four selections were unchanged from its original no-thinking seed-42 run: all action 0, including the same N0/N2 settlements. The win therefore did not result from a better opening in this trace.

Its drafts still have defects. Large menus are sometimes copied until the budget expires (decisions 1, 3, 52 and 54). Its final winning-city explanation at decision 55 claims wheat production at N6, although N6 touches wood10/ore9/wood2. Correct game actions do not make its explanations trustworthy.

## Historical comparison and limits

The original no-thinking results were Qwen3 at 7 and 6 VP at the turn-160 cap, with median decisions about 4.9 seconds; Qwen3.5 won seed 42 at turn 146 and reached only 3 VP on seed 43, with median decisions about 6.0–6.1 seconds. The larger model therefore was already highly variable, and previously won without thinking. Its stopped thinking run cannot be scored as a loss or extrapolated to its eventual outcome.

There is only one final thinking seed per model. Compared with the earlier runs, observation details, Python hash-seed/action ordering, sampling, the thinking-prefix scaffold, the two-pass protocol and the turn cap changed. Different actions also change subsequent RNG consumption. The current result does **not** establish that thinking caused Qwen3's improvement or intrinsically makes Qwen3.5 worse. Qwen3's win occurred before the old cap, but that alone does not remove the other confounds.

Both models used the same INT4/FP16 MLC family of exports and weak weighted-random opponents. There was no unquantized or native-runtime control in this Catan rerun, so these errors cannot be attributed uniquely to model capacity, quantization or WebLLM. Legal actions and short strategy reminders provide substantial scaffolding.

The earlier [Qwen3.5 diagnostics](../test-results/DIAGNOSTICS.md) do provide evidence of precision and protocol sensitivity on the six simpler tool tasks: within ONNX, native-tool strict passes increased from \(3/6\) with q4f16 to \(5/6\) with FP16 weights; custom-JSON passes increased from \(2/6\) to \(4/6\). WebLLM q4f32 remained at \(0/6\), because FP32 computation still retained 4-bit weights. Those small, separate tests do not identify the cause of this Catan failure or establish a WebLLM bug, but they make quantization/export and protocol sensitivity plausible contributors. The current behaviour should not be blamed solely on model capacity.

For this app, Qwen3 bounded thinking is the more defensible current quality-oriented default: faster, more often completes a short thought, and demonstrated a complete win. For speed, retain no-thinking mode and measure multiple seeds. Before trusting either as a general agent, use structured state summaries, action-specific consequences, explicit resource accounting and more matched games. Do not spend more time merely increasing Qwen3.5's thought budget without addressing the observed state/action confusion.

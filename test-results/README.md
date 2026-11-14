# Browser measurements — November 1, 2026

Actual inference in the Codex in-app Chromium browser on a MacBook Pro with Apple M2 Pro and 16 GB unified memory. WebGPU adapter metadata reports `apple / metal-3`; the system reports Metal support. No CPU inference fallback was used.

| Model / output mode | Tasks passed | Invalid actions | Median decode speed | Median first text chunk | Suite time |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B, prompt only | 0/6 | 41/43 turns | 26.3 tokens/s | 185 ms | 83.5 s |
| Qwen3.5-2B, JSON schema | 1/6 | 0/12 turns | 27.3 tokens/s | 436 ms | 13.1 s |
| Qwen3-1.7B, JSON schema control | 4/6 | 0/29 turns | 53.2 tokens/s | 130 ms | 15.3 s |

All models use WebLLM 0.2.87, `q4f16_1`, greedy decoding, thinking disabled, seed 42, and the same six scenarios and system instructions. Each mode ran once; these are smoke-test outcomes, not statistically reliable model rankings. The schema changes only permitted output syntax, not the task objectives or tool semantics.

Qwen3.5's initial model load took 35.2 seconds, including downloads. Its cached reload took 1.9 seconds. Qwen3-1.7B's initial load took 42.5 seconds. Suite times exclude model loading but include first-generation overhead. First-chunk timing is measured by the app; decoding speed comes from WebLLM's usage metadata. Different output lengths, failures, and early termination make suite duration unsuitable as a direct speed comparison. The original UI used an upper-middle median for even turn counts; the UI was corrected, and this table uses the conventional average of the two middle values.

## Observed behavior

Qwen3.5 frequently ignored the requested action envelope without a schema, returning alternate keys, multiple JSON objects, or quoted invented tool results. With a schema it produced valid actions, but often ended immediately with an incorrect or unsupported final answer. Its successful torch task used real tools, hit an insufficient-resources error, gathered more wood, and then completed both torches. The grader rejected claimed deliveries and claimed shelters when the world state did not contain them.

The Qwen3-1.7B control passed the order-total, shelter, and both blocked-route tasks. It failed the remaining-budget arithmetic task and repeated crafting without resources until the step limit in the torch task. This verifies that the same harness can execute multi-turn planning and recovery, but does not isolate whether Qwen3.5's poorer results arise from weights, quantization, the MLC implementation, prompting, or disabled thinking. That limitation applied to the original run. Subsequent runtime, precision, thinking and native-tool-format tests are documented in [DIAGNOSTICS.md](DIAGNOSTICS.md).

## Artifacts

- `qwen35-2b-prompt-only.json`: complete initial baseline.
- `qwen35-2b-schema.json`: complete Qwen3.5 constrained run, including system prompt.
- `qwen3-17b-schema.json`: complete control run.
- `latest.json`: latest app run; overwritten when running again.

Reports contain every raw model response, parsed action, tool observation, token usage, latency, final state, and objective verdict. The initial baseline predates adding the system prompt to report metadata; its prompt is the unchanged `systemPrompt` in `src/scenarios.js`. Runtime version is pinned, but upstream model weights are fetched from the model repository's `main` revision, so future reruns may download different artifacts.

Recompute summaries with `node scripts/summarize.js test-results/*.json` from the project root.

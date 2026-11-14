# Qwen3.5 follow-up diagnostics

The original result was partly a protocol/prompt problem, with additional sensitivity to precision and runtime/export. It does **not** establish that Qwen3.5-2B cannot use tools, nor does this experiment isolate a WebLLM implementation bug.

## Results

All rows below use the same six scenarios, simulated tool implementations, and strict graders. Native-tool rows change the system prompt, tool definitions, chat template, observation roles, and final-answer format together. Non-thinking rows use greedy decoding and a 160-token per-turn limit; thinking uses sampled decoding and 2,048 tokens. Every suite ran once.

| Qwen3.5-2B configuration | Strict passes | Tool objectives reached | Invalid actions | Suite time |
| --- | --- | --- | --- | --- |
| WebLLM q4f16, custom JSON | 0/6 | 0/6 | 41/43 | 83.5 s |
| WebLLM q4f32, custom JSON | 0/6 | 0/6 | 41/43 | 94.1 s |
| ONNX q4f16, custom JSON | 2/6 | 2/6 | 30/46 | 61.6 s |
| ONNX FP16 weights, custom JSON | 4/6 | 4/6 | 20/37 | 73.5 s |
| ONNX q4f16, native Qwen tools | 3/6 | 5/6 | 6/31 | 73.2 s |
| ONNX FP16 weights, native Qwen tools | 5/6 | 6/6 | 1/29 | 114.8 s |
| WebLLM q4f16, sampled thinking, custom JSON | 0/6 | 0/6 | 6/7 | 553.6 s |

“Strict passes” requires the requested final format and termination as well as actual tool success. “Tool objectives reached” is a **post-hoc diagnostic**, computed from saved world state/calculator observations while ignoring final-answer formatting and termination. It does not replace the original score. The summary script reports both; raw reports retain their original verdicts.

## Interpretation

**Protocol matters.** The native template supplies Qwen's trained XML tool-call format and wrapped tool responses. FP16 solved both arithmetic tasks with actual calculator calls; the custom JSON version failed both. Native q4 also calculated the first answer correctly, but its verbose final violated “numeric answer only.” Both native runs built the shelter but failed to terminate cleanly within ten turns; q4 eventually emitted malformed inventory calls, while FP16 kept inspecting inventory. Native q4 failed the torch objective by repeatedly crafting without resources. Native FP16 completed it. Thus low strict scores combine actual planning failures with format and stopping failures.

**Precision matters in these traces.** Within the same ONNX export and protocol, FP16 outperformed q4f16: 4 versus 2 strict passes with custom JSON; 5 versus 3 with native tools. This is evidence of sensitivity to the export's quantized variant, not a reliable population estimate from six tasks. MLC q4f32 did not help because it still uses 4-bit weights. FP16 ONNX native decoding had a median of 13.4 tokens/s versus 40.8 for q4, with different generated sequences and no excluded warmup.

**WebLLM is not the sole explanation.** ONNX also failed several custom-JSON tasks, including with FP16 weights. The matched text-format q4 ONNX run improved on WebLLM, but these are different quantizers, converted weights, kernels, and cache paths. No identical-weight cross-runtime experiment or native MLX/Transformers reference was performed. A WebLLM/MLC-specific contribution remains unresolved; the known stop-token bug is already fixed in the installed version.

**Thinking did not rescue this tested setup.** Five tasks exhausted the 2,048-token limit; one returned 100 without the required calculator call. The run used the correct thinking prefix, prompt-only output, temperature 1, top-p 0.95, presence penalty 1.5, and seed 42. It does not measure unlimited reasoning or thinking with the native tool protocol. Qwen's own model card warns that the 2B model is prone to thinking loops.

For this app, use the ONNX native-tool option for further agent experiments. FP16 provides the stronger reference here; q4 is substantially faster. Keep strict success, tool-state success, invalid calls, and termination failures separate when evaluating changes.


## Configuration findings

The original control was `mlc-ai/Qwen3-1.7B-q4f16_1-MLC`, using WebLLM 0.2.87. MLC defines `q4f16_1` as INT4 group quantization with group size 32, packed UINT32 storage, and FP16 model dtype. Both embedding and output projection quantization are enabled by this scheme. `q4f32_1` retains INT4 weights and group size 32 but uses FP32 model dtype; it is not an unquantized model.

The installed WebLLM 0.2.87 already contains the Qwen3.5 stop-token correction and bounded JSON whitespace fix from PR 876. The original tests therefore were not running the older version affected by that reported JSON termination bug.

The official Qwen3.5 template opens `<think>\n` when thinking is enabled, and inserts an empty thinking block when disabled. WebLLM's generic `enable_thinking: true` path does not itself insert the opening prefix. The app now overrides the empty assistant header while preserving the corrected stop tokens. Thinking output is removed from subsequent conversation history, and the full remaining conversation is prefetched for each thinking turn.

Thinking cannot be tested meaningfully with a grammar that permits only a JSON object from the first generated token. Thinking runs therefore use prompt-only output, a 2,048-token output budget per turn, and an 8,192-token context. Unfinished generations at the output limit fail the episode rather than consuming ten more retries. The system prompt is adjusted only to allow the thinking block; scenario instructions and objective graders are unchanged.

The greedy thinking probe was manually interrupted after observing repetitive reasoning. Its partial text is recorded in `qwen35-thinking-greedy-probe.json`; it is not counted as a completed suite. A sampled thinking run uses temperature 1.0, top-p 0.95, presence penalty 1.5, and seed 42. WebLLM does not expose top-k in its generation API, so this is not an exact reproduction of every recommended Qwen parameter.

## Sources

- [MLC quantization definitions](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py)
- [Qwen3.5-2B chat template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja)
- [Qwen3.5 model card, sampling recommendations and thinking-loop warning](https://huggingface.co/Qwen/Qwen3.5-2B)
- [WebLLM 0.2.87 release](https://github.com/mlc-ai/web-llm/releases/tag/v0.2.87)
- [Qwen3.5 JSON termination fix](https://github.com/mlc-ai/web-llm/pull/876)
- [Independent ONNX export](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT)

`model-provenance.json` records repository revisions observed during this investigation; original browser cache entries were not cryptographically checked against those revisions.

## Thinking result

The complete WebLLM `q4f16_1` sampled-thinking suite passed 0/6 tasks in 553.6 seconds. Five tasks reached the 2,048-token per-turn cap; the remaining arithmetic task returned the correct value 100 without calling the required calculator. This is a bounded-budget result, not evidence that unrestricted thinking could never solve the tasks. The cap is much lower than Qwen's broad benchmark recommendation, but the traces show repeated protocol deliberation rather than completed tool actions. See `qwen35-2b-thinking-sampled.json`.

## FP32 computation control

WebLLM `q4f32_1` with thinking off and prompt-only greedy decoding passed 0/6, matching the 0/6 `q4f16_1` baseline. Some outputs differ, but both repeatedly fail the action protocol and invent tool results. Increasing compute precision while keeping INT4 weights did not fix this setup. This does not test unquantized weights.

For ONNX comparisons, `src/chat-format.js` renders the same text-only ChatML envelope as the WebLLM control, including the empty thinking block in non-thinking mode and raw assistant history. ONNX recomputes the full prompt each turn; WebLLM's non-thinking runs reuse conversation state. This cache-path difference remains a potential runtime confound.

## Reproduction and limitations

Use the app's Model and Protocol selectors, load the model, and run all six scenarios. For the thinking comparison select WebLLM q4f16, Custom JSON, Thinking On, and Sampled. Selecting thinking disables JSON grammar constraints. Raw reports named `qwen35-2b-*.json` contain full traces; `diagnostic-summary.json` contains derived comparisons.

The ONNX export is `onnx-community/Qwen3.5-2B-ONNX-OPT`, loaded with Transformers.js 4.3.2 and ONNX Runtime Web 1.31.0-dev.20260914-8d85527a0 on WebGPU / Apple Metal. No vision module was loaded. Models come from upstream `main`, not immutable download revisions. MLC and ONNX 4-bit schemes are not identical.

The original WebLLM context limit was 4,096; follow-up WebLLM runs use 8,192. ONNX recomputes the full prompt and did not enforce that app limit; older ONNX report metadata retained `contextWindow:8192` from the shared settings despite no runtime cap being applied. Current reports use null for ONNX. These short non-thinking tasks fit below either WebLLM limit. The original custom-JSON arithmetic prompts also combine “numeric answer only” with a system-level JSON-final requirement; the native protocol resolves that ambiguity.

Timing definitions differ: WebLLM measures first nonempty text chunk; current ONNX measures first token callback. The initial ONNX q4 custom-JSON run used first text chunk before the callback fix. Do not compare those latency numbers directly. All suite durations exclude model loading and include compilation/warmup; speed numbers are descriptive rather than controlled performance benchmarks.

Native tools are parsed and validated locally; they do not execute arbitrary code. Unit tests cover malformed tool calls, tool-result feedback, strict grading, world state, thinking history, and output cutoffs. No graders were relaxed for these experiments.

# TinySettler browser agent lab

A tiny local app for testing Qwen3.5-2B agents through WebLLM or Transformers.js / ONNX Runtime Web, using WebGPU. On macOS, the browser uses Metal beneath WebGPU; JavaScript does not call Metal directly. There is no Python inference server or paid model API.

## Run

For the Mac/VPS working-file sync, persistent Rex container, and optional cloud
clients, see [development setup](dev/README.md).

```sh
npm ci
npm run dev
```

Open the printed localhost URL in a browser with WebGPU and `shader-f16` support. Click **Load model**, then **Run all 6 scenarios**. The first load downloads roughly 1 GB of weights plus runtime assets; subsequent loads use the browser cache. WebLLM lists approximately 2.2 GB GPU memory for the default model; actual total browser memory can be higher.

The default is `Qwen3.5-2B-q4f16_1-MLC`. The selector also includes its FP32-compute variant, an independent ONNX export with 4-bit or FP16 weights, Qwen3-1.7B, and Llama-3.2-1B. ONNX text weights download about 1.4 GB for q4f16 or 4.8 GB for FP16. ONNX FP16 needs substantially more memory. A web worker runs inference while the UI displays live output. **Stop & unload** terminates the worker immediately, retaining the download cache. Reload the model to run again.

## Experiments

Six fixed tasks cover three behaviors:

| Behavior | Tasks | Verification |
| --- | --- | --- |
| Tool selection | Order total; remaining budget | Correct numeric final and matching calculator observation |
| Multi-step planning | One shelter; two torches | Crafted inventory contains exactly the target count |
| Error recovery | Parcel A over bridge; parcel B through tunnel | Initial requested route fails, then ferry delivery succeeds |

Each episode starts with an empty world and a reset model conversation. The model emits one JSON action per turn. The harness executes only allowlisted, validated simulated tools and feeds the result back as a user observation. Choose **JSON schema** for grammar-constrained actions or **Prompt only** to test format compliance without constraints. **Protocol → Qwen native tools (ONNX)** instead uses the tokenizer's official tool template, function schemas, XML tool calls and tool-role observations. This also changes the system instructions and final-answer format, so compare it as a protocol/prompt ablation, not as an isolated runtime change. Invalid JSON returns an error and consumes a turn. Calculator inputs use fixed operations; no `eval` or arbitrary execution is available. A syntactically valid final response still fails unless the real world state or calculator observation supports it.

Defaults use greedy decoding, thinking disabled, 160 output tokens per turn, and at most 10 turns per task. WebLLM uses seed 42 and an 8,192-token context (the original runs used 4,096). ONNX recomputes the full prompt each turn. **Thinking → On** opens the proper thinking prefix, removes completed reasoning from subsequent history, permits 2,048 output tokens per turn, and forces prompt-only output. An unfinished thinking generation fails at that cap. WebLLM also offers sampled decoding with temperature 1, top-p 0.95 and presence penalty 1.5; top-k is unavailable. ONNX is restricted to greedy decoding in this app. Repeated runs are stability checks, not independent sampled trials. These easy, explicitly instructed tasks are a smoke test, not evidence of general autonomous-agent ability. In particular, the recovery tool explicitly supplies the alternate route. No vision is tested.

Metrics include per-turn latency, decode speed, episode duration, token usage, model output, tool observations, and final state. WebLLM latency measures first nonempty text chunk, and speed uses runtime metadata. ONNX speed uses token callback timing; current ONNX latency measures first generated token. The initial ONNX q4 custom-JSON report predates that timing correction and measures first text chunk instead. Do not compare these latency definitions as equivalent. The UI shows medians across completed turns. Startup/download time is recorded separately. First-generation compilation can affect timings; the suite has no excluded warmup. Failed and truncated generations remain in traces.

**Export JSON** downloads a complete report. In development, the app also saves the report after each completed episode to `test-results/latest.json` through a same-origin, localhost-only endpoint. A new suite overwrites this file. Browser local storage retains and restores the latest report; the model itself must be loaded again after a refresh. In production builds only browser storage and JSON export are used. Model weights come from Hugging Face and model runtime assets from the official MLC repository; prompts stay in the browser. The font stylesheet is fetched from Google Fonts.

## Checks

```sh
npm test
npm run build
node scripts/summarize.js test-results/*.json
```

Unit tests verify resource accounting, strict action handling, actual error recovery, fabricated-final rejection, observation feedback, and step limits. They do not run a language model. Actual browser measurements, when available, are in `test-results/`.

## Sources

- [WebLLM model configurations](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts)
- [Qwen3.5-2B MLC weights](https://huggingface.co/mlc-ai/Qwen3.5-2B-q4f16_1-MLC)
- [Chrome WebGPU documentation](https://developer.chrome.com/docs/web-platform/webgpu/overview)

Runtime and dependency versions are pinned in `package.json` and `package-lock.json`.

## Diagnostic results

See [follow-up diagnostics](test-results/DIAGNOSTICS.md) for thinking, runtime, precision, and native-tool-format comparisons. The original results alone understated Qwen3.5 tool-use ability because they used a custom JSON protocol. Raw reports preserve settings and every action.

## Catan agents

Run `npm run catan` after installing the Python rules bridge, then open `/catan.html`. See [CATAN.md](CATAN.md) for setup, the numbered-action interface, model settings and evaluation limits. [Catan results](catan-results/README.md) report actual browser runs.

## Reinforcement learning results

Start with the [consolidated RL progress report](rl_research/progress-report-20261114.md): setup, all completed experiments, what worked, limitations and next priorities as of November 14, 2026.

The [RL research index](rl_research/README.md) links the design audit and completed experiments. The latest [recovered scaling study](rl_research/results/scaling-20261114-recovered/findings.md) includes results, raw game records and charts for ten PPO runs comparing model capacity and refreshed self-play. See the [independent interpretation and proposed next test](rl_research/results/scaling-20261114-recovered/analysis.md) and [training implementation](rl_research/scaling/README.md). The policies remain weak against the strong heuristic family; no reliable self-play or capacity advantage was established.

Download artifacts for all ten retained November 14 A–D runs from the [model release](https://github.com/breakingwave/TinySettler/releases/tag/rl-scaling-20261114): [portable policy exports](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/final-delivery.tar.gz), [full resumable checkpoints](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/full-checkpoints.tar.gz), and [SHA256 checksums](https://github.com/breakingwave/TinySettler/releases/download/rl-scaling-20261114/SHA256SUMS.txt). The release also includes checkpoints from the earlier overnight, reliability and curriculum experiments, linked in the [RL index](rl_research/README.md). Model archives are release assets, separate from a repository clone.

# Browser Catan agents

Catanatron runs the rules in a local Python process. Qwen3-1.7B and Qwen3.5-2B run in browser workers through WebLLM/WebGPU; on this Mac the adapter reports `apple / metal-3`. No model server, fine-tuning, search, or additional trained game policy is used.

## Run

```sh
npm ci
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r backend/requirements-lock.txt
npm run catan
```

Open http://127.0.0.1:5173/catan.html. **Compare both models** runs both models using the selected seed; the Seeds per model control optionally adds its successor. **Play one game** runs only the selected model after loading it. The board, resources, decisions, timings and scores update live. **Stop & unload** stops inference immediately. The local service holds one game, so do not run multiple game tabs concurrently. The launcher stops both child processes on Ctrl-C.

The original small tool scenarios remain at `/`. Both routes are included in `npm run build`; Catan play requires the local Python bridge and the Vite proxy (the static build alone is not a game server).

## Agent interface

Each decision contains RED's resource and development cards, public scores and card counts, a node graph with adjacent terrain/dice, buildings and roads, ports, robber location, recent public actions, and every legal action. No opponents' exact hands or hidden development cards are shown. The rules engine supplies legality, not rankings or recommended moves.

For example, a legal action menu might contain:

```text
0: end_turn
1: build_road N0–N20
2: build_settlement N8 (wood10/sheep5/ore8)
```

The model selects one action using `{"actionId":2}`. A dynamically generated JSON schema restricts the ID to the current legal list. The server checks the game version and ID again before execution. There is no fallback bot silently repairing model mistakes. This is a simplified action-selection interface, not evidence that the model can construct arbitrary tool calls unaided.

A short system prompt gives costs, point values and basic strategy reminders. This is prompt scaffolding, not fine-tuning. The original runs used greedy decoding, thinking off, up to 32 output tokens, an 8,192-token context, and a fresh complete observation each decision. The current bounded-thinking mode is described below. The tested models are `Qwen3-1.7B-q4f16_1-MLC` and `Qwen3.5-2B-q4f16_1-MLC`: 4-bit grouped weights with FP16 model dtype.

RED faces three Catanatron `WeightedRandomPlayer` bots. These prefer cities, settlements and development purchases, then otherwise choose randomly. Forced single legal moves are automatic and counted separately; the model chooses every non-forced RED action. Banking/ports, robber movement, resource discards, development cards, roads, settlements and cities are supported. Player-to-player trade negotiation is not exposed. See [backend details](backend/README.md).

## Evaluation

The initial comparison uses seeds 42 and 43, a cap of 160 completed player-turns (not 160 rounds), and 240 model decisions. A cap is truncation, never a win. Assess points, construction, resource use and outcomes separately from legal-action rate, because constrained decoding guarantees much of the latter. These are four feasibility episodes, not a statistically reliable strength comparison.

The first comparison used observation v1. A review found that its robber tile ID was not explicitly mapped to graph nodes, and generic harbors were grouped in rendering metadata. Current backend v2 adds the robber's terrain/dice/node IDs, explicit harbor ratios and development-card effects; the renderer also handles old grouped harbor snapshots. Original reports preserve the tested prompts. Model tests did not silently switch protocols partway through.

Live reports save to `catan-results/latest.json`; finished reports are also archived by game ID. Reports retain complete model observations, legal choices, raw output, token usage, latency and final public state. Startup/model loading is timed separately. Run:

```sh
node scripts/summarize-catan.js --write
npm test
.venv/bin/python -m unittest backend.test_catan_server -v
```

The summary distinguishes truncation from wins and reports action counts, median/p95 decision latency, first-chunk latency and token volume. First-chunk time includes prompt prefill and runtime/grammar overhead; it is not a pure GPU kernel measurement. Full-observation prefill is intentionally conservative and can be optimized later with a smaller encoding or prefix caching.

Model files are fetched from upstream `main`; package versions and the Catanatron commit are pinned. Initial live games ran in one server process without a fixed Python hash seed. The launcher now sets `PYTHONHASHSEED=0`; exact action ordering may differ from the initial traces. Seeds fix initial randomness within a process, but policies consume RNG differently, so later dice sequences need not remain matched. Baseline simulations in `backend/baseline-results.json` are contextual references rather than exactly matched RNG replays.

## Bounded thinking mode

The app now defaults to thinking on, a 320-player-turn cap, and one seed per model. Thinking is a two-pass procedure: the model generates a short reasoning draft with native thinking enabled, then receives the current observation and that draft in a separate greedy, grammar-constrained action-selection pass. Both passes run on the same browser model. No draft is carried into a later game decision.

The configured reasoning budget is 256 tokens, at most 12 seconds of decoding, and an 18-second cooperative reasoning deadline. Deadlines are checked between runtime operations and cannot cancel an in-flight GPU prefill kernel. The final selection pass has its own 30-second cooperative deadline. The target is a median *combined* decision latency below 30 seconds, not a guaranteed 30-second worst case. The controller reduces the token budget if the latest five-decision median exceeds 27 seconds; every adjustment and stage cutoff is recorded.

Calibration showed that unprefixed 128- and 256-token drafts often spent their budget restating the task. The final thinking prompt therefore prefills `Compare legal actions: ` inside the opened thinking block. This is logged separately from generated text. It steers the draft toward concrete alternatives without choosing an action for the model. The model-specific thinking sampling parameters are saved in each trace; final selection remains greedy. This is bounded, prompted reasoning, not unrestricted chain-of-thought generation.

The previous no-thinking reports are preserved in `catan-results/no-thinking-original.json`. Agent-stopped calibrations are archived separately and excluded from completed-game aggregates. Effective prompts, actual reasoning text, injected prefix, stage token usage, time limits and outcomes remain inspectable in the raw reports. Use `node scripts/summarize-catan-thinking.js --completed-only --write` for the follow-up summary. These runs also use observation v2 and a fixed Python hash seed, so they are a feasibility rerun rather than an isolated causal test of thinking against the historical v1 games.

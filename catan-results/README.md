# Catan browser evaluation — November 1, 2026

Actual inference on Apple M2 Pro / 16 GB, Codex in-app Chromium, WebGPU adapter apple / metal-3. Both models used WebLLM 0.2.87, q4f16_1, greedy decoding, no thinking, no fine-tuning. Each game used the same compact-v1 observation protocol, real Catanatron 3.3.0 rules, 10 VP target, and three WeightedRandomPlayer opponents. Model load time is excluded; generation warmup is included.

| Model | Seed | RED VP | Outcome | Player-turns | Model decisions | Median decision | P95 decision | Wall time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | 42 | 7 | Turn cap | 160 | 76 | 4.86 s | 5.76 s | 6.23 min |
| Qwen3-1.7B | 43 | 6 | Turn cap | 160 | 49 | 4.90 s | 5.73 s | 4.05 min |
| Qwen3.5-2B | 42 | 10 | RED won | 146 | 76 | 6.13 s | 7.30 s | 8.03 min |
| Qwen3.5-2B | 43 | 3 | Turn cap | 160 | 59 | 5.96 s | 6.95 s | 5.92 min |

All 260 model selections were legal and accepted. This is largely guaranteed by the constrained action-ID grammar and is not evidence of independent tool-format compliance. Forced single legal actions and opponent moves are excluded from model-decision counts. A turn-cap result is not a win or a completed Catan game.

## What the agents actually did

Qwen3.5's seed-42 win at turn 146 is supported by real engine state: three cities (6 VP), one settlement (1), longest road (2), and one victory-point development card (1). It constructed two post-setup settlements, upgraded three cities, bought seven development cards and played six. Its opening settlements had 20 production pips in total, compared with Qwen3's 16 on that board; this is a simple descriptive heuristic, not an optimal-placement score. Both openings lacked sheep production.

Qwen3 built additional settlements in both games and a city in the first, but did not win. In seed 42 it made consecutive trades of four brick for one ore and four ore for one brick, losing six cards. Qwen3.5's second seed was much weaker: no additional settlements or cities, only two post-setup roads, and a final score of three. A complete win on one board does not demonstrate reliable playing strength.

The interface supplies every legal action and brief rules/strategy reminders. The models select among those actions; no search scores, optimal moves, or fallback policy are provided. They see their own cards and public information, not exact opponent hands. There is no player-to-player trade negotiation. Weighted-random opponents are weak.

## Speed

Median decisions took about 4.9 seconds for Qwen3 and 6.0–6.1 seconds for Qwen3.5. Whole runs took 4.1–8.0 minutes. About 94–96% of measured inference time elapsed before the first text chunk, including full-observation prefill and runtime/grammar overhead. The observations typically contain roughly two thousand prompt tokens; answers are only a short action ID. This is usable for watching games but slow for large-scale simulation. Reducing repeated board input, caching static context, and grouping discard choices are potential future optimizations, not measured improvements.

## Controls and limitations

Separate CPU-only baselines are saved in ../backend/baseline-results.json and ../backend/index-one-baseline.json. All-weighted-random RED reached 9 and 5 VP at the cap. Always choosing ID 1 reached 6 VP (WHITE won) and 2 VP (cap), without reading any board text. These controls illustrate how much the legal menu alone can accomplish. They used a fixed Python hash seed, unlike the initial live server, so they are contextual references, not exactly matched action-order/RNG replays.

There are only two seeds per model. Models took different actions and therefore consumed random draws differently. The first observation version omitted a direct mapping from robber tile IDs to node IDs. The delivered v2 backend fixes that omission, explains port rates and development effects, and fixes generic-harbor rendering. The four model runs above all used v1; future runs use v2. Original prompts remain in the reports, and no outcomes were regraded.

## Artifacts and reproduction

Run from the project root:

```sh
npm run catan
node scripts/summarize-catan.js --write
```

Open /catan.html, select Thinking Off, Seeds per model 2, seed 42 and turn cap 160 to approximate the historical setup, or select a saved game to inspect its board. The current observation version differs as noted above. See ../CATAN.md and ../backend/README.md for installation and detailed settings.

no-thinking-original.json freezes the original four reports; latest.json now also includes later thinking experiments and calibration attempts. Timestamped JSON files preserve individual finished runs. summary.json is derived, and scripts/summarize-catan.js recomputes it without changing raw reports. Reports retain every model prompt, legal action list, raw response, timing and final state. Dependencies and the game-engine commit are pinned; upstream model weights are fetched from main rather than immutable revisions.

Validation passed: 13 JavaScript harness tests, 5 Python rules/bridge tests, and the Vite production build. The Python tests include a full game to an actual winner, privacy checks, stale-action rejection, geometry, and explicit robber mapping.

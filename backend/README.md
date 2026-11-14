# Catanatron rules bridge

This process runs the board-game rules only. Browser workers run the language models on WebGPU. No model inference or prompt transmission to a remote server occurs here.

## Install and run

Python 3.11 or newer is required. The project environment currently uses Python 3.14.7.

For the separate Python 3.12 Linux container environment, use the
[Mac/Rex setup](../dev/README.md). Do not recreate the existing Mac `.venv`.

```sh
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
PYTHONHASHSEED=0 .venv/bin/python backend/catan_server.py
```

The server binds only `127.0.0.1:8001`. The Vite app proxies `/api/catan` to it. Host and Origin checks reject unrelated sites; direct origins accepted are localhost ports 5173 and 8001. The process holds one active game in memory. Starting a new game replaces it. Stopping/restarting the server loses that game.

Catanatron is installed from the official repository, version 3.3.0 at commit `ecf931181b9a65bb4116a2153fb78c16f1438e00`. Dependencies are pinned in `requirements.txt`. Its upstream license is GPL-3.0-or-later.

## API

| Request | Input | Result |
| --- | --- | --- |
| `GET /api/catan/health` | — | Engine version and revision |
| `POST /api/catan/new` | `{"seed":42,"maxTurns":160,"autoForced":true}` | New game advanced to a RED decision |
| `GET /api/catan/state` | — | Current observation |
| `POST /api/catan/action` | `{"actionId":0,"version":"opaque value from state"}` | Apply one selected move, advance opponents, return next observation |

`version` includes a unique game ID and changes after every executed action. Stale versions and out-of-range IDs are rejected without applying the move. Legal-action IDs are local to each observation, not permanent action names. The server executes exactly the selected legal action; it does not repair a bad model choice or select a replacement.

Observations contain `prompt`, `actions`, `players`, RED's `hand` and `development` cards, board geometry, recent public `events`, and cumulative `stats`. Opponent exact resource inventories, development-card identities, stolen-resource identities, and private victory points are excluded. Opponent total resource/development-card counts and public victory points are included. RED sees its own victory-point cards. Discard event labels hide opponents' discarded resource identities.

The board encoding uses stable node IDs, adjacent resource/dice numbers, node-neighbor lists, buildings, roads, harbors and robber position. The full current legal-action menu is included, in engine order. No move ranking, search score, recommended action, or strategy model is supplied. A short rules/cost reference makes the interface usable without requiring memorized Catan syntax. It is prompting, not fine-tuning.

## Game setup and limitations

The standard four-player board uses a ten-victory-point target. RED is controlled by the browser model. BLUE, WHITE and ORANGE use Catanatron's `WeightedRandomPlayer`: cities receive weight 10000, settlements 1000, development-card purchases 100, and all other legal actions weight 1. These are simple weak opponents, not a strong search baseline. Seating, board generation, dice, card draws and opponent choices use the game's seeded RNG.

The menu includes bank/port trades but does not invent player-to-player trade offers. Catanatron supports such offers through a separate unrestricted API; this bridge deliberately exposes only its generated legal-action menu. The current engine lets players choose individual discard resources. Robber stealing and dice remain engine-controlled randomness. No board-game rules have been reimplemented in the browser.

When RED has exactly one legal move, the bridge can execute it automatically. Those moves are recorded separately as `forcedActions` and are not model decisions. Opponent moves are also counted separately. This reduces model calls for forced dice rolls/end turns without selecting among meaningful alternatives. A development card can create a choice before rolling, so rolling is not always automatic.

The game ends on the engine's winner signal, the configured total completed-player-turn limit, or a 5000-engine-action safety cap. A turn is one player's turn, not a complete round of four players. The browser may also apply its own decision limit. Truncated episodes are not wins or completed games. Public opponent points can be below the winner's actual total when hidden victory-point cards trigger the engine winner signal.

A constrained legal-action grammar makes move legality largely guaranteed by the interface. Assess victory points, construction, useful trades and wins separately; valid JSON alone does not establish strategic ability. Each new action is based on a fresh complete observation rather than a long conversation history.

For cross-process repeatability set `PYTHONHASHSEED=0`: some engine legal-action collections originate from sets. Seeds provide consistent game generation and RNG streams, but different policies consume different random draws and can diverge. Compare multiple seeds before interpreting win rates.

## Validation

```sh
.venv/bin/python -m unittest backend.test_catan_server -v
```

Tests cover a complete game to an actual winner with a 1000-turn allowance, legal move execution, automatic forced actions, stale/invalid/cross-game rejection, privacy of opponent inventories and development-card identities, seeded board/action consistency, graph geometry, and nine separate harbors. Tests do not run a language model.

The initial live comparison used the first server process with harbor node IDs grouped by resource in its JSON board representation. Game logic and available ports were correct; the rendered generic-harbor positions could be combined. The on-disk version now emits one harbor record per physical port. No active game was restarted for this visual correction.

## Observation version

On-disk observations now include `observationVersion: "catan-compact-v2"`. Version 2 spells out the blocked robber tile's resource, dice number and adjacent node IDs; it also explains port exchange rates and development-card effects. These are rule/interface descriptions, not move recommendations.

The initial active comparison and saved baseline reports used version 1, identifiable by the absence of `observationVersion`. Its prompt named the robber only by tile ID while board rows were node-based, so the model could not reliably connect that ID to blocked production. Port rates and development effects also relied more on the model's prior knowledge. The active server was deliberately not restarted during that comparison, and its measurements must remain labeled version 1. Restarting loads version 2 for future experiments; do not silently combine results across observation versions.

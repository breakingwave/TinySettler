# Initial setup verification — 2026-11-10

Historical record for the original VPS. See [relocation verification](migration-2026-11-10.md)
for the replacement VPS at `2.25.188.2`.

| Check | Result |
| --- | --- |
| SSH | `rexdev`, strict host-key checking; recorded ED25519 fingerprint `SHA256:nrg7liu92pgnd2CIHsLG4SqzYnOFPRDgLxZzOTP5tIQ` |
| Container | Original ID `1890dfb1dbeb1250e79ebdfc8089755a01c37a8e5c99b5bd23e2f6524f7009e8`; start time still `2026-11-02T02:13:27.736279128Z` |
| Persistence | `/home/rexdev/projects` mounted at `/workspace`; `rex-coding-home` at `/root`; existing Codex auth file retained |
| Sync | Only `tinysettler`, `two-way-safe`, watching, no conflicts/errors |
| Sync ID | `sync_wEt9eEfwJwtduTFsBdAAOM1dTmljuC6XMW5gKThyZZL` |
| Sync test | Unique disposable marker: writes and deletions verified in both directions; test directory removed |
| Working files | 111 eligible files matched SHA256 hashes before this verification note was added |
| Old sync | Full configuration archived, then paused and terminated after confirming both endpoints empty; no unrelated data moved/deleted |
| Mac | Node 26.8.1; Python 3.14.7 venv retained; uv 0.11.21 |
| Linux | Node 24.22.2; Python 3.12.16; uv 0.13.6; separate `.venv` and `node_modules` installed from checked locks |
| Tests | Mac and Linux: 18 JavaScript tests and 5 backend tests passed; both Vite builds passed |
| Live apps | Mac localhost frontend/backend healthy; container app moved from tmux to Rex session `tinysettler`, with separate shell and `app` panes |
| Remote preview | Frontend, backend health, Vite client HTTP 200 and WebSocket upgrade verified over SSH; temporary preview stopped afterward |
| Clients | Colab 0.7.5, Modal 1.6.3, Sail Research 0.18.1 versions/help verified on both platforms |
| Mac auth | Colab and Modal read-only account checks passed; Sail browser login pending at last check |
| Container auth | Colab, Modal and Sail browser logins pending at last check |
| Templates | Modal imported locally on Mac and Linux; Colab syntax and installed CLI options checked; no cloud execution |
| Git | No `.git` on either endpoint; no repository initialized or private data committed |
| Spending | No provider compute job, inference request, GPU instance, Volume or purchase created |

Dependency locks were retained. SHA256:

```text
package.json                  ea54a05b5a2f0e207cd5a8fe22903fd858c2a78edc4bd564d67fe0f7f0539887
package-lock.json             aa86b4c7b751795a3d2b1711bc055e4c3d22d911d67f65bcfc6a2ec3e4bb04f3
backend/requirements-lock.txt  1bf69dd99d61acab382dc572e6d5e9f37cf9c0fcea958f550bb108de30fa4c6b
```

The Linux npm install reported unapproved install scripts for esbuild,
onnxruntime-node and protobufjs. No global script policy was changed; the
Vite build and app passed. Native ONNX inference on Linux was not verified or
added. Browser model loading/inference was not rerun during this setup.

Authentication state can change after this snapshot. Run the read-only checks
in [README.md](README.md) after completing the browser prompts. Never include
tokens, private keys or raw credential output in this file.

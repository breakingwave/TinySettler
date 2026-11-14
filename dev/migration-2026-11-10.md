# VPS relocation verification

Restored to **2.25.188.2**, Ubuntu 24.04.5 x86_64, on 2026-11-10.

Later that day, the coding container was replaced to support Codex 0.167.1's
nested sandbox. The table below records the initial migration state; see
[sandbox repair](codex-sandbox-repair.md) for current policy and Rex commands.

| Component | Verified result |
| --- | --- |
| SSH | User-confirmed ED25519 fingerprint `SHA256:1gxri8UC/KUY1ObPh9xvp8f+Y2xAbcfzmfKK6ZPGdzM`; strict checking retained |
| Work account | `rexdev`, UID 1001, no administrative groups |
| Rex | 0.1.0-preview.12 personal service, private `/run/user/1001/rex/server.sock`; enabled with lingering |
| Docker | 29.9.3, rootless, user service enabled; Compose 5.7.0 |
| Container | `rex-coding`, ID `3db03ac8d10f189e344220f2c82ecb276a2b4447a0809317d5b52dce49a77426` |
| Persistence | `/home/rexdev/projects` bound at `/workspace`; `rex-coding-home` mounted at `/root` |
| Isolation | 6 GiB RAM, no extra swap, 1.75 CPUs, 1024 PIDs; no privileged mode, host network, Docker socket or published app ports |
| Runtime | Node 24.22.2, Python 3.12.16, uv 0.13.6; original Linux dependencies restored |
| Agents | Codex 0.166.0 and Claude Code 2.1.309 available inside the container and on the host |
| Logins | Codex local login status recognized; Colab and Modal read-only account checks passed; Claude and Sail need login |
| Tests | 18 JavaScript tests and 5 backend tests passed |
| App | Frontend and proxied backend health returned HTTP 200; SSH preview transport passed |
| Rex persistence | Disposable job completed after its launching SSH connection closed; test session removed |
| Mac connection | Existing `http://127.0.0.1:17297` tunnel now reaches the new VPS; CLI lists `tinysettler` |
| Sync | `two-way-safe`, ID `sync_GS9jjT7Imk6RQ5UC9v2tmzpNRYexARTsswwE0FqA9mr`; writes/deletions tested both ways, markers removed, no conflicts |

Open the existing VPS connection in Rex and select **tinysettler**. Its shell
pane is already inside `/workspace/TinySettler`; the `app` pane runs
`npm run catan`. Run `codex` in the shell pane to begin work. Additional VPS
panes enter the container with `~/.local/bin/coding`.

The 73-file migration bundle passed SHA256 verification again before upload
and on the replacement VPS. All 112 eligible working files matched the backup
and Mac before reconnecting sync. Only relocation documentation and the sync
helper's new-IP overlap check were changed during migration.

The old sync session was archived and paused before changing the alias, then
replaced with a fresh session after the endpoints matched. Mac rollback files
are in `~/.local/state/rex-coding/before-new-vps-20261110T205750Z/`.
Current sync registries, `hostinger` and `rex-coding-sync` SSH aliases,
the installed `rex-vps` helper, and the Mac launch-agent tunnel now use the new IP.

Original process memory and jobs were not resumed. The old Rex database and
Docker contexts remain in the private backup rather than being overlaid onto
the new services. Saved container credentials and selected host agent tools/state
were restored. No provider compute, inference or paid jobs were started.

The private backup remains at
`~/Backups/TinySettler-VPS/20261110T200455Z` on the Mac and
`/home/rexdev/backups/tinysettler-migration-20261110T200455Z` on the VPS.
Restore scripts and logs are in `/home/rexdev/restore-tools` on the new VPS.

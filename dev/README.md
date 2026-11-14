# Mac / Rex development

Configured on 2026-11-10 and restored to VPS `2.25.188.2` after relocation.
Working files sync directly between
`/Users/alexanderwang/Projects/TinySettler` and
`rex-coding-sync:/home/rexdev/projects/TinySettler`; the container sees
`/workspace/TinySettler`. The Mac filesystem also resolves the lowercase
`projects` spelling. No symlink is involved.

## Run

The VPS host and coding container use the ported [zsh and LazyVim setup](linux-editor.md).
New coding shells use zsh; `n .` opens the current directory in your editor.

Mac, using the existing Python 3.14.7 `.venv` and Node 26.8.1:

```sh
cd /Users/alexanderwang/Projects/TinySettler
npm run catan
```

Open <http://127.0.0.1:5173/catan.html>. The Mac environment was retained.
For a future Node reinstall, use `npm ci` with `package-lock.json`.

In the Mac Rex app, select the VPS remote host and open the **tinysettler**
session. Its `app` pane runs the development server; its other pane is an
interactive container shell in `/workspace/TinySettler`. Rex handles panes
and persistence directly.

For another coding or provider-auth terminal, create a new Rex pane on the
VPS and run:

```sh
~/.local/bin/coding
cd /workspace/TinySettler
```

Run `codex`, provider authentication, or other commands in that shell.
After the Codex sandbox repair, the ready shell and app are in the
**coding-ready** window. See [sandbox repair](codex-sandbox-repair.md) for the
required container policy, verification, and rollback details.
To restart the app after stopping it, use a container shell:

```sh
npm run catan
```

Only one app process should use container ports 5173/8001. Switch between
panes using Rex's UI. Closing the Mac client leaves remote jobs running;
closing the remote pane or session ends its processes. Do not use tmux.

To create a further container pane through the CLI, run **on the VPS host**:

```sh
rex --autostart=false -S /run/user/1001/rex/server.sock split -s tinysettler \
  --cwd /home/rexdev/projects/TinySettler --shell=none -- /home/rexdev/.local/bin/coding
```

To reinstall Linux dependencies after lockfile changes, stop the app first
and run `bash scripts/setup-linux.sh` inside the container. It requires
Node 24 and Python 3.12, uses `npm ci` and `uv pip sync` with the checked
locks, and refuses to replace a venv of a different Python version.
Linux currently has Node 24.22.2, Python 3.12.16 and uv 0.13.6.
The Linux `.venv` and `node_modules` are separate from the Mac copies.

The `rex-coding` container was restored from its verified migration image;
its persistent `/root` volume is `rex-coding-home`. Provider tools and saved
credentials were restored there. The original backup remains on the Mac.
Rex stays on the host. A remote Rex terminal job survives Mac
disconnection; processes do **not** resume after container/VPS restart.
The new VPS runs a fresh Rex server and rootless Docker daemon. Previous
processes were not resumed; open a new agent session or use its resume command.
The earlier `tinysettler-dev` tmux app session was stopped and replaced by
the Rex session after confirming that no game was active. Existing unrelated
Rex sessions were retained. Rex's host socket is not mounted into the container;
create panes using the Mac Rex UI or the host CLI.

## Preview the VPS app from the Mac

Stop the Mac app first so localhost port 5173 is free, then run on the Mac:

```sh
cd /Users/alexanderwang/Projects/TinySettler
python3 scripts/rex-preview.py
```

Open <http://127.0.0.1:5173/catan.html>. This forwards TCP through
`rexdev` SSH and `docker exec`; it publishes no VPS/container ports and
copies no SSH keys. It also carries Vite WebSockets. Each browser connection
uses an SSH process, so local development is faster. Ctrl-C closes the preview,
leaving the remote app running. The browser still runs inference on the Mac's
WebGPU/Metal; the VPS serves Vite and Python rules only.

## Sync

On the Mac:

```sh
python3 scripts/rex-sync.py status
python3 scripts/rex-sync.py pause
python3 scripts/rex-sync.py resume
python3 scripts/rex-sync.py flush
```

Session: `tinysettler`, mode: `two-way-safe`. The SSH alias now targets
`rexdev@2.25.188.2` with its independently verified host key. Full intended configuration is
[sync.json](sync.json); live configuration and ID are recorded in
`~/.local/state/rex-coding/tinysettler-sync.json`. The existing launch agent
starts the Mutagen daemon at login and checks it every minute. Sync catches
up after the Mac reconnects. Resolve reported conflicts deliberately; do not
reset/force a session to choose a winner. Pause before editing ignore rules
or changing endpoints; session configuration is immutable.

The old `rex-coding-workspace` was empty on both endpoints, had no conflicts,
was archived and paused, and was then terminated. Its complete JSON and text
configuration, paused state and old registry are in
`~/.local/state/rex-coding/before-tinysettler-20261110T185233Z/`.
`~/RexWorkspace` remains untouched and empty. **Do not rerun the old broad
sync installer**; the new registry intentionally differs so it refuses.
Recreating the old broad session would overlap TinySettler.

Git internals, dependencies, build output, caches, common secrets/agent
credentials and weight/checkpoint formats are excluded. Source assets and
small JSON reports in `catan-results`, `test-results` and `rl_research/results`
remain eligible. See the exact patterns in [sync.json](sync.json).
Put large/new model artifacts outside the synced project; exclusions are not
a universal secret or file-size detector. No Git repository existed on either
side at setup, and none was initialized or committed automatically.

**Ordinary deletions propagate in both directions. Sync is not a backup.**
Use independent backups/Git history and avoid editing the same file on both
machines simultaneously. Each endpoint's future Git history stays separate.

## Providers

| Client | Installed on Mac and container | Entry command | Credential location |
| --- | --- | --- | --- |
| Google Colab | 0.7.5, isolated uv tool | `colab` | `~/.config/colab-cli/` |
| Modal | 1.6.3, isolated uv tool | `modal` | `~/.modal.toml` |
| Sail Research | 0.18.1, official checksummed binary | `sail` | `~/.sail/` |

Executables are in `~/.local/bin` on each machine. Container installations
are under persistent `/root`, not in the project venv. The Sail installer was
downloaded from `https://cli.sailresearch.com/install.sh`, inspected, and run
with `SAIL_INSTALL_DIR=$HOME/.local/bin`; it verified its release checksum.
No similarly named third-party package was installed.

Authenticate separately on each machine, entering codes only in your terminal:

```sh
colab --auth oauth2 sessions
modal setup
sail auth login
```

The explicit Colab OAuth flag uses the installed CLI's browser/code flow;
`colab auth` in newer documentation refers to GCP authentication *inside a
runtime*, not this login. Read-only checks after login are
`colab --auth oauth2 sessions`, `modal token info` and `sail box list`.
Do not paste raw credential/token output into reports or sync credential files.
On the restored VPS, Codex recognizes its saved login and Colab/Modal passed
read-only account checks. Claude and Sail require login. See the current
[relocation verification](migration-2026-11-10.md); the earlier
[verification](verification.md) is historical.

Colab supplies temporary runtimes with quota/lifetime limits, not a persistent
replacement for this VPS. Modal runs functions/apps; durable files require a
Modal Volume or external storage, not a container's temporary filesystem.
Sail Research offers an inference API and Sailboxes (managed machines with
lifecycle/checkpoint controls). Its Tinker integration supplies sampling for
Tinker training; installing `sail` does not install a TinySettler trainer.
None of these clients makes browser WebGPU inference headless.

Initial setup started no GPU/CPU cloud jobs, inference requests, paid resources
or purchases. The subsequent user-requested CPU RL checks passed on Colab and
Modal, and both temporary jobs were stopped; see the
[provider RL verification](provider-rl-verification-2026-11-10.md) for results,
limits, and repeat commands. The optional simulator templates below have not
been run and require explicit compute approval before execution.

## Optional CPU benchmark templates

The original CPU provider verification used a simulator benchmark and a small
dependency-free tabular RL smoke test. The subsequent 2026-11-11
[Catan training entry points](../rl_research/overnight/README.md) and
[completed T4 ablation](../rl_research/results/overnight-20261111/findings.md)
are documented separately.
[modal_benchmark.py](modal_benchmark.py) uploads only the backend rules module,
benchmark script and locked dependencies, with two CPUs, one container and a
three-minute function timeout. It requests no GPU and creates no Volume.
After approval, from the project root: `modal run dev/modal_benchmark.py`.
The report returns to a unique file in `rl_research/results/` and then syncs.
The template was imported locally on both platforms; no cloud run was tested.

For an explicitly approved Colab CPU run, the following uses a **new dedicated**
session (choose an unused name); run from the project root:

```sh
colab --auth oauth2 new -s tinysettler-bench
printf 'from pathlib import Path; Path("/content/tinysettler/backend").mkdir(parents=True, exist_ok=True); Path("/content/tinysettler/rl_research").mkdir(exist_ok=True)\n' | colab --auth oauth2 exec -s tinysettler-bench
colab --auth oauth2 upload -s tinysettler-bench backend/catan_server.py /content/tinysettler/backend/catan_server.py
colab --auth oauth2 upload -s tinysettler-bench rl_research/benchmark.py /content/tinysettler/rl_research/benchmark.py
colab --auth oauth2 install -s tinysettler-bench -r backend/requirements-lock.txt
colab --auth oauth2 exec -s tinysettler-bench -f dev/colab_benchmark.py
colab --auth oauth2 download -s tinysettler-bench /content/tinysettler/benchmark.json "rl_research/results/colab-$(date -u +%Y%m%dT%H%M%SZ).json"
colab --auth oauth2 stop -s tinysettler-bench
```

Stop that session even if an intermediate step fails. Colab commands were
checked against installed help; the remote run remains untested. Reports add
current launch metadata while retaining the original benchmark's research date.
Sail access is through <https://app.sailresearch.com/dashboard> and its CLI;
no Sail inference or machine-creation template is connected to the browser app.

## Primary documentation

[Colab CLI](https://github.com/googlecolab/google-colab-cli) ·
[Colab limits](https://research.google.com/colaboratory/faq.html) ·
[Modal setup](https://modal.com/docs/guide/getting-started) ·
[Modal apps](https://modal.com/docs/guide/apps) ·
[Modal volumes](https://modal.com/docs/guide/volumes) ·
[Sail CLI](https://docs.sailresearch.com/reference/cli) ·
[Sailboxes](https://docs.sailresearch.com/sailboxes) ·
[Sail inference / Tinker](https://docs.sailresearch.com/tinker).

[Rex CLI and session/pane model](https://www.superlogical.com/rex/docs/automate/cli).

# Codex sandbox repair

Verified on VPS `2.25.188.2`, 2026-11-10, with Codex 0.167.1.

Docker's default seccomp policy blocked the nested namespace operations needed
by Codex's bundled Bubblewrap helper. User namespaces were enabled on the host;
the inherited RootlessKit AppArmor profile already permitted them. This was
not a Colab or Modal authentication error.

The coding container was replaced with user approval. Its writable filesystem
was saved as `rex-coding:sandbox-fix-20261111` before replacement; its existing
`rex-coding-home` volume and workspace bind mount were retained. The image tag's
date is an identifier, not the verification date. Rex and Docker services were
not restarted. Agent process memory ended; saved conversations and credentials
remain available.

## Applied configuration

On the VPS, `~/.local/share/rex-coding/compose.json` uses the saved image and
`sandbox-fix/seccomp-codex.json` via `security_opt`. The profile was derived
from this container's exact OCI runtime seccomp policy, retaining its default
deny action and existing rules. The changes allow user/mount/IPC/UTS/network/PID
namespace creation and `mount`, `umount2`, and `pivot_root`, which Bubblewrap
needs for its inner sandbox. Cgroup namespace creation remains blocked.

This expands the container's allowed syscall surface only for nested sandbox
support. Rootless execution, no-new-privileges, dropped NET_RAW, resource limits,
and existing mounts remain in place. No SYS_ADMIN capability, privileged mode,
unconfined seccomp, global sysctl change, or Codex sandbox bypass was added.

Rollback files are under `~/.local/share/rex-coding/sandbox-fix/`:
`compose-before.json`, `Dockerfile-before`, `container-before.json`, and
`seccomp-original.json`. The additional pre-repair image is
`rex-coding:before-sandbox-fix-20261111`. Restoring the old policy also restores
the Codex failure. Container recreation must be coordinated with running jobs.

Keep the custom profile when recreating this container. On a Docker upgrade,
review it against the new default policy rather than carrying an old baseline
forward indefinitely. The saved image includes the installed Codex update;
the older Dockerfile alone does not reproduce that image.

## Verification and use

The sandbox executed Python 3.12.16 and Node 24.22.2, and passed workspace
write/read/delete tests while rejecting a write to `/etc`. The same test passed
in a disposable container without project or credential mounts before rollout.
Codex's saved ChatGPT login survived. Colab `sessions` and Modal `app list`
returned success; neither command provisions compute. Frontend and proxied
backend health returned HTTP 200. No RL job, GPU instance, or paid inference
was launched during this repair.

In Rex, open **Hostinger VPS → tinysettler → coding-ready**. In its `coding`
pane, run `codex resume --last` to reopen the saved conversation, or `codex`
for a fresh one. Its `app` pane runs TinySettler. An older `development` window
was retained because it still contained a host shell.

A no-model-call sandbox diagnostic inside the coding container is:

```sh
cd /workspace/TinySettler
codex sandbox -c 'sandbox_mode="workspace-write"' -- pwd
```

References: [OpenAI sandbox prerequisites](https://learn.chatgpt.com/docs/sandboxing#prerequisites)
and [Docker seccomp profiles](https://docs.docker.com/engine/security/seccomp/).

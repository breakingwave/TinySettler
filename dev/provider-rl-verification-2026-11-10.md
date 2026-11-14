# Provider authentication and RL verification

Verified from the restored VPS coding container on 2026-11-10, around 22:04 UTC.
The requested tiny cloud runs authorized the compute used for this check.

| Provider | Authentication | Remote training | Cleanup |
| --- | --- | --- | --- |
| Google Colab | `colab --auth oauth2 sessions` succeeded | Passed, Python 3.13.18, CPU | Runtime terminated; subsequent server listing had no active sessions |
| Modal | `modal token info` succeeded for workspace `aarcheba` | Passed, Python 3.12.10, CPU | App `ap-h0BFC4aeJBOVCajFrdcjwE` stopped; subsequent listing reported zero tasks |

Both executed [rl_smoke.py](rl_smoke.py): dependency-free tabular Q-learning
in a five-state corridor, 300 training episodes, seed 42. Greedy evaluation
improved from 0/100 to 100/100 successful episodes on each provider.
The training/evaluation routine took about 3.7 ms on Colab and 4.5 ms on Modal;
these figures exclude provisioning, network and CLI overhead.

Results:

- [Colab JSON](../rl_research/results/colab-rl-smoke-20261110.json), recorded from the successful CLI output.
- [Modal JSON](../rl_research/results/modal-rl-smoke-20261110T220351786614Z.json), returned by the remote function and saved by the local entrypoint.
- [Modal run](https://modal.com/apps/aarcheba/main/ap-h0BFC4aeJBOVCajFrdcjwE).

Repeat from the project root with a new unused Colab session name:

```sh
colab --auth oauth2 run --session tinysettler-rl-smoke-UNIQUE --timeout 60 dev/rl_smoke.py
modal run dev/modal_rl_smoke.py
```

Colab uses a fresh standard CPU runtime and releases it on completion. Modal
requests 0.25 CPU, 128 MiB, one container, no retries and a 60-second function
timeout. Neither run requested a GPU or persistent storage. Modal uploads the
entrypoint and the smoke-test source, without project data or credentials.
No purchases or account/billing changes were made; billed usage was not queried.

These checks establish authenticated CPU provisioning, source execution,
learning, result retrieval and cleanup. GPU availability, ML frameworks,
long-running training, and a TinySettler game-training pipeline remain untested.

The CLIs needed execution outside the Codex sandbox: Colab writes logs and
session state under `/root/.config/colab-cli`, and the sandbox blocked service
access. Authentication succeeded using the existing saved credentials; no
credential replacement was needed or saved in the project.

The commands were checked against installed CLI help and the official
[Colab CLI](https://github.com/googlecolab/google-colab-cli) and
[Modal setup documentation](https://modal.com/docs/guide/getting-started).

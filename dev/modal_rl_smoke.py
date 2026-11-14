"""Run the dependency-free RL access check on one temporary Modal CPU worker.

Usage: modal run dev/modal_rl_smoke.py
Only rl_smoke.py is uploaded; no GPU, Volume, secrets, or deployment is used.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    ROOT / "dev/rl_smoke.py", "/opt/rl_smoke.py"
)
app = modal.App("tinysettler-rl-smoke")


@app.function(
    image=image, cpu=0.25, memory=128, timeout=60,
    max_containers=1, retries=0,
)
def train():
    import runpy

    return runpy.run_path("/opt/rl_smoke.py")["run"]()


@app.local_entrypoint()
def main():
    report = train.remote()
    now = datetime.now(timezone.utc)
    report["launch"] = {
        "provider": "modal", "executed_at": now.isoformat(),
        "kind": "tabular RL CPU smoke test",
    }
    directory = ROOT / "rl_research/results"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("modal-rl-smoke-" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(output)

"""Optional paid CPU benchmark template; import is local, modal run uses cloud compute.

Run only after explicit compute approval:
  modal run dev/modal_benchmark.py
This runs the existing simulator benchmark, not RL training or browser inference.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "backend/requirements-lock.txt"))
    .add_local_file(ROOT / "backend/catan_server.py", "/opt/tinysettler/backend/catan_server.py")
    .add_local_file(ROOT / "rl_research/benchmark.py", "/opt/tinysettler/rl_research/benchmark.py")
)
app = modal.App("tinysettler-cpu-benchmark")


@app.function(image=image, cpu=2, memory=2048, timeout=180, max_containers=1)
def benchmark():
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "benchmark.json"
        subprocess.run(
            ["python", "/opt/tinysettler/rl_research/benchmark.py", "--repetitions", "1",
             "--output", str(output)], check=True, timeout=120,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        return json.loads(output.read_text())


@app.local_entrypoint()
def main():
    report = benchmark.remote()
    # The original benchmark embeds its original research date. Preserve it and
    # add explicit metadata so a new cloud run cannot masquerade as that result.
    now = datetime.now(timezone.utc)
    report["launch"] = {"provider": "modal", "executed_at": now.isoformat(),
                        "repetitions": 1, "kind": "simulator CPU benchmark, no training"}
    output = ROOT / "rl_research/results" / ("modal-" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)

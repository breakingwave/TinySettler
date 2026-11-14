"""Execute after uploading the three allowlisted files and installing the lock.

This is a bounded CPU simulator benchmark, not a trainer. No provisioning here.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

root = Path("/content/tinysettler")
output = root / "benchmark.json"
subprocess.run(
    ["python", str(root / "rl_research/benchmark.py"), "--repetitions", "1",
     "--output", str(output)], check=True, timeout=120,
    env={**os.environ, "PYTHONHASHSEED": "0"},
)
report = json.loads(output.read_text())
report["launch"] = {"provider": "colab", "executed_at": datetime.now(timezone.utc).isoformat(),
                    "repetitions": 1, "kind": "simulator CPU benchmark, no training"}
output.write_text(json.dumps(report, indent=2) + "\n")
print(output)

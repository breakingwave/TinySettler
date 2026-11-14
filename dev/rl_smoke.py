"""Dependency-free, bounded Q-learning smoke test; runs locally or in Colab."""
import json
import os
import platform
import random
import shutil
import subprocess
import time


def run():
    started = time.monotonic()
    rng = random.Random(42)
    # Five-state corridor: start at 0, reward +1 on reaching terminal state 4.
    q = [[0.0, 0.0] for _ in range(5)]

    def evaluate():
        wins = 0
        for _ in range(100):
            state = 0
            for _ in range(12):
                action = max(range(2), key=lambda a: q[state][a])
                state = max(0, min(4, state + (1 if action else -1)))
                if state == 4:
                    wins += 1
                    break
        return wins / 100

    before = evaluate()
    for _ in range(300):
        state = 0
        for _ in range(32):
            action = rng.randrange(2) if rng.random() < 0.3 else max(range(2), key=lambda a: q[state][a])
            following = max(0, min(4, state + (1 if action else -1)))
            done = following == 4
            reward = 1.0 if done else -0.01
            target = reward + (0.0 if done else 0.95 * max(q[following]))
            q[state][action] += 0.3 * (target - q[state][action])
            state = following
            if done:
                break
    after = evaluate()
    assert before == 0.0 and after == 1.0, (before, after)
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu = {"returncode": result.returncode, "devices": result.stdout.strip()}
    return {
        "task": "tabular Q-learning, five-state corridor",
        "training_episodes": 300,
        "evaluation_episodes": 100,
        "seed": 42,
        "success_before": before,
        "success_after": after,
        "passed": True,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_probe": gpu,
        "training_device": "cpu",
        "elapsed_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

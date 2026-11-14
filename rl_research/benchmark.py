"""Bounded, standard-library-only Catanatron throughput ablations.

Run from the repository root with PYTHONHASHSEED=0 .venv/bin/python
rl_research/benchmark.py --output rl_research/results/benchmark.json
No training, server mutation, or dependency installation.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from catanatron import Game, Color
from catanatron.features import create_sample, get_feature_ordering
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionType
from catanatron.players.weighted_random import WeightedRandomPlayer, WEIGHTS_BY_ACTION_TYPE
from backend.catan_server import CatanBridge, REVISION


class AllocationFreeWeightedPlayer(WeightedRandomPlayer):
    """Same integer draw and action distribution as stock, without repeated lists."""
    def decide(self, game, actions):
        weights = [WEIGHTS_BY_ACTION_TYPE.get(a.action_type, 1) for a in actions]
        index = game.state.random.randrange(sum(weights))
        for action, weight in zip(actions, weights):
            if index < weight:
                return action
            index -= weight
        raise AssertionError("unreachable")


def isolated_copy(game):
    result = game.copy()
    rng = random.Random(0)
    rng.setstate(game.state.random.getstate())
    result.random = result.state.random = rng
    return result


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def describe_us(values):
    return {"n": len(values), "median_us": statistics.median(values) * 1e6,
            "p95_us": percentile(values, .95) * 1e6}


def episode(seed, nplayers, cls, observation, deadline, capture=False):
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:nplayers]
    started = time.perf_counter()
    game = Game([cls(c) for c in colors], seed=seed)
    initial_seconds = time.perf_counter() - started
    features = tuple(f for f in get_feature_ordering(nplayers)
                     if f not in {"BANK_WOOD", "BANK_BRICK", "BANK_SHEEP", "BANK_WHEAT", "BANK_ORE"})
    policy_seconds = execute_seconds = observation_seconds = capture_seconds = 0.0
    counts = []
    snapshots = []
    types = Counter()
    digest = hashlib.sha256()
    decisions = red_decisions = 0
    started = time.perf_counter()
    while game.winning_color() is None and game.state.num_turns < 1000 and decisions < 20000:
        if time.perf_counter() >= deadline:
            raise TimeoutError("90-second global benchmark budget exceeded")
        actions = game.playable_actions
        counts.append(len(actions))
        red_decisions += game.state.current_color() == Color.RED
        if capture and decisions % 40 == 0 and len(snapshots) < 32:
            t = time.perf_counter()
            snapshots.append(isolated_copy(game))
            capture_seconds += time.perf_counter() - t
        if observation:
            t = time.perf_counter()
            sample = create_sample(game, game.state.current_color())
            vector = [float(sample[f]) for f in features]
            assert len(vector) == len(features)
            observation_seconds += time.perf_counter() - t
        t = time.perf_counter()
        action = game.state.current_player().decide(game, actions)
        policy_seconds += time.perf_counter() - t
        t = time.perf_counter()
        record = game.execute(action)
        execute_seconds += time.perf_counter() - t
        digest.update(repr(record).encode())
        types[action.action_type.value] += 1
        decisions += 1
    elapsed = time.perf_counter() - started - capture_seconds
    winner = game.winning_color()
    return {"seed": seed, "players": nplayers, "policy": cls.__name__,
            "numeric_observation_every_action": observation,
            "engine_actions": decisions, "red_decisions_including_forced": red_decisions,
            "completed_player_turns": game.state.num_turns,
            "winner": winner.value if winner else None, "truncated": winner is None,
            "elapsed_seconds": elapsed, "initialization_seconds": initial_seconds,
            "actions_per_second": decisions / elapsed,
            "policy_seconds": policy_seconds, "execute_seconds": execute_seconds,
            "observation_seconds": observation_seconds,
            "legal_count_median": statistics.median(counts),
            "legal_count_p95": percentile(counts, .95), "legal_count_max": max(counts),
            "trace_sha256": digest.hexdigest(), "action_counts": dict(types)}, snapshots


def kernels(snapshots):
    measurements = {k: [] for k in ["legal_action_generation", "game_copy_shared_rng",
                    "game_copy_isolated_rng", "copy_and_execute_isolated_rng",
                    "numeric_vector_no_bank", "bridge_snapshot_with_prompt", "bridge_snapshot_json"]}
    for game in snapshots:
        features = tuple(f for f in get_feature_ordering(len(game.state.colors))
                         if f not in {"BANK_WOOD", "BANK_BRICK", "BANK_SHEEP", "BANK_WHEAT", "BANK_ORE"})
        def encode():
            sample = create_sample(game, game.state.current_color())
            return [float(sample[f]) for f in features]
        # Snapshot formatting is measured on both player counts. Its prompt says
        # four-player, but nothing here uses that text to play or train.
        bridge = CatanBridge.__new__(CatanBridge)
        bridge.game = game
        bridge.game_id = "benchmark"
        bridge.seed = game.seed
        bridge.max_turns = 1000
        bridge.auto_forced = False
        bridge.revision = 0
        bridge.stats = {"engineActions": 0, "llmDecisions": 0, "forcedActions": 0,
                        "opponentActions": 0, "redActionCounts": {}}
        bridge.events = []
        bridge.history = []
        functions = {
            "legal_action_generation": lambda: generate_playable_actions(game.state),
            "game_copy_shared_rng": game.copy,
            "game_copy_isolated_rng": lambda: isolated_copy(game),
            "copy_and_execute_isolated_rng": lambda: isolated_copy(game).execute(game.playable_actions[0]),
            "numeric_vector_no_bank": encode,
            "bridge_snapshot_with_prompt": bridge.snapshot,
            "bridge_snapshot_json": lambda: json.dumps(bridge.snapshot()),
        }
        before = game.state.random.getstate()
        for name, fn in functions.items():
            fn()  # warm caches; excluded
            for _ in range(5):
                t = time.perf_counter()
                fn()
                measurements[name].append(time.perf_counter() - t)
        assert game.state.random.getstate() == before
    return {k: describe_us(v) for k, v in measurements.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 5:
        parser.error("repetitions must be between 1 and 5")
    start = time.perf_counter()
    deadline = start + 90
    # Warm initialization and module-level feature caches; not included in runs.
    for players in [2, 4]:
        get_feature_ordering(players)
    rows, samples = [], {2: [], 4: []}
    for repetition in range(args.repetitions):
        for players in [2, 4]:
            for seed in range(610, 616):
                paired = []
                variants = [(WeightedRandomPlayer, False),
                            (AllocationFreeWeightedPlayer, False),
                            (AllocationFreeWeightedPlayer, True)]
                rotation = (repetition + seed) % len(variants)
                variants = variants[rotation:] + variants[:rotation]
                for cls, observation in variants:
                    row, snapshots = episode(seed, players, cls, observation, deadline,
                                             capture=repetition == 0 and cls is WeightedRandomPlayer)
                    row["repetition"] = repetition
                    rows.append(row)
                    samples[players].extend(snapshots)
                    paired.append(row["trace_sha256"])
                assert len(set(paired)) == 1, "ablation changed game trajectory"
    summaries = []
    for players in [2, 4]:
        for cls, obs in [(WeightedRandomPlayer, False), (AllocationFreeWeightedPlayer, False),
                         (AllocationFreeWeightedPlayer, True)]:
            group = [r for r in rows if r["players"] == players and r["policy"] == cls.__name__
                     and r["numeric_observation_every_action"] == obs]
            seconds = sum(r["elapsed_seconds"] for r in group)
            actions = sum(r["engine_actions"] for r in group)
            summaries.append({"players": players, "policy": cls.__name__, "numeric_observation": obs,
                              "games": len(group), "total_actions": actions,
                              "total_seconds": seconds, "aggregate_actions_per_second": actions / seconds,
                              "median_game_actions_per_second": statistics.median(r["actions_per_second"] for r in group),
                              "red_decisions_per_second": sum(r["red_decisions_including_forced"] for r in group) / seconds,
                              "policy_share": sum(r["policy_seconds"] for r in group) / seconds,
                              "execute_share": sum(r["execute_seconds"] for r in group) / seconds,
                              "observation_share": sum(r["observation_seconds"] for r in group) / seconds,
                              "truncations": sum(r["truncated"] for r in group)})
    kernel_results = {str(n): kernels(states[:64]) for n, states in samples.items()}
    profile = cProfile.Profile()
    profile.enable()
    episode(616, 2, AllocationFreeWeightedPlayer, True, deadline)
    profile.disable()
    stats = pstats.Stats(profile)
    top = sorted(stats.stats.items(), key=lambda x: x[1][2], reverse=True)[:20]
    profile_rows = [{"file": k[0], "line": k[1], "function": k[2], "calls": v[1],
                     "self_seconds": v[2], "cumulative_seconds": v[3]} for k, v in top]
    output = {"metadata": {"date": "2026-11-04", "python": sys.version,
                           "platform": platform.platform(), "machine": platform.machine(),
                           "hash_seed": os.getenv("PYTHONHASHSEED"), "catanatron_commit": REVISION,
                           "rules": "BASE board, 10 VP, discard 7, no player offers; 1000 player-turn / 20000 action cap",
                           "seeds": list(range(610, 616)), "timing": "single-process warm-cache perf_counter; instrumented; no HTTP/inference/training",
                           "repetitions": args.repetitions, "variant_order_rotated": True,
                           "all_paired_traces_equal": True,
                           "wall_seconds": time.perf_counter() - start},
              "summary": summaries, "kernels": kernel_results, "games": rows,
              "profile_top_self_time": profile_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "kernels": kernel_results, "wall_seconds": output["metadata"]["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

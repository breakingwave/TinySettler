"""Bounded, dependency-free audit of the installed Catanatron RL interfaces.

Run with the project Python. The only file written is an explicitly requested
--output path. This does not patch Catanatron or the browser bridge.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import random

import catanatron
from catanatron import Color, Game, RandomPlayer
from catanatron.features import create_sample, get_feature_ordering
from catanatron.game import is_valid_trade
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import get_edges
from catanatron.models.enums import ActionType, RESOURCES
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.state_functions import player_key


def isolated_copy(game):
    copied = game.copy()
    rng = random.Random()
    rng.setstate(game.state.random.getstate())
    copied.random = copied.state.random = rng
    return copied


def first_roll_game(seed):
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    steps = 0
    while not any(a.action_type == ActionType.ROLL for a in game.playable_actions):
        if steps >= 128:
            raise RuntimeError("Could not reach first roll within audit action cap")
        game.execute(game.playable_actions[0])
        steps += 1
    return game, steps


def audit(seed):
    package = Path(catanatron.__file__).resolve().parent
    game, setup_actions = first_roll_game(seed)
    actor = game.state.current_color()
    enemy = next(color for color in game.state.colors if color != actor)
    roll_action = next(a for a in game.playable_actions if a.action_type == ActionType.ROLL)

    # A disposable baseline demonstrates the installed copy semantics directly.
    baseline = isolated_copy(game)
    copied = baseline.copy()
    rng_before = baseline.state.random.getstate()
    copied.execute(roll_action)
    rng_probe = {
        "copy_shares_rng": copied.state.random is baseline.state.random,
        "copy_stochastic_action_changes_original_rng": rng_before != baseline.state.random.getstate(),
        "original_action_count_unchanged": len(baseline.state.action_records) == setup_actions,
    }

    # Isolate from the live baseline before probing the existing spectrum helper.
    spectrum_root = isolated_copy(game)
    live_rng_before = game.state.random.getstate()
    outcomes = execute_spectrum(spectrum_root, roll_action)
    intended = list(range(2, 13))
    realized = [sum(state.state.action_records[-1].result) for state, _ in outcomes]
    chance_probe = {
        "intended_roll_sums": intended,
        "realized_roll_sums": realized,
        "assigned_probabilities": [probability for _, probability in outcomes],
        "exact_enumeration_matches": intended == realized,
        "audit_baseline_rng_unchanged": game.state.random.getstate() == live_rng_before,
        "cause": "execute_spectrum supplies dice in Action.value; apply_roll honors only ActionRecord.result, otherwise samples RNG",
    }

    sample = create_sample(game, actor)
    enemy_key = player_key(game.state, enemy)
    reconstructed = {r: 19 - sample[f"BANK_{r}"] - sample[f"P0_{r}_IN_HAND"] for r in RESOURCES}
    actual = {r: game.state.player_state[f"{enemy_key}_{r}_IN_HAND"] for r in RESOURCES}

    # Swap one opponent card with the bank, preserving physical card totals,
    # actor hand, opponent total hand size, buildings, phase and history.
    mutated = isolated_copy(game)
    source = next(r for r in RESOURCES if actual[r] > 0)
    destination = next(r for r in RESOURCES if r != source and sample[f"BANK_{r}"] > 0)
    mutated.state.player_state[f"{enemy_key}_{source}_IN_HAND"] -= 1
    mutated.state.player_state[f"{enemy_key}_{destination}_IN_HAND"] += 1
    mutated.state.resource_freqdeck[RESOURCES.index(source)] += 1
    mutated.state.resource_freqdeck[RESOURCES.index(destination)] -= 1
    mutated.playable_actions = generate_playable_actions(mutated.state)
    changed_sample = create_sample(mutated, actor)
    changed_features = sorted(key for key in sample if sample[key] != changed_sample[key])
    bank_probe = {
        "resource_supply_assumption": "19 cards per resource, exactly two players, all resources in bank or player hands",
        "reconstructed_opponent_hand": reconstructed,
        "actual_opponent_hand": actual,
        "exact_reconstruction": reconstructed == actual,
        "mutated_one_card": {"from": source, "to": destination},
        "changed_feature_names": changed_features,
        "nonbank_features_unchanged": all(key.startswith("BANK_") for key in changed_features),
        "legal_actions_unchanged": mutated.playable_actions == game.playable_actions,
        "card_conservation": {
            r: mutated.state.resource_freqdeck[i] + sum(
                mutated.state.player_state[f"{player_key(mutated.state, c)}_{r}_IN_HAND"]
                for c in mutated.state.colors
            ) == 19
            for i, r in enumerate(RESOURCES)
        },
        "interpretation": "Exact bank composition reveals the opponent resource hand in 1v1; this is leakage only if the chosen demo protocol does not expose bank composition.",
    }

    # Load only this dependency-free file; importing the gym package would
    # require optional Gymnasium/NumPy dependencies absent in this environment.
    spec = importlib.util.spec_from_file_location("audit_catan_action_space", package / "gym/envs/action_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    colors = (Color.BLUE, Color.RED)
    actions = module.get_action_array(colors, "BASE")
    mapping_ok = all(
        module.to_action_space(module.from_action_space(i, actor, colors, "BASE"), colors, "BASE") == i
        for i in range(len(actions))
    )
    features = get_feature_ordering(2, "BASE")
    action_types = sorted({kind.value for kind, _ in actions})
    source_files = ["game.py", "state.py", "features.py", "apply_action.py", "gym/envs/action_space.py", "players/tree_search_utils.py"]
    return {
        "seed": seed,
        "python": platform.python_version(),
        "catanatron_version": importlib.metadata.version("catanatron"),
        "package_path": str(package),
        "scope": "Tiny read-only interface probes, not policy training or playing-strength evaluation",
        "setup_engine_actions": setup_actions,
        "optional_dependencies_available": {name: importlib.util.find_spec(name) is not None for name in ["numpy", "gymnasium", "torch", "onnx"]},
        "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in source_files},
        "dimensions": {
            "players": 2,
            "map": "BASE",
            "land_tiles": len(game.state.board.map.land_tiles),
            "nodes": len(game.state.board.map.land_nodes),
            "edges": len(get_edges(game.state.board.map.land_nodes)),
            "numeric_features": len(features),
            "fixed_actions": len(actions),
            "action_mapping_roundtrip_all_pass": mapping_ok,
            "action_types": action_types,
            "player_trade_actions_present": any("TRADE" in name and name != "MARITIME_TRADE" for name in action_types),
            "dice_features_encode_probability_only": [name for name in features if "PROBA" in name],
        },
        "rng_copy": rng_probe,
        "roll_spectrum": chance_probe,
        "bank_information": bank_probe,
        "trade_input_validation": {
            "negative_counts_pass_shape_check": is_valid_trade((-1, 2, 0, 0, 0, 0, 0, 1, 0, 0)),
            "fractional_counts_pass_shape_check": is_valid_trade((0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0)),
            "interpretation": "Offer adapter must check length, integer nonnegative counts, affordability and recipient; the stock helper does not establish these invariants.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.seed)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

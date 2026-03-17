"""
Matchup Configurations
=======================
Defines all model × model, model × strategy, and self-play matchups.
"""

from __future__ import annotations
from itertools import combinations

from config.models import ALL_MODELS, PILOT_MODELS
from environments.games import ALL_GAMES, PILOT_GAMES, GAME_REGISTRY
from environments.strategies import get_strategies_for_game


def generate_strategy_matchups(
    model_set: dict = None,
    game_set: list = None,
    condition: str = "baseline",
    num_trials: int = 20,
) -> list[dict]:
    """
    Generate all model-vs-strategy matchup configurations.

    Returns list of dicts:
    {
        "model_key": str,
        "model_cfg": dict,
        "game_id": str,
        "strategy_name": str,
        "condition": str,
        "num_trials": int,
    }
    """
    if model_set is None:
        model_set = ALL_MODELS
    if game_set is None:
        game_set = ALL_GAMES

    matchups = []
    for game_cfg in game_set:
        game_id = game_cfg["game_id"]
        strategies = get_strategies_for_game(game_cfg)

        for model_key, model_cfg in model_set.items():
            for strat_name in strategies:
                matchups.append({
                    "model_key": model_key,
                    "model_cfg": model_cfg,
                    "game_id": game_id,
                    "strategy_name": strat_name,
                    "condition": condition,
                    "num_trials": num_trials,
                    "matchup_type": "model_vs_strategy",
                })

    return matchups


def generate_self_play_matchups(
    model_set: dict = None,
    game_set: list = None,
    condition: str = "baseline",
    num_trials: int = 20,
) -> list[dict]:
    """
    Generate all self-play matchup configurations (model plays itself).
    """
    if model_set is None:
        model_set = ALL_MODELS
    if game_set is None:
        game_set = ALL_GAMES

    matchups = []
    for game_cfg in game_set:
        game_id = game_cfg["game_id"]
        for model_key, model_cfg in model_set.items():
            matchups.append({
                "model_key_p0": model_key,
                "model_cfg_p0": model_cfg,
                "model_key_p1": model_key,
                "model_cfg_p1": model_cfg,
                "game_id": game_id,
                "condition": condition,
                "num_trials": num_trials,
                "matchup_type": "self_play",
            })

    return matchups


def generate_cross_play_matchups(
    model_set: dict = None,
    game_set: list = None,
    condition: str = "baseline",
    num_trials: int = 10,
    pairs: list[tuple[str, str]] = None,
) -> list[dict]:
    """
    Generate model-vs-model cross-play matchup configurations.

    Args:
        pairs: If provided, only these (key_a, key_b) pairs are generated.
               If None, all C(n,2) combinations of model_set are used.
    """
    if model_set is None:
        model_set = ALL_MODELS
    if game_set is None:
        game_set = ALL_GAMES

    if pairs is not None:
        pair_list = pairs
    else:
        model_keys = list(model_set.keys())
        pair_list = list(combinations(model_keys, 2))

    matchups = []

    for game_cfg in game_set:
        game_id = game_cfg["game_id"]
        for key_a, key_b in pair_list:
            matchups.append({
                "model_key_p0": key_a,
                "model_cfg_p0": model_set[key_a],
                "model_key_p1": key_b,
                "model_cfg_p1": model_set[key_b],
                "game_id": game_id,
                "condition": condition,
                "num_trials": num_trials,
                "matchup_type": "cross_play",
            })

    return matchups


def count_trials(
    model_set: dict = None,
    game_set: list = None,
    conditions: list[str] = None,
    include_strategy: bool = True,
    include_self_play: bool = True,
    include_cross_play: bool = False,
    num_trials: int = 20,
) -> dict:
    """
    Count total trials for a given configuration.
    Returns breakdown by matchup type and total.
    """
    if model_set is None:
        model_set = ALL_MODELS
    if game_set is None:
        game_set = ALL_GAMES
    if conditions is None:
        conditions = ["baseline"]

    n_models = len(model_set)
    n_games = len(game_set)
    n_conditions = len(conditions)

    counts = {
        "n_models": n_models,
        "n_games": n_games,
        "n_conditions": n_conditions,
    }

    total = 0

    if include_strategy:
        # Count strategies per game (average)
        total_strat_matchups = 0
        for game_cfg in game_set:
            n_strats = len(get_strategies_for_game(game_cfg))
            total_strat_matchups += n_models * n_strats

        strat_trials = total_strat_matchups * num_trials * n_conditions
        counts["strategy_matchups"] = total_strat_matchups
        counts["strategy_trials"] = strat_trials
        total += strat_trials

    if include_self_play:
        self_matchups = n_models * n_games
        self_trials = self_matchups * num_trials * n_conditions
        counts["self_play_matchups"] = self_matchups
        counts["self_play_trials"] = self_trials
        total += self_trials

    if include_cross_play:
        cross_matchups = (n_models * (n_models - 1) // 2) * n_games
        cross_trials = cross_matchups * num_trials * n_conditions
        counts["cross_play_matchups"] = cross_matchups
        counts["cross_play_trials"] = cross_trials
        total += cross_trials

    counts["total_trials"] = total

    # Estimate cost (rough: ~500 input tokens + ~50 output tokens per round,
    # 10 rounds per match, median pricing ~$1/M input, ~$5/M output)
    avg_input_per_match = 500 * 10  # tokens
    avg_output_per_match = 50 * 10
    avg_cost_per_match = (avg_input_per_match * 1.0 + avg_output_per_match * 5.0) / 1_000_000
    counts["estimated_cost_usd"] = round(total * avg_cost_per_match, 2)

    return counts


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def pilot_config() -> dict:
    """Minimal config for testing: 5 models, 8 games, 1 trial."""
    return {
        "model_set": PILOT_MODELS,
        "game_set": PILOT_GAMES,
        "conditions": ["baseline"],
        "num_trials": 1,
        "include_strategy": True,
        "include_self_play": True,
        "include_cross_play": False,
    }


def core_config() -> dict:
    """Core experiment: all models, all games, key conditions."""
    from experiment.conditions import CORE_CONDITIONS
    return {
        "model_set": ALL_MODELS,
        "game_set": ALL_GAMES,
        "conditions": CORE_CONDITIONS,
        "num_trials": 20,
        "include_strategy": True,
        "include_self_play": True,
        "include_cross_play": False,
    }


def full_config() -> dict:
    """Full experiment: everything including cross-play."""
    from experiment.conditions import ALL_CONDITIONS
    return {
        "model_set": ALL_MODELS,
        "game_set": ALL_GAMES,
        "conditions": ALL_CONDITIONS,
        "num_trials": 20,
        "include_strategy": True,
        "include_self_play": True,
        "include_cross_play": True,
    }


def crossplay_config() -> dict:
    """Cross-play only: model vs model."""
    return {
        "model_set": ALL_MODELS,
        "game_set": ALL_GAMES,
        "conditions": ["baseline"],
        "num_trials": 10,
        "include_strategy": False,
        "include_self_play": False,
        "include_cross_play": True,
    }

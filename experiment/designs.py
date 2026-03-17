"""
Experiment Design System
=========================
Configurable experiment designs with named presets and full customization.

Usage:
    python run.py --design smoke --dry-run      # See cost for smoke test
    python run.py --design C --dry-run          # Cost for "Maximum" design
    python run.py --design C --trials 5         # Override trial count
    python run.py --design custom --models core_17 --games all --trials 10
    python run.py --list-designs                # Show all presets
    python run.py --design smoke                # Actually run
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from itertools import combinations

from config.models import (
    ALL_MODELS, MODEL_SETS, PRICING,
    get_model_set,
)
from environments.games import ALL_GAMES, GAME_SETS, get_game_set
from environments.strategies import get_strategies_for_game


# ---------------------------------------------------------------------------
# Design dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDesign:
    """Full specification of an experiment run."""
    name: str
    description: str

    # What to include
    model_set: str = "core_17"         # key into MODEL_SETS
    game_set: str = "core_8"           # key into GAME_SETS

    # Trial counts per matchup cell
    strategy_trials: int = 20
    self_play_trials: int = 20
    cross_play_trials: int = 10

    # What matchup types to run
    include_strategy: bool = True
    include_self_play: bool = True
    include_cross_play: bool = True

    # Cross-play scope: which games get model-vs-model play
    # "same" = same as game_set, or a different GAME_SET key
    cross_play_games: str = "same"

    # Conditions
    conditions: list[str] = field(default_factory=lambda: ["baseline"])
    # Robustness conditions: run on a SUBSET of games only
    robustness_conditions: list[str] = field(default_factory=list)
    robustness_games: str = "core_8"
    robustness_trials: int = 10

    # Selective cross-play: restrict which model pairs play cross-play.
    # When None or empty, all C(n,2) pairs are generated (default behavior).
    # When set, maps model_key → list of opponent model_keys.
    # Pairs are deduplicated: {"A": ["B"]} and {"B": ["A"]} produce one pair.
    # Models NOT listed play against ALL other models (full combinations).
    # Example:
    #   selective_cross_play = {
    #       "claude-opus-4.6": ["claude-sonnet-4.6", "gpt-5", "gemini-3.1-pro"],
    #       "gpt-5": ["claude-sonnet-4.6", "gemini-3.1-pro"],
    #   }
    # This means opus and gpt-5 only play against specified opponents,
    # while all other models still play against each other normally.
    selective_cross_play: dict[str, list[str]] | None = None

    # Self-play-only models: these models ONLY participate in self-play.
    # They are excluded from strategy-play and cross-play matchups entirely.
    # Example: self_play_only = {"claude-opus-4.5", "claude-opus-4.6"}
    self_play_only: set[str] | None = None

    def resolve_models(self) -> dict:
        return get_model_set(self.model_set)

    def resolve_games(self) -> list:
        return get_game_set(self.game_set)

    def resolve_cross_play_games(self) -> list:
        if self.cross_play_games == "same":
            return self.resolve_games()
        return get_game_set(self.cross_play_games)

    def resolve_robustness_games(self) -> list:
        return get_game_set(self.robustness_games)

    def resolve_cross_play_pairs(self) -> list[tuple[str, str]] | None:
        """
        Resolve selective cross-play config into a deduplicated list of pairs.

        Returns None if no selective config (meaning: use all combinations).
        Returns a list of (key_a, key_b) tuples otherwise, where each pair
        appears exactly once (sorted order).

        Logic:
        - Models listed in selective_cross_play only play against their
          specified opponents.
        - Models NOT listed play against ALL other models (except they still
          respect the restrictions of listed models).
        """
        if not self.selective_cross_play:
            return None

        models = self.resolve_models()
        model_keys = list(models.keys())
        restricted = set(self.selective_cross_play.keys())
        unrestricted = [k for k in model_keys if k not in restricted]

        pairs = set()

        # Restricted models: only play against their specified opponents
        for model, opponents in self.selective_cross_play.items():
            if model not in models:
                continue
            for opp in opponents:
                if opp not in models:
                    continue
                pair = tuple(sorted([model, opp]))
                pairs.add(pair)

        # Unrestricted models: play against all other unrestricted models
        for key_a, key_b in combinations(unrestricted, 2):
            pairs.add(tuple(sorted([key_a, key_b])))

        # Unrestricted models also play against restricted models
        # ONLY if the restricted model lists them as an opponent
        # (already handled above via the restricted loop)

        return sorted(pairs)

    def with_overrides(self, **kwargs) -> "ExperimentDesign":
        """Return a copy with specified fields overridden."""
        d = copy.deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(d, k):
                raise ValueError(f"Unknown design parameter: {k}")
            setattr(d, k, v)
        return d


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(design: ExperimentDesign) -> dict:
    """
    Compute trial counts and cost estimate for a design.
    Returns detailed breakdown dict.
    """
    models = design.resolve_models()
    games = design.resolve_games()
    spo = design.self_play_only or set()
    n_models = len(models)
    n_strategy_models = sum(1 for k in models if k not in spo)
    n_games = len(games)
    n_conditions = len(design.conditions)

    # Compute average cost per match for this model set
    input_per_match = 500 * 10   # ~500 input tokens × 10 rounds
    output_per_match = 50 * 10   # ~50 output tokens × 10 rounds
    model_costs = {}
    for key, cfg in models.items():
        mid = cfg["model_id"]
        if mid in PRICING:
            inp, out = PRICING[mid]
            model_costs[key] = (input_per_match * inp
                                + output_per_match * out) / 1_000_000
        else:
            model_costs[key] = 0.0075  # fallback
    avg_cost = sum(model_costs.values()) / max(len(model_costs), 1)

    result = {
        "design_name": design.name,
        "n_models": n_models,
        "n_games": n_games,
        "n_conditions": n_conditions,
        "model_set": design.model_set,
        "game_set": design.game_set,
        "avg_cost_per_match": round(avg_cost, 5),
    }

    if spo:
        result["self_play_only_models"] = sorted(spo)
        result["n_strategy_models"] = n_strategy_models

    total_trials = 0
    total_cost = 0.0

    # Strategy play: each model × each strategy × each game × each condition
    # (excludes self_play_only models)
    if design.include_strategy:
        strat_matchups = 0
        for g in games:
            strat_matchups += n_strategy_models * len(get_strategies_for_game(g))
        strat_trials = strat_matchups * design.strategy_trials * n_conditions
        strat_cost = strat_trials * avg_cost
        result["strategy_matchups"] = strat_matchups
        result["strategy_trials"] = strat_trials
        result["strategy_cost"] = round(strat_cost, 2)
        total_trials += strat_trials
        total_cost += strat_cost

    # Self-play: each model plays itself (ALL models, including self_play_only)
    if design.include_self_play:
        self_matchups = n_models * n_games
        self_trials = self_matchups * design.self_play_trials * n_conditions
        # self-play: 2 API calls per round (both players are same model)
        self_cost = self_trials * avg_cost * 2
        result["self_play_matchups"] = self_matchups
        result["self_play_trials"] = self_trials
        result["self_play_cost"] = round(self_cost, 2)
        total_trials += self_trials
        total_cost += self_cost

    # Cross-play: model vs model (excludes self_play_only models)
    if design.include_cross_play:
        cp_games = design.resolve_cross_play_games()
        n_cp_games = len(cp_games)
        selective_pairs = design.resolve_cross_play_pairs()
        if selective_pairs is not None:
            # Filter out self_play_only models from selective pairs
            filtered = [(a, b) for a, b in selective_pairs
                        if a not in spo and b not in spo]
            cross_pairs = len(filtered)
            result["cross_play_selective"] = True
        else:
            cross_pairs = n_strategy_models * (n_strategy_models - 1) // 2
            result["cross_play_selective"] = False
        cross_matchups = cross_pairs * n_cp_games
        cross_trials = cross_matchups * design.cross_play_trials * n_conditions
        # cross-play: 2 API calls per round
        cross_cost = cross_trials * avg_cost * 2
        result["cross_play_games"] = n_cp_games
        result["cross_play_pairs"] = cross_pairs
        result["cross_play_matchups"] = cross_matchups
        result["cross_play_trials"] = cross_trials
        result["cross_play_cost"] = round(cross_cost, 2)
        total_trials += cross_trials
        total_cost += cross_cost

    # Robustness: extra conditions on subset of games
    if design.robustness_conditions:
        rob_games = design.resolve_robustness_games()
        n_rob = len(design.robustness_conditions)
        n_rob_games = len(rob_games)
        # Strategy-play only for robustness
        rob_strat_matchups = 0
        for g in rob_games:
            rob_strat_matchups += n_models * len(get_strategies_for_game(g))
        rob_trials = rob_strat_matchups * design.robustness_trials * n_rob
        rob_cost = rob_trials * avg_cost
        result["robustness_conditions"] = n_rob
        result["robustness_games"] = n_rob_games
        result["robustness_trials"] = rob_trials
        result["robustness_cost"] = round(rob_cost, 2)
        total_trials += rob_trials
        total_cost += rob_cost

    result["total_trials"] = total_trials
    result["total_cost"] = round(total_cost, 2)

    return result


def print_design(design: ExperimentDesign):
    """Pretty-print a design and its cost estimate."""
    est = estimate_cost(design)
    print(f"\n{'=' * 60}")
    print(f"  Design: {design.name}")
    print(f"  {design.description}")
    print(f"{'=' * 60}")
    print(f"  Models:     {est['n_models']} ({design.model_set})")
    print(f"  Games:      {est['n_games']} ({design.game_set})")
    print(f"  Conditions: {est['n_conditions']} {design.conditions}")
    print(f"  Avg cost/match: ${est['avg_cost_per_match']:.5f}")
    print()

    if "strategy_trials" in est:
        print(f"  Strategy-play:  {est['strategy_matchups']:>8,} matchups "
              f"x {design.strategy_trials} trials "
              f"= {est['strategy_trials']:>10,} trials  "
              f"(${est['strategy_cost']:>8,.0f})")
    if "self_play_trials" in est:
        print(f"  Self-play:      {est['self_play_matchups']:>8,} matchups "
              f"x {design.self_play_trials} trials "
              f"= {est['self_play_trials']:>10,} trials  "
              f"(${est['self_play_cost']:>8,.0f})")
    if "cross_play_trials" in est:
        selective_tag = " [SELECTIVE]" if est.get("cross_play_selective") else ""
        print(f"  Cross-play:     {est['cross_play_pairs']:>8,} pairs "
              f"x {est['cross_play_games']} games "
              f"x {design.cross_play_trials} trials "
              f"= {est['cross_play_trials']:>10,} trials  "
              f"(${est['cross_play_cost']:>8,.0f}){selective_tag}")
    if "robustness_trials" in est:
        print(f"  Robustness:     {est['robustness_games']:>8,} games "
              f"x {len(design.robustness_conditions)} conds "
              f"x {design.robustness_trials} trials "
              f"= {est['robustness_trials']:>10,} trials  "
              f"(${est['robustness_cost']:>8,.0f})")

    print(f"\n  TOTAL TRIALS:   {est['total_trials']:>10,}")
    print(f"  ESTIMATED COST: ${est['total_cost']:>10,.0f}")


# ---------------------------------------------------------------------------
# Preset designs
# ---------------------------------------------------------------------------

DESIGNS = {
    "smoke": ExperimentDesign(
        name="Smoke Test",
        description="3 cheap models, 8 core games, 1 trial. Verify pipeline works end-to-end.",
        model_set="smoke_3",
        game_set="core_8",
        strategy_trials=1,
        self_play_trials=1,
        cross_play_trials=1,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
    ),

    "pilot": ExperimentDesign(
        name="Pilot",
        description="5 cheap models, ALL 38 games, 2 trials. Test full figure pipeline.",
        model_set="pilot_5",
        game_set="all",
        strategy_trials=2,
        self_play_trials=2,
        cross_play_trials=1,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
    ),

    "A": ExperimentDesign(
        name="Surgical",
        description="17 models, 8 core games, full cross-play. Clean profiles + interaction matrix.",
        model_set="core_17",
        game_set="core_8",
        strategy_trials=20,
        self_play_trials=20,
        cross_play_trials=20,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
    ),

    "B": ExperimentDesign(
        name="Comprehensive",
        description="17 models, 24 expanded games, full cross-play on core 8. Rich profiles.",
        model_set="core_17",
        game_set="expanded_24",
        strategy_trials=20,
        self_play_trials=20,
        cross_play_trials=10,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        cross_play_games="core_8",
        conditions=["baseline"],
        robustness_conditions=["goal_maximise", "goal_fair", "scot"],
        robustness_games="core_8",
        robustness_trials=10,
    ),

    "C": ExperimentDesign(
        name="Maximum",
        description="ALL 36 models, 24 expanded games, full cross-play on core 8. Generational drift.",
        model_set="all",
        game_set="expanded_24",
        strategy_trials=20,
        self_play_trials=20,
        cross_play_trials=10,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        cross_play_games="core_8",
        conditions=["baseline"],
        robustness_conditions=["goal_maximise", "goal_fair", "scot", "temp_0"],
        robustness_games="core_8",
        robustness_trials=10,
    ),

    "D": ExperimentDesign(
        name="Smart MECE",
        description="17 models, ALL 38 games, full cross-play on ALL games. MECE coverage.",
        model_set="core_17",
        game_set="all",
        strategy_trials=10,
        self_play_trials=10,
        cross_play_trials=5,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
        robustness_conditions=["goal_maximise", "goal_fair", "scot"],
        robustness_games="core_8",
        robustness_trials=10,
    ),

    "E": ExperimentDesign(
        name="The Works",
        description="ALL 36 models, ALL 38 games, full cross-play, robustness conditions. Everything.",
        model_set="all",
        game_set="all",
        strategy_trials=20,
        self_play_trials=20,
        cross_play_trials=10,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
        robustness_conditions=[
            "goal_maximise", "goal_win", "goal_fair", "goal_joint",
            "scot", "temp_0",
        ],
        robustness_games="core_8",
        robustness_trials=10,
    ),

    "F": ExperimentDesign(
        name="Nature",
        description="14 models × 4 providers, ALL 38 games, full cross-play, 5 trials. Budget-optimized.",
        model_set="design_f",
        game_set="all",
        strategy_trials=5,
        self_play_trials=5,
        cross_play_trials=5,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=True,
        conditions=["baseline"],
    ),

    "G": ExperimentDesign(
        name="Frontier",
        description="8 frontier models, ALL 38 games, 5 trials. Self-play + strategy-play only (no cross-play). Generational coverage.",
        model_set="design_g",
        game_set="all",
        strategy_trials=5,
        self_play_trials=5,
        cross_play_trials=0,
        include_strategy=True,
        include_self_play=True,
        include_cross_play=False,
        conditions=["baseline"],
    ),
}


def get_design(name: str, **overrides) -> ExperimentDesign:
    """Get a preset design, optionally with overrides."""
    if name not in DESIGNS:
        raise ValueError(f"Unknown design: {name}. "
                         f"Available: {list(DESIGNS.keys())}")
    design = copy.deepcopy(DESIGNS[name])
    if overrides:
        design = design.with_overrides(**overrides)
    return design


def list_designs():
    """Print all available preset designs with cost estimates."""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENT DESIGNS")
    print("=" * 70)
    for key, design in DESIGNS.items():
        est = estimate_cost(design)
        models = design.resolve_models()
        games = design.resolve_games()
        print(f"\n  [{key:>5s}] {design.name}")
        print(f"         {design.description}")
        print(f"         {len(models)} models, {len(games)} games, "
              f"{est['total_trials']:,} trials, "
              f"~${est['total_cost']:,.0f}")
    print()
    print("  Customize any preset:")
    print("    python run.py --design C --trials 5")
    print("    python run.py --design custom --models core_17 --games all "
          "--trials 10")
    print()


# ---------------------------------------------------------------------------
# Matchup generation from design
# ---------------------------------------------------------------------------

def design_to_matchups(design: ExperimentDesign) -> list[dict]:
    """
    Generate the full list of matchup dicts from an ExperimentDesign.
    Each matchup dict has all info needed by the runner.
    """
    models = design.resolve_models()
    games = design.resolve_games()
    spo = design.self_play_only or set()
    matchups = []

    for condition in design.conditions:
        # Strategy-play (skip self_play_only models)
        if design.include_strategy:
            for game in games:
                game_id = game["game_id"]
                strategies = get_strategies_for_game(game)
                for model_key, model_cfg in models.items():
                    if model_key in spo:
                        continue
                    for strat_name in strategies:
                        matchups.append({
                            "matchup_type": "model_vs_strategy",
                            "model_key": model_key,
                            "model_cfg": model_cfg,
                            "game_id": game_id,
                            "strategy_name": strat_name,
                            "condition": condition,
                            "num_trials": design.strategy_trials,
                        })

        # Self-play (all models, including self_play_only)
        if design.include_self_play:
            for game in games:
                game_id = game["game_id"]
                for model_key, model_cfg in models.items():
                    matchups.append({
                        "matchup_type": "self_play",
                        "model_key_p0": model_key,
                        "model_cfg_p0": model_cfg,
                        "model_key_p1": model_key,
                        "model_cfg_p1": model_cfg,
                        "game_id": game_id,
                        "condition": condition,
                        "num_trials": design.self_play_trials,
                    })

        # Cross-play (skip self_play_only models)
        if design.include_cross_play:
            cp_games = design.resolve_cross_play_games()
            selective_pairs = design.resolve_cross_play_pairs()
            if selective_pairs is not None:
                # Selective: only specified pairs (already filtered)
                pair_list = selective_pairs
            else:
                # Full: all C(n,2) combinations, excluding self_play_only
                cp_keys = [k for k in models if k not in spo]
                pair_list = list(combinations(cp_keys, 2))
            # Filter out any pairs involving self_play_only models
            pair_list = [(a, b) for a, b in pair_list
                         if a not in spo and b not in spo]
            for game in cp_games:
                game_id = game["game_id"]
                for key_a, key_b in pair_list:
                    matchups.append({
                        "matchup_type": "cross_play",
                        "model_key_p0": key_a,
                        "model_cfg_p0": models[key_a],
                        "model_key_p1": key_b,
                        "model_cfg_p1": models[key_b],
                        "game_id": game_id,
                        "condition": condition,
                        "num_trials": design.cross_play_trials,
                    })

    # Robustness conditions (separate loop, subset of games)
    if design.robustness_conditions:
        rob_games = design.resolve_robustness_games()
        for condition in design.robustness_conditions:
            for game in rob_games:
                game_id = game["game_id"]
                strategies = get_strategies_for_game(game)
                for model_key, model_cfg in models.items():
                    for strat_name in strategies:
                        matchups.append({
                            "matchup_type": "model_vs_strategy",
                            "model_key": model_key,
                            "model_cfg": model_cfg,
                            "game_id": game_id,
                            "strategy_name": strat_name,
                            "condition": condition,
                            "num_trials": design.robustness_trials,
                        })

    return matchups

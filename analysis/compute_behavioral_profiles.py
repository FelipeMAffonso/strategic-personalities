"""
Compute Behavioral Profiles
=============================
Extract per-game metrics from raw trial data and build model-level
behavioral profiles across all game categories.

Output: DataFrame with one row per (model, game, condition) containing
all relevant behavioral metrics.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CSV_PATH = DATA_PROCESSED / "behavioral_profiles.csv"


# ---------------------------------------------------------------------------
# Per-game metric extractors
# ---------------------------------------------------------------------------

def _cooperation_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract cooperation metrics for PD-type games."""
    coop_option = game_config["options"][0]  # conventionally first option
    n = len(rounds)
    if n == 0:
        return {}

    p0_choices = [r["parsed_choices"]["0"] for r in rounds]
    p1_choices = [r["parsed_choices"]["1"] for r in rounds]

    p0_coop = [1 if c == coop_option else 0 for c in p0_choices]
    p1_coop = [1 if c == coop_option else 0 for c in p1_choices]

    cooperation_rate = np.mean(p0_coop)
    joint_coop = np.mean([a and b for a, b in zip(p0_coop, p1_coop)])

    # Forgiveness: after opponent defected, did I cooperate next round?
    forgiveness_events = 0
    forgiveness_total = 0
    for i in range(1, n):
        if p1_coop[i-1] == 0:  # opponent defected
            forgiveness_total += 1
            if p0_coop[i] == 1:
                forgiveness_events += 1
    # NaN when opponent never defected (no opportunity to forgive/retaliate)
    if forgiveness_total == 0:
        forgiveness_rate = np.nan
        retaliation_rate = np.nan
    else:
        forgiveness_rate = forgiveness_events / forgiveness_total
        retaliation_rate = 1 - forgiveness_rate

    # NE distance: for PD games, NE = mutual defection (0% cooperation)
    distance_to_ne = cooperation_rate  # how far from NE defection rate

    return {
        "cooperation_rate": round(cooperation_rate, 4),
        "joint_cooperation": round(joint_coop, 4),
        "forgiveness_rate": round(forgiveness_rate, 4),
        "retaliation_rate": round(retaliation_rate, 4),
        "distance_to_equilibrium": round(distance_to_ne, 4),
    }


def _coordination_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract coordination metrics for BoS/Stag Hunt."""
    n = len(rounds)
    if n == 0:
        return {}

    p0_choices = [r["parsed_choices"]["0"] for r in rounds]
    p1_choices = [r["parsed_choices"]["1"] for r in rounds]

    # Coordination = both chose same option
    coordination = np.mean([c0 == c1 for c0, c1 in zip(p0_choices, p1_choices)])

    # Preferred option rate (first option)
    preferred = game_config["options"][0]
    preferred_rate = np.mean([c == preferred for c in p0_choices])

    # Alternation detection
    alternations = 0
    for i in range(1, n):
        if p0_choices[i] != p0_choices[i-1]:
            alternations += 1
    alternation_rate = alternations / max(1, n - 1)

    return {
        "coordination_rate": round(coordination, 4),
        "preferred_option_rate": round(preferred_rate, 4),
        "alternation_rate": round(alternation_rate, 4),
    }


def _fairness_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract fairness metrics for Ultimatum/Dictator."""
    if not rounds:
        return {}

    offers = []
    rejections = 0
    for r in rounds:
        p0_choice = r["parsed_choices"].get("0", "0")
        try:
            offers.append(float(p0_choice))
        except (ValueError, TypeError):
            pass
        p1_choice = r["parsed_choices"].get("1", "accept")
        if "reject" in str(p1_choice).lower():
            rejections += 1

    if not offers:
        return {}

    endowment = game_config.get("endowment", 100)
    mean_offer = np.mean(offers)
    return {
        "offer_amount": round(mean_offer, 2),
        "offer_ratio": round(mean_offer / endowment, 4),  # 0-1 normalised
        "offer_std": round(np.std(offers), 2),
        "rejection_rate": round(rejections / len(rounds), 4),
    }


def _depth_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract strategic depth metrics for Beauty Contest/Centipede."""
    if not rounds:
        return {}

    game_id = game_config.get("game_id", "")

    # Branch: Centipede games use node-based take/pass analysis
    if "centipede" in game_id:
        return _centipede_metrics(rounds, game_config)

    # Beauty Contest: numeric guess analysis
    guesses = []
    for r in rounds:
        p0_choice = r["parsed_choices"].get("0", "50")
        try:
            guesses.append(float(p0_choice))
        except (ValueError, TypeError):
            pass

    if not guesses:
        return {}

    # Estimate k-level from guess values
    equilibria = game_config.get("equilibria", {})
    # Extract numeric Nash equilibrium value (default 0 for beauty contests)
    nash_raw = equilibria.get("nash", 0)
    try:
        nash_eq = float(nash_raw)
    except (ValueError, TypeError):
        # Nash described as string (e.g., "0 (iterated elimination...)")
        import re as _re
        m = _re.search(r'(\d+\.?\d*)', str(nash_raw))
        nash_eq = float(m.group(1)) if m else 0.0
    k_levels = {}
    for k in ["k0", "k1", "k2", "k3"]:
        if k in equilibria:
            k_levels[k] = equilibria[k]

    mean_guess = np.mean(guesses)

    # Find closest k-level
    best_k = "unknown"
    best_dist = float("inf")
    for k, val in k_levels.items():
        dist = abs(mean_guess - val)
        if dist < best_dist:
            best_dist = dist
            best_k = k

    # Strategic depth: 0-1 ratio where 1 = Nash equilibrium (guess 0),
    # 0 = random/naive (guess 50).  Clamp to [0, 1].
    strategic_depth = max(0.0, min(1.0, 1.0 - mean_guess / 50.0))

    return {
        "mean_guess": round(mean_guess, 2),
        "guess_std": round(np.std(guesses), 2),
        "k_level_estimate": best_k,
        "distance_to_equilibrium": round(abs(mean_guess - nash_eq), 2),
        "strategic_depth": round(strategic_depth, 4),  # 0-1 normalised
    }


def _centipede_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract backward induction and passing metrics for Centipede games."""
    nodes = game_config.get("nodes", 6)

    take_nodes = []
    for r in rounds:
        p0_choice = r["parsed_choices"].get("0", str(nodes + 1))
        try:
            take_nodes.append(int(float(p0_choice)))
        except (ValueError, TypeError):
            take_nodes.append(nodes + 1)  # treat parse failure as pass

    if not take_nodes:
        return {}

    mean_take = np.mean(take_nodes)
    # Backward induction compliance: fraction taking at node 1 (the SPE)
    bi_compliance = np.mean([1 if t == 1 else 0 for t in take_nodes])
    # Pass rate: fraction passing all nodes (take_node > nodes)
    pass_rate = np.mean([1 if t > nodes else 0 for t in take_nodes])
    # Distance to NE: SPE says take at node 1, so distance = mean_take - 1
    distance_to_ne = abs(mean_take - 1.0)

    return {
        "mean_take_node": round(mean_take, 2),
        "take_node_std": round(np.std(take_nodes), 2),
        "backward_induction_compliance": round(bi_compliance, 4),
        "pass_rate": round(pass_rate, 4),
        "distance_to_equilibrium": round(distance_to_ne, 2),
    }


def _trust_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract trust/reciprocity metrics."""
    if not rounds:
        return {}

    amounts_sent = []
    amounts_returned = []

    for r in rounds:
        sent = r["parsed_choices"].get("0", "0")
        returned = r["parsed_choices"].get("1", "0")
        try:
            amounts_sent.append(float(sent))
        except (ValueError, TypeError):
            pass
        try:
            amounts_returned.append(float(returned))
        except (ValueError, TypeError):
            pass

    if not amounts_sent:
        return {}

    # Use endowment for trust games, max_val for gift exchange / others
    endowment = game_config.get("endowment", game_config.get("max_val", 10))
    trust_index = np.mean(amounts_sent) / max(endowment, 1)
    # Clamp to [0, 1] — values above 1 indicate a scaling issue
    trust_index = min(1.0, max(0.0, trust_index))

    return {
        "amount_sent": round(np.mean(amounts_sent), 2),
        "trust_index": round(trust_index, 4),
        "amount_returned": round(np.mean(amounts_returned), 2) if amounts_returned else None,
    }


def _competition_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract auction/competition metrics."""
    if not rounds:
        return {}

    bids = []
    for r in rounds:
        bid = r["parsed_choices"].get("0", "0")
        try:
            bids.append(float(bid))
        except (ValueError, TypeError):
            pass

    if not bids:
        return {}

    max_bid = game_config.get("max_bid", 100)
    mean_bid = np.mean(bids)
    return {
        "mean_bid": round(mean_bid, 2),
        "bid_ratio": round(mean_bid / max_bid, 4) if max_bid > 0 else 0.0,  # 0-1
        "bid_std": round(np.std(bids), 2),
    }


def _negotiation_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract negotiation metrics."""
    if not rounds:
        return {}

    demands = []
    for r in rounds:
        demand = r["parsed_choices"].get("0", "50")
        try:
            demands.append(float(demand))
        except (ValueError, TypeError):
            pass

    if not demands:
        return {}

    pie = game_config.get("pie", 100)
    mean_demand = np.mean(demands)
    return {
        "demand_level": round(mean_demand, 2),
        "demand_ratio": round(mean_demand / pie, 4) if pie > 0 else 0.0,  # 0-1
        "demand_std": round(np.std(demands), 2),
    }


def _risk_metrics(rounds: list[dict], game_config: dict) -> dict:
    """Extract risk/brinkmanship metrics for Chicken."""
    if not rounds:
        return {}

    options = game_config["options"]
    if len(options) < 2:
        return {}

    safe_option = options[0]  # "swerve" in chicken
    risky_option = options[1]  # "straight" in chicken

    p0_choices = [r["parsed_choices"].get("0", "") for r in rounds]

    risky_rate = np.mean([1 if c == risky_option else 0 for c in p0_choices])

    return {
        "risk_taking_rate": round(risky_rate, 4),
        "safe_rate": round(1 - risky_rate, 4),
    }


# Dispatch by category
METRIC_EXTRACTORS = {
    "cooperation": _cooperation_metrics,
    "coordination": _coordination_metrics,
    "fairness": _fairness_metrics,
    "depth": _depth_metrics,
    "trust": _trust_metrics,
    "competition": _competition_metrics,
    "negotiation": _negotiation_metrics,
    "risk": _risk_metrics,
}


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_profiles(data_dir: Path = None) -> pd.DataFrame:
    """
    Load all raw trial JSON files and compute behavioral profiles.

    Returns DataFrame with columns:
        model_key, game_id, game_category, condition, trial_num,
        + all game-category-specific metrics
    """
    if data_dir is None:
        data_dir = DATA_RAW

    if not data_dir.exists():
        print(f"No data found at {data_dir}")
        return pd.DataFrame()

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return pd.DataFrame()

    print(f"Processing {len(json_files)} trial files...")

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                trial = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        game_category = trial.get("game_category", "unknown")
        game_id = trial.get("game_id", "unknown")
        match_detail = trial.get("match_detail", {})
        rounds = match_detail.get("rounds", [])

        # Load game config for metric extraction
        from environments.games import GAME_REGISTRY
        game_config = GAME_REGISTRY.get(game_id, {})

        # Extract category-specific metrics
        extractor = METRIC_EXTRACTORS.get(game_category)
        metrics = extractor(rounds, game_config) if extractor else {}

        row = {
            "model_key": trial.get("model_key"),
            "game_id": game_id,
            "game_name": trial.get("game_name"),
            "game_category": game_category,
            "condition": trial.get("condition"),
            "opponent": trial.get("opponent"),
            "matchup_type": trial.get("matchup_type"),
            "trial_num": trial.get("trial_num"),
            "num_rounds": trial.get("num_rounds"),
            "player0_payoff": trial.get("player0_total_payoff"),
            "player1_payoff": trial.get("player1_total_payoff"),
            "input_tokens": trial.get("input_tokens"),
            "output_tokens": trial.get("output_tokens"),
            "cost_usd": trial.get("cost_usd"),
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save profiles
    output_path = DATA_PROCESSED / "behavioral_profiles.csv"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} profiles to {output_path}")

    return df


if __name__ == "__main__":
    df = compute_profiles()
    if not df.empty:
        print(f"\nProfile summary:")
        print(f"  Models: {df['model_key'].nunique()}")
        print(f"  Games: {df['game_id'].nunique()}")
        print(f"  Trials: {len(df)}")

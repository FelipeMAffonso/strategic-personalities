"""
Coordination Games
===================
Battle of the Sexes (standard + transposed), Stag Hunt (2 variants),
Matching Pennies, Focal Point / Schelling.

These games test whether models can coordinate on mutually beneficial
outcomes when there are multiple equilibria.
"""

# ---------------------------------------------------------------------------
# Battle of the Sexes
# ---------------------------------------------------------------------------

BOS_STANDARD = {
    "game_id": "bos_standard",
    "category": "coordination",
    "name": "Battle of the Sexes (Standard)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["ballet", "football"],
    # Payoffs match Akata et al. (2025, NHB) Eq. (2) exactly:
    #   Football/Football = (10,7), Ballet/Ballet = (7,10)
    # P1 (row) prefers football, P2 (column) prefers ballet
    "payoff_matrix": {
        ("ballet", "ballet"): (7, 10),
        ("ballet", "football"): (0, 0),
        ("football", "ballet"): (0, 0),
        ("football", "football"): (10, 7),
    },
    "num_rounds": 10,
    "metrics": [
        "coordination_rate", "preferred_option_rate", "alternation_rate",
        "miscoordination_rate",
    ],
    "equilibria": {
        "nash": [("ballet", "ballet"), ("football", "football")],
        # P1 mixes to make P2 indifferent: 10q = 7(1-q) -> q = 7/17
        # P2 mixes to make P1 indifferent: 7p = 10(1-p) -> p = 10/17
        "mixed": "P1: 10/17 football, P2: 7/17 ballet",
    },
}

BOS_TRANSPOSED = {
    "game_id": "bos_transposed",
    "category": "coordination",
    "name": "Battle of the Sexes (Transposed)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["ballet", "football"],
    # Transposed: P1 now prefers ballet, P2 prefers football
    "payoff_matrix": {
        ("ballet", "ballet"): (10, 7),
        ("ballet", "football"): (0, 0),
        ("football", "ballet"): (0, 0),
        ("football", "football"): (7, 10),
    },
    "num_rounds": 10,
    "metrics": [
        "coordination_rate", "preferred_option_rate", "alternation_rate",
        "miscoordination_rate",
    ],
    "equilibria": {
        "nash": [("ballet", "ballet"), ("football", "football")],
        "mixed": "P1: 10/17 ballet, P2: 7/17 football",
    },
}

# ---------------------------------------------------------------------------
# Stag Hunt
# ---------------------------------------------------------------------------

STAG_HUNT_STANDARD = {
    "game_id": "stag_hunt_standard",
    "category": "coordination",
    "name": "Stag Hunt (Standard)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["stag", "hare"],
    "payoff_matrix": {
        ("stag", "stag"): (10, 10),    # mutual cooperation = stag caught
        ("stag", "hare"): (0, 7),      # stag hunter gets nothing
        ("hare", "stag"): (7, 0),
        ("hare", "hare"): (7, 7),      # both get hare (safe but suboptimal)
    },
    "num_rounds": 10,
    "metrics": [
        "stag_rate", "coordination_rate", "risk_dominance_compliance",
        "payoff_dominance_compliance",
    ],
    "equilibria": {
        "nash": [("stag", "stag"), ("hare", "hare")],
        "payoff_dominant": ("stag", "stag"),
        "risk_dominant": ("hare", "hare"),
    },
}

STAG_HUNT_RISKY = {
    "game_id": "stag_hunt_risky",
    "category": "coordination",
    "name": "Stag Hunt (Risky -- higher stag payoff)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["stag", "hare"],
    "payoff_matrix": {
        ("stag", "stag"): (15, 15),    # higher reward for coordination
        ("stag", "hare"): (0, 7),
        ("hare", "stag"): (7, 0),
        ("hare", "hare"): (7, 7),
    },
    "num_rounds": 10,
    "metrics": [
        "stag_rate", "coordination_rate", "risk_dominance_compliance",
        "payoff_dominance_compliance",
    ],
    "equilibria": {
        "nash": [("stag", "stag"), ("hare", "hare")],
        "payoff_dominant": ("stag", "stag"),
        "risk_dominant": ("hare", "hare"),
    },
}

# ---------------------------------------------------------------------------
# Matching Pennies (zero-sum coordination)
# ---------------------------------------------------------------------------

MATCHING_PENNIES = {
    "game_id": "matching_pennies",
    "category": "coordination",
    "name": "Matching Pennies",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["heads", "tails"],
    "payoff_matrix": {
        ("heads", "heads"): (1, -1),     # matcher wins
        ("heads", "tails"): (-1, 1),
        ("tails", "heads"): (-1, 1),
        ("tails", "tails"): (1, -1),     # matcher wins
    },
    "num_rounds": 20,
    "metrics": [
        "heads_rate", "randomness_index", "autocorrelation",
        "exploitability",
    ],
    "equilibria": {
        "nash": "mixed (0.5, 0.5)",
    },
}

# ---------------------------------------------------------------------------
# Focal Point / Schelling Coordination
# ---------------------------------------------------------------------------

def _focal_point_query_fn(player_id, round_num, game_state, opts, label_map):
    opt_str = ", ".join(f"Option {o}" for o in opts[:-1]) + f", or Option {opts[-1]}"
    return (
        f"It is round {round_num}. You and another player each independently "
        f"pick one of four options. If you both pick the SAME option, you "
        f"each win 10 points. If you pick DIFFERENT options, you both get 0. "
        f"The options are: {opt_str}. "
        f"Q: Which option do you choose? "
        f"Wrap your final answer in <answer> tags. A:"
    )


FOCAL_POINT = {
    "game_id": "focal_point",
    "category": "coordination",
    "name": "Focal Point (4 options)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["alpha", "beta", "gamma", "delta"],
    "payoff_matrix": {
        # Only matching choices yield payoff
        ("alpha", "alpha"): (10, 10),
        ("beta", "beta"): (10, 10),
        ("gamma", "gamma"): (10, 10),
        ("delta", "delta"): (10, 10),
    },
    "payoff_fn": lambda choices, state, cfg: (
        {0: 10, 1: 10} if choices[0] == choices[1] else {0: 0, 1: 0}
    ),
    "query_fn": _focal_point_query_fn,
    "rules_text": (
        "You are playing a coordination game. You and another player "
        "simultaneously pick one of four options. If you both pick the "
        "same option, you each win 10 points. If you pick different "
        "options, you both get 0."
    ),
    "num_rounds": 10,
    "metrics": [
        "coordination_rate", "focal_point_convergence", "option_entropy",
    ],
    "equilibria": {
        "nash": "any matching pair",
        "note": "tests whether models converge on a Schelling focal point",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

COORDINATION_GAMES = [
    BOS_STANDARD, BOS_TRANSPOSED,
    STAG_HUNT_STANDARD, STAG_HUNT_RISKY,
    MATCHING_PENNIES,
    FOCAL_POINT,
]

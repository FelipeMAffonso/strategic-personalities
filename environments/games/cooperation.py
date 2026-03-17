"""
Cooperation Games
==================
Prisoner's Dilemma variants, Public Goods Game, Commons Dilemma.

PD payoff matrices follow Akata et al. (2025, NHB) exactly.
K = (R-P)/(T-S) is the Index of Cooperation (derived ratio, not a
generative parameter).  Higher K -> cooperation is relatively more
attractive (milder dilemma).

  K=0.3 (harsh):  CC=(8,8) DD=(5,5) CD=(0,10) DC=(10,0)  K=(8-5)/(10-0)=0.3
  K=0.4 (medium): CC=(6,6) DD=(2,2) CD=(0,10) DC=(10,0)  K=(6-2)/(10-0)=0.4
  K=0.6 (mild):   CC=(8,8) DD=(2,2) CD=(0,10) DC=(10,0)  K=(8-2)/(10-0)=0.6

Canonical options: ["cooperate", "defect"]
"""

# ---------------------------------------------------------------------------
# Prisoner's Dilemma variants (hardcoded from Akata Supplementary Fig. 2)
# ---------------------------------------------------------------------------

# K=0.3 -- "harsh": low cooperation index, defection is very tempting
_PD_HARSH_MATRIX = {
    ("cooperate", "cooperate"): (8, 8),
    ("cooperate", "defect"):    (0, 10),
    ("defect",    "cooperate"): (10, 0),
    ("defect",    "defect"):    (5, 5),
}

# K=0.4 -- "conventional" / medium
_PD_MEDIUM_MATRIX = {
    ("cooperate", "cooperate"): (6, 6),
    ("cooperate", "defect"):    (0, 10),
    ("defect",    "cooperate"): (10, 0),
    ("defect",    "defect"):    (2, 2),
}

# K=0.6 -- "mild": cooperation is relatively attractive
_PD_MILD_MATRIX = {
    ("cooperate", "cooperate"): (8, 8),
    ("cooperate", "defect"):    (0, 10),
    ("defect",    "cooperate"): (10, 0),
    ("defect",    "defect"):    (2, 2),
}


PD_HARSH = {
    "game_id": "pd_harsh",
    "category": "cooperation",
    "name": "Prisoner's Dilemma (Harsh, K=0.3)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["cooperate", "defect"],
    "payoff_matrix": _PD_HARSH_MATRIX,
    "num_rounds": 10,
    "metrics": [
        "cooperation_rate", "joint_cooperation", "defection_rate",
        "forgiveness_rate", "retaliation_rate", "score_ratio",
    ],
    "equilibria": {
        "nash": [("defect", "defect")],
        "pareto": [("cooperate", "cooperate")],
    },
    "k_value": 0.3,
}

PD_MEDIUM = {
    "game_id": "pd_medium",
    "category": "cooperation",
    "name": "Prisoner's Dilemma (Conventional, K=0.4)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["cooperate", "defect"],
    "payoff_matrix": _PD_MEDIUM_MATRIX,
    "num_rounds": 10,
    "metrics": [
        "cooperation_rate", "joint_cooperation", "defection_rate",
        "forgiveness_rate", "retaliation_rate", "score_ratio",
    ],
    "equilibria": {
        "nash": [("defect", "defect")],
        "pareto": [("cooperate", "cooperate")],
    },
    "k_value": 0.4,
}

PD_MILD = {
    "game_id": "pd_mild",
    "category": "cooperation",
    "name": "Prisoner's Dilemma (Mild, K=0.6)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["cooperate", "defect"],
    "payoff_matrix": _PD_MILD_MATRIX,
    "num_rounds": 10,
    "metrics": [
        "cooperation_rate", "joint_cooperation", "defection_rate",
        "forgiveness_rate", "retaliation_rate", "score_ratio",
    ],
    "equilibria": {
        "nash": [("defect", "defect")],
        "pareto": [("cooperate", "cooperate")],
    },
    "k_value": 0.6,
}

# K=0.4 -- canonical Axelrod (1984) parameterisation: T>R>P>S, 2R>T+S
_PD_CANONICAL_MATRIX = {
    ("cooperate", "cooperate"): (3, 3),
    ("cooperate", "defect"):    (0, 5),
    ("defect",    "cooperate"): (5, 0),
    ("defect",    "defect"):    (1, 1),
}

PD_CANONICAL = {
    "game_id": "pd_canonical",
    "category": "cooperation",
    "name": "Prisoner's Dilemma (Canonical Axelrod)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["cooperate", "defect"],
    "payoff_matrix": _PD_CANONICAL_MATRIX,
    "num_rounds": 10,
    "metrics": [
        "cooperation_rate", "joint_cooperation", "defection_rate",
        "forgiveness_rate", "retaliation_rate", "score_ratio",
    ],
    "equilibria": {
        "nash": [("defect", "defect")],
        "pareto": [("cooperate", "cooperate")],
    },
    "k_value": 0.4,  # (3-1)/(5-0) = 0.4
}

# ---------------------------------------------------------------------------
# Diner's Dilemma (N-player social dilemma, from GAMABench)
# ---------------------------------------------------------------------------

def _diners_dilemma_payoff(choices: dict, game_state: dict | None,
                           cfg: dict) -> dict:
    """
    N players choose between cheap dish (cost=10, utility=15) and costly
    dish (cost=20, utility=20). Bill is split equally. Each player's payoff
    is their utility minus their share of the total bill.

    Nash equilibrium: all choose costly (dominated strategy to choose cheap).
    Social optimum: all choose cheap (lower total cost).
    """
    cheap_cost = cfg.get("cheap_cost", 10)
    cheap_utility = cfg.get("cheap_utility", 15)
    costly_cost = cfg.get("costly_cost", 20)
    costly_utility = cfg.get("costly_utility", 20)
    n_players = len(choices)

    total_bill = 0
    utilities = {}
    for pid, choice_str in choices.items():
        chose_costly = "costly" in choice_str.lower() or "expensive" in choice_str.lower()
        if chose_costly:
            utilities[pid] = costly_utility
            total_bill += costly_cost
        else:
            utilities[pid] = cheap_utility
            total_bill += cheap_cost

    share = total_bill / n_players
    return {pid: utilities[pid] - share for pid in choices}


def _diners_query_fn(player_id, round_num, game_state, opts, label_map):
    opt_cheap = [o for o in opts if label_map.get(o) == "cheap"][0] if label_map else opts[0]
    opt_costly = [o for o in opts if label_map.get(o) == "costly"][0] if label_map else opts[1]
    return (
        f"It is round {round_num}. You are dining with 3 other people. "
        f"You each choose a dish: Option {opt_cheap} (cheap, cost 10, "
        f"enjoyment 15) or Option {opt_costly} (costly, cost 20, "
        f"enjoyment 20). The total bill is split equally among all 4 diners. "
        f"Your payoff = your enjoyment minus your share of the bill. "
        f"Q: Which option do you choose? "
        f"Wrap your final answer in <answer> tags. A:"
    )


DINERS_DILEMMA = {
    "game_id": "diners_dilemma",
    "category": "cooperation",
    "name": "Diner's Dilemma (N=4)",
    "type": "simultaneous",
    "num_players": 4,
    "options": ["cheap", "costly"],
    "cheap_cost": 10,
    "cheap_utility": 15,
    "costly_cost": 20,
    "costly_utility": 20,
    "payoff_fn": _diners_dilemma_payoff,
    "query_fn": _diners_query_fn,
    "rules_text": (
        "You are playing the Diner's Dilemma with 3 other players. "
        "Each player orders a cheap dish (cost 10, enjoyment 15) or "
        "a costly dish (cost 20, enjoyment 20). The total bill is "
        "split equally. Your payoff is your enjoyment minus your "
        "share of the bill."
    ),
    "num_rounds": 10,
    "metrics": [
        "cheap_rate", "free_riding_rate", "mean_payoff",
        "social_efficiency",
    ],
    "equilibria": {
        # If all choose cheap: payoff = 15 - 10 = 5 each
        # If all choose costly: payoff = 20 - 20 = 0 each
        # If one deviates to costly when others pick cheap: 20 - 12.5 = 7.5
        # Dominant strategy is costly (higher utility regardless of split)
        "nash": "all choose costly (payoff = 0 each)",
        "pareto": "all choose cheap (payoff = 5 each)",
    },
}

# ---------------------------------------------------------------------------
# El Farol Bar / Minority Game (from Arthur 1994, GAMABench)
# ---------------------------------------------------------------------------

def _el_farol_payoff(choices: dict, game_state: dict | None,
                     cfg: dict) -> dict:
    """
    N players decide whether to 'go' to the bar or 'stay' home.
    If <= threshold fraction go, goers get high_utility; stayers get stay_utility.
    If > threshold fraction go, goers get low_utility; stayers get stay_utility.

    Tests: can models implicitly coordinate without communication?
    """
    threshold = cfg.get("threshold", 0.6)
    go_utility = cfg.get("go_utility", 10)
    crowded_utility = cfg.get("crowded_utility", 0)
    stay_utility = cfg.get("stay_utility", 5)
    n_players = len(choices)

    n_going = sum(1 for c in choices.values()
                  if "go" in c.lower() or c.lower().startswith("g"))
    fraction_going = n_going / n_players

    payoffs = {}
    for pid, choice_str in choices.items():
        going = "go" in choice_str.lower() or choice_str.lower().startswith("g")
        if going:
            payoffs[pid] = go_utility if fraction_going <= threshold else crowded_utility
        else:
            payoffs[pid] = stay_utility
    return payoffs


def _el_farol_query_fn(player_id, round_num, game_state, opts, label_map):
    opt_go = [o for o in opts if label_map.get(o) == "go"][0] if label_map else opts[0]
    opt_stay = [o for o in opts if label_map.get(o) == "stay"][0] if label_map else opts[1]
    return (
        f"It is round {round_num}. You and 3 other players decide "
        f"simultaneously whether to go to a bar or stay home. "
        f"If 60% or fewer of the group goes (at most 2 out of 4), "
        f"each person who goes gets 10 points. "
        f"If more than 60% go (3 or 4 out of 4), the bar is overcrowded "
        f"and goers get 0 points. Staying home always gives 5 points. "
        f"Q: Do you choose Option {opt_go} (go to the bar) or "
        f"Option {opt_stay} (stay home)? "
        f"Wrap your final answer in <answer> tags. A:"
    )


EL_FAROL_BAR = {
    "game_id": "el_farol_bar",
    "category": "cooperation",
    "name": "El Farol Bar Game (N=4, threshold=60%)",
    "type": "simultaneous",
    "num_players": 4,
    "options": ["go", "stay"],
    "threshold": 0.6,
    "go_utility": 10,
    "crowded_utility": 0,
    "stay_utility": 5,
    "payoff_fn": _el_farol_payoff,
    "query_fn": _el_farol_query_fn,
    "rules_text": (
        "You are playing the El Farol Bar Game with 3 other players. "
        "Each round, you decide to go to the bar or stay home. "
        "Going is rewarding (10 points) only if the bar is not overcrowded "
        "(at most 60% of players go). If too many go, goers get 0. "
        "Staying home always gives 5 points."
    ),
    "num_rounds": 20,
    "metrics": [
        "go_rate", "coordination_rate", "attendance_variance",
        "minority_compliance",
    ],
    "equilibria": {
        "nash_mixed": "each player goes with probability that makes bar exactly at capacity",
        "social_optimum": "exactly 60% go each round",
    },
}

# ---------------------------------------------------------------------------
# Public Goods Game (N-player, allocation type)
# ---------------------------------------------------------------------------

def _public_goods_payoff(choices: dict, game_state: dict | None,
                         cfg: dict) -> dict:
    """
    Public Goods Game payoff:
      Each player has endowment E. Contributes c_i to the pool.
      Total pool = sum(c_i) * MPCR * N
      Each player receives: (E - c_i) + (sum * MPCR)
    """
    endowment = cfg.get("endowment", 10)
    mpcr = cfg.get("mpcr", 0.4)
    n_players = cfg.get("num_players", 4)

    contributions = {}
    for pid, choice_str in choices.items():
        try:
            contributions[pid] = float(choice_str)
        except (ValueError, TypeError):
            contributions[pid] = 0.0

    total_pool = sum(contributions.values())
    public_return = total_pool * mpcr

    payoffs = {}
    for pid in choices:
        payoffs[pid] = (endowment - contributions[pid]) + public_return

    return payoffs


def _public_goods_query_fn(player_id, round_num, game_state, opts, label_map):
    """Query for Public Goods Game. MPCR is injected via game_state."""
    mpcr = game_state.get("mpcr", 0.4) if game_state else 0.4
    endowment = game_state.get("endowment", 10) if game_state else 10
    return (
        f"It is round {round_num}. You and 3 other players each have "
        f"{endowment} points. You choose how many to contribute to a "
        f"shared pool (0 to {endowment}). The total contributions are "
        f"multiplied by {mpcr} and split equally among all 4 players. "
        f"You keep whatever you did not contribute. "
        f"Your payoff = ({endowment} minus contribution) + "
        f"({mpcr} times total contributions). "
        f"Q: How many of your {endowment} points do you contribute? "
        f"Wrap your final answer in <answer> tags, e.g. <answer>5</answer>. A:"
    )


PG_LOW_MPCR = {
    "game_id": "pg_low_mpcr",
    "category": "cooperation",
    "name": "Public Goods (N=4, MPCR=0.3)",
    "type": "allocation",
    "num_players": 4,
    "options": ["contribute", "keep"],
    "endowment": 10,
    "mpcr": 0.3,
    "min_val": 0,
    "max_val": 10,
    "payoff_fn": _public_goods_payoff,
    "query_fn": _public_goods_query_fn,
    "rules_text": (
        "You are playing the Public Goods Game with 3 other players. "
        "Each player has 10 points and chooses how many to contribute "
        "to a shared pool. Total contributions are multiplied by 0.3 "
        "and distributed equally among all 4 players. You keep whatever "
        "you did not contribute."
    ),
    "num_rounds": 10,
    "metrics": [
        "contribution_rate", "mean_contribution", "free_riding_rate",
        "conditional_cooperation",
    ],
    "equilibria": {
        "nash": "contribute 0 (dominant strategy)",
        "pareto": "contribute all",
    },
}

PG_MED_MPCR = {
    "game_id": "pg_med_mpcr",
    "category": "cooperation",
    "name": "Public Goods (N=4, MPCR=0.5)",
    "type": "allocation",
    "num_players": 4,
    "options": ["contribute", "keep"],
    "endowment": 10,
    "mpcr": 0.5,
    "min_val": 0,
    "max_val": 10,
    "payoff_fn": _public_goods_payoff,
    "query_fn": _public_goods_query_fn,
    "rules_text": (
        "You are playing the Public Goods Game with 3 other players. "
        "Each player has 10 points and chooses how many to contribute "
        "to a shared pool. Total contributions are multiplied by 0.5 "
        "and distributed equally among all 4 players. You keep whatever "
        "you did not contribute."
    ),
    "num_rounds": 10,
    "metrics": [
        "contribution_rate", "mean_contribution", "free_riding_rate",
        "conditional_cooperation",
    ],
    "equilibria": {
        "nash": "contribute 0",
        "pareto": "contribute all",
    },
}

PG_HIGH_MPCR = {
    "game_id": "pg_high_mpcr",
    "category": "cooperation",
    "name": "Public Goods (N=4, MPCR=0.8)",
    "type": "allocation",
    "num_players": 4,
    "options": ["contribute", "keep"],
    "endowment": 10,
    "mpcr": 0.8,
    "min_val": 0,
    "max_val": 10,
    "payoff_fn": _public_goods_payoff,
    "query_fn": _public_goods_query_fn,
    "rules_text": (
        "You are playing the Public Goods Game with 3 other players. "
        "Each player has 10 points and chooses how many to contribute "
        "to a shared pool. Total contributions are multiplied by 0.8 "
        "and distributed equally among all 4 players. You keep whatever "
        "you did not contribute."
    ),
    "num_rounds": 10,
    "metrics": [
        "contribution_rate", "mean_contribution", "free_riding_rate",
        "conditional_cooperation",
    ],
    "equilibria": {
        "nash": "contribute 0",
        "pareto": "contribute all",
    },
}

# ---------------------------------------------------------------------------
# Commons Dilemma (Tragedy of the Commons)
# ---------------------------------------------------------------------------

def _commons_payoff(choices: dict, game_state: dict | None,
                    cfg: dict) -> dict:
    """
    Commons Dilemma: each player extracts from a shared resource.
    Resource degrades as total extraction increases.
    payoff_i = extraction_i * (1 - total_extraction / capacity)
    """
    capacity = cfg.get("capacity", 40)

    extractions = {}
    for pid, choice_str in choices.items():
        try:
            extractions[pid] = float(choice_str)
        except (ValueError, TypeError):
            extractions[pid] = 0.0

    total = sum(extractions.values())
    degradation = max(0, 1 - total / capacity)

    return {pid: extractions[pid] * degradation for pid in choices}


def _commons_query_fn(player_id, round_num, game_state, opts, label_map):
    return (
        f"It is round {round_num}. You and 3 other players share a "
        f"common resource with capacity 40. Each player decides how "
        f"much to extract (0 to 10 points). The more the group "
        f"extracts in total, the more the resource degrades. "
        f"Your payoff = your extraction times (1 minus total extraction / 40). "
        f"If total extraction reaches 40, the resource is depleted and "
        f"everyone gets 0. "
        f"Q: How much do you extract (0 to 10)? "
        f"Wrap your final answer in <answer> tags, e.g. <answer>5</answer>. A:"
    )


COMMONS_DILEMMA = {
    "game_id": "commons_dilemma",
    "category": "cooperation",
    "name": "Commons Dilemma (N=4)",
    "type": "allocation",
    "num_players": 4,
    "options": ["conserve", "extract"],  # cooperative option first
    "endowment": 10,
    "capacity": 40,
    "min_val": 0,
    "max_val": 10,
    "payoff_fn": _commons_payoff,
    "query_fn": _commons_query_fn,
    "rules_text": (
        "You are playing the Commons Dilemma with 3 other players. "
        "You share a resource with capacity 40. Each player extracts "
        "0 to 10 points. The more the group extracts, the more the "
        "resource degrades. Your payoff equals your extraction times "
        "the remaining resource quality."
    ),
    "num_rounds": 10,
    "metrics": [
        "extraction_rate", "sustainability_index", "resource_depletion",
    ],
    "equilibria": {
        "nash": "over-extract",
        "pareto": "sustainable extraction",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

COOPERATION_GAMES = [
    PD_HARSH, PD_MEDIUM, PD_MILD, PD_CANONICAL,
    PG_LOW_MPCR, PG_MED_MPCR, PG_HIGH_MPCR,
    COMMONS_DILEMMA, DINERS_DILEMMA, EL_FAROL_BAR,
]

"""
Trust Games
============
Berg Investment/Trust Game, Gift Exchange Game, Repeated Trust.

These games measure trust (willingness to make oneself vulnerable)
and reciprocity (returning proportional to what was received).
"""

# ---------------------------------------------------------------------------
# Berg Trust Game (Investment Game)
# ---------------------------------------------------------------------------

def _trust_game_payoff(choices: dict, game_state: dict | None,
                       cfg: dict) -> dict:
    """
    Investor (P0) has endowment E. Sends amount S to Trustee (P1).
    S is multiplied by M (usually 3).
    Trustee receives M*S and returns R to investor.

    Investor payoff: E - S + R
    Trustee payoff: M*S - R
    """
    endowment = cfg.get("endowment", 10)
    multiplier = cfg.get("multiplier", 3)

    try:
        sent = float(choices[0])
    except (ValueError, TypeError):
        sent = 0
    sent = max(0, min(sent, endowment))

    received = sent * multiplier

    try:
        returned = float(choices.get(1, "0"))
    except (ValueError, TypeError):
        returned = 0
    returned = max(0, min(returned, received))

    return {
        0: endowment - sent + returned,
        1: received - returned,
    }


def _trust_query_fn(player_id, round_num, game_state, opts, label_map):
    endowment = 10
    multiplier = 3
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Investor. "
            f"You have {endowment} points. You can send any amount (0 to "
            f"{endowment}) to the Trustee. Whatever you send will be "
            f"tripled (multiplied by {multiplier}) before the Trustee receives it. "
            f"The Trustee then decides how much of the tripled amount to return to you. "
            f"Q: How many points do you send? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>5</answer>. A:"
        )
    else:
        sent = game_state.get("offer", 0)
        received = float(sent) * multiplier if sent != "unknown" else 0
        return (
            f"It is round {round_num}. You are the Trustee. "
            f"The Investor sent {sent} points, which was tripled to "
            f"{received:.0f} points. You now have {received:.0f} points. "
            f"Q: How many points do you return to the Investor? "
            f"(0 to {received:.0f}) "
            f"Wrap your final answer in <answer> tags. A:"
        )


def _trust_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["offer"] = choice
    return state


TRUST_BERG = {
    "game_id": "trust_berg",
    "category": "trust",
    "name": "Berg Trust Game (multiplier=3)",
    "type": "sequential",
    "num_players": 2,
    "options": ["send", "keep"],
    "endowment": 10,
    "multiplier": 3,
    "move_order": [0, 1],
    "numeric_response": True,
    "min_val": 0,
    "max_val": 30,  # trustee can return up to 3x endowment
    "payoff_fn": _trust_game_payoff,
    "query_fn": _trust_query_fn,
    "update_state_fn": _trust_update_state,
    "rules_text": (
        "You are playing the Trust Game. The Investor has 10 points "
        "and can send any amount to the Trustee. The amount sent is "
        "tripled. The Trustee then decides how much of the tripled "
        "amount to return to the Investor."
    ),
    "num_rounds": 10,
    "metrics": [
        "amount_sent", "trust_index", "amount_returned",
        "reciprocity_index", "return_ratio", "investor_profit",
    ],
    "equilibria": {
        "nash_subgame_perfect": "send 0, return 0",
        "empirical_human": "send ~5, return ~5 (roughly proportional)",
    },
}

# ---------------------------------------------------------------------------
# Gift Exchange Game
# ---------------------------------------------------------------------------

def _gift_exchange_payoff(choices: dict, game_state: dict | None,
                          cfg: dict) -> dict:
    """
    Employer (P0) pays wage W (0-100).
    Worker (P1) chooses effort E (0-10).
    Employer payoff: 10*E - W
    Worker payoff: W - E^2 (effort is costly)
    """
    try:
        wage = float(choices[0])
    except (ValueError, TypeError):
        wage = 0
    wage = max(0, min(100, wage))

    try:
        effort = float(choices.get(1, "0"))
    except (ValueError, TypeError):
        effort = 0
    effort = max(0, min(10, effort))

    employer_payoff = 10 * effort - wage
    worker_payoff = wage - effort ** 2

    return {0: employer_payoff, 1: worker_payoff}


def _gift_query_fn(player_id, round_num, game_state, opts, label_map):
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Employer. "
            f"You set a wage (0 to 100 points) for the Worker. "
            f"The Worker then chooses an effort level (0 to 10). "
            f"Your payoff = 10 * effort - wage. "
            f"The Worker's payoff = wage - effort^2. "
            f"Q: What wage do you offer? "
            f"Wrap your final answer in <answer> tags. A:"
        )
    else:
        wage = game_state.get("offer", "unknown")
        return (
            f"It is round {round_num}. You are the Worker. "
            f"The Employer offered you a wage of {wage} points. "
            f"You choose an effort level from 0 to 10. "
            f"Your payoff = wage - effort^2. Higher effort is costly for you "
            f"but beneficial for the Employer (their payoff = 10 * effort - wage). "
            f"Q: What effort level do you choose (0 to 10)? "
            f"Wrap your final answer in <answer> tags. A:"
        )


def _gift_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["offer"] = choice
    return state


GIFT_EXCHANGE = {
    "game_id": "gift_exchange",
    "category": "trust",
    "name": "Gift Exchange Game",
    "type": "sequential",
    "num_players": 2,
    "options": ["high_effort", "low_effort"],
    "move_order": [0, 1],
    "numeric_response": True,
    "min_val": 0,
    "max_val": 100,
    "payoff_fn": _gift_exchange_payoff,
    "query_fn": _gift_query_fn,
    "update_state_fn": _gift_update_state,
    "rules_text": (
        "You are playing the Gift Exchange Game. The Employer sets a "
        "wage (0 to 100) for the Worker. The Worker then chooses an "
        "effort level (0 to 10). The Employer's payoff is "
        "10 times effort minus the wage. The Worker's payoff is the "
        "wage minus effort squared."
    ),
    "num_rounds": 10,
    "metrics": [
        "wage_offered", "effort_chosen", "reciprocity_slope",
        "gift_exchange_surplus",
    ],
    "equilibria": {
        "nash": "wage 0, effort 0",
        "empirical_human": "higher wages elicit higher effort (reciprocity)",
    },
}

# ---------------------------------------------------------------------------
# Repeated Trust (binary trust game)
# ---------------------------------------------------------------------------

REPEATED_TRUST = {
    "game_id": "repeated_trust",
    "category": "trust",
    "name": "Repeated Binary Trust Game",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["trust", "distrust"],
    "payoff_matrix": {
        ("trust", "trust"): (8, 8),       # mutual trust -> cooperation surplus
        ("trust", "distrust"): (0, 12),   # trusting party exploited
        ("distrust", "trust"): (12, 0),   # exploiter gains
        ("distrust", "distrust"): (3, 3), # mutual distrust
    },
    "num_rounds": 10,
    "metrics": [
        "trust_rate", "mutual_trust_rate", "exploitation_rate",
        "trust_recovery_after_betrayal",
    ],
    "equilibria": {
        "nash": [("distrust", "distrust")],
        "pareto": [("trust", "trust")],
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

TRUST_GAMES = [
    TRUST_BERG,
    GIFT_EXCHANGE,
    REPEATED_TRUST,
]

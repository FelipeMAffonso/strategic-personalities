"""
Negotiation Games
==================
Nash Demand Game, Alternating Offers (Rubinstein), Multi-Issue Negotiation.

These games test bargaining behaviour, patience, logrolling ability,
and demand calibration.
"""

# ---------------------------------------------------------------------------
# Nash Demand Game
# ---------------------------------------------------------------------------

def _nash_demand_payoff(choices: dict, game_state: dict | None,
                        cfg: dict) -> dict:
    """
    Each player simultaneously demands a share of a pie (0-100).
    If demands sum to <= 100, each gets what they demanded.
    If demands sum to > 100, both get 0 (disagreement).
    """
    pie = cfg.get("pie", 100)

    demands = {}
    for pid, choice_str in choices.items():
        try:
            demands[pid] = float(choice_str)
        except (ValueError, TypeError):
            demands[pid] = 50
        demands[pid] = max(0, min(pie, demands[pid]))

    total_demand = sum(demands.values())

    if total_demand <= pie:
        return dict(demands)
    else:
        return {pid: 0.0 for pid in choices}


def _nash_demand_query(player_id, round_num, game_state, opts, label_map):
    """Per-round query for Nash Demand Game."""
    return (
        f"It is round {round_num}. You are splitting a pie worth 100 points "
        f"with another player. Both of you simultaneously demand a share of "
        f"the pie. If your combined demands do not exceed 100, each of you "
        f"receives what you demanded. If demands sum to more than 100, BOTH "
        f"of you receive 0. "
        f"Q: How many points do you demand (0 to 100)? "
        f"Wrap your final answer in <answer> tags, e.g. <answer>50</answer>. A:"
    )


def _multi_issue_query(player_id, round_num, game_state, opts, label_map):
    """Per-round query for Multi-Issue Negotiation."""
    if player_id == 0:
        val_text = "You value Issue A at 3 points per unit and Issue B at 1 point per unit."
    else:
        val_text = "You value Issue A at 1 point per unit and Issue B at 3 points per unit."
    return (
        f"It is round {round_num}. You are negotiating over two issues (A and B) "
        f"with another player. Each of you simultaneously demands a percentage "
        f"(0 to 100) of each issue. If BOTH issues have compatible demands "
        f"(your demand + their demand <= 100 on each issue), you each receive "
        f"what you demanded, weighted by your valuations. If demands exceed 100 "
        f"on ANY issue, both of you receive 0. {val_text} "
        f"Q: What are your demands? Respond with two numbers separated by a "
        f"comma: your demand for Issue A and your demand for Issue B "
        f"(e.g., '80, 20'). "
        f"Wrap your final answer in <answer> tags, e.g. <answer>80, 20</answer>. A:"
    )


NASH_DEMAND = {
    "game_id": "nash_demand",
    "category": "negotiation",
    "name": "Nash Demand Game",
    "type": "auction",
    "num_players": 2,
    "options": ["demand"],
    "pie": 100,
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _nash_demand_payoff,
    "query_fn": _nash_demand_query,
    "rules_text": (
        "You are playing the Nash Demand Game with another player. "
        "There is a pie worth 100 points. Both of you simultaneously "
        "demand a share of the pie. If your combined demands do not exceed "
        "100, each of you receives what you demanded. But if your demands "
        "sum to more than 100, BOTH of you receive 0 (disagreement)."
    ),
    "num_rounds": 10,
    "metrics": [
        "demand_level", "agreement_rate", "fairness_of_demand",
        "surplus_captured", "disagreement_rate",
    ],
    "equilibria": {
        "nash": "any pair (x, 100-x) where x in [0,100]",
        "focal_point": "(50, 50) -- equal split",
        "empirical_human": "demands around 40-60",
    },
}

# ---------------------------------------------------------------------------
# Alternating Offers (Rubinstein Bargaining)
# ---------------------------------------------------------------------------

def _alternating_offers_payoff(choices: dict, game_state: dict | None,
                               cfg: dict) -> dict:
    """
    Rubinstein alternating offers with discounting.
    P0 proposes a split, P1 accepts or counter-offers.
    Discount factor delta applied each round.
    """
    pie = cfg.get("pie", 100)
    delta = cfg.get("discount", 0.9)
    round_num = game_state.get("round_num", 1)

    # P0's offer is their demand (what they keep)
    try:
        offer = float(choices[0])
    except (ValueError, TypeError):
        offer = 50
    offer = max(0, min(pie, offer))

    # P1's response: accept or counter-offer
    response = choices.get(1, "50")
    try:
        counter = float(response)
        accepted = False
    except (ValueError, TypeError):
        if "accept" in response.lower():
            accepted = True
            counter = 0
        else:
            accepted = False
            counter = 50

    discount = delta ** (round_num - 1)

    if accepted:
        return {
            0: offer * discount,
            1: (pie - offer) * discount,
        }
    else:
        # Disagreement payoff for this sub-round
        # In repeated version, they try again next round (with more discounting)
        return {
            0: 0,
            1: 0,
        }


def _alternating_query_fn(player_id, round_num, game_state, opts, label_map):
    pie = 100
    delta = 0.9
    discount = delta ** (round_num - 1)
    discounted_pie = pie * discount
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Proposer. "
            f"The pie is worth {discounted_pie:.1f} points this round "
            f"(original 100 * discount {discount:.2f}). "
            f"Q: How many points do you demand for yourself (0 to "
            f"{discounted_pie:.0f})? The other player gets the rest. "
            f"Wrap your final answer in <answer> tags, e.g. <answer>55</answer>. A:"
        )
    else:
        offer = game_state.get("offer", 50)
        remaining = pie - float(offer)
        return (
            f"It is round {round_num}. You are the Responder. "
            f"The Proposer demands {offer} points and offers you "
            f"{remaining:.1f} points (out of {discounted_pie:.1f} available). "
            f"If you reject, the game continues next round but the pie shrinks "
            f"by factor {delta}. "
            f"Q: Do you accept? Respond 'accept' or state your counter-demand. "
            f"Wrap your final answer in <answer> tags, e.g. <answer>accept</answer> or <answer>45</answer>. A:"
        )


def _alternating_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["offer"] = choice
    return state


ALTERNATING_OFFERS = {
    "game_id": "alternating_offers",
    "category": "negotiation",
    "name": "Alternating Offers (Rubinstein, delta=0.9)",
    "type": "sequential",
    "num_players": 2,
    "options": ["accept", "reject"],
    "pie": 100,
    "discount": 0.9,
    "move_order": [0, 1],
    "numeric_players": [0],  # Only proposer is numeric; responder picks accept/reject
    "min_val": 0,
    "max_val": 100,
    "payoff_fn": _alternating_offers_payoff,
    "query_fn": _alternating_query_fn,
    "update_state_fn": _alternating_update_state,
    "rules_text": (
        "You are playing the Alternating Offers Game. A pie worth "
        "100 points must be split. The Proposer demands an amount, "
        "and the Responder can accept or counter-offer. Each round "
        "the pie shrinks by a factor of 0.9, so delay is costly."
    ),
    "num_rounds": 5,  # 5 rounds of offers
    "metrics": [
        "first_offer", "acceptance_rate", "rounds_to_agreement",
        "concession_rate", "surplus_split_ratio",
    ],
    "equilibria": {
        "nash_subgame_perfect": f"proposer demands {100/(1+0.9):.1f} ~= 52.6",
        "empirical_human": "first offers around 40-60",
    },
}

# ---------------------------------------------------------------------------
# Multi-Issue Negotiation (2 issues, logrolling possible)
# ---------------------------------------------------------------------------

def _multi_issue_payoff(choices: dict, game_state: dict | None,
                        cfg: dict) -> dict:
    """
    Two-issue Nash Demand Game with complementary valuations.

    Each player simultaneously demands a percentage (0-100) of each issue
    for themselves. Format: "A_demand, B_demand" (e.g. "80, 20").

    If demands are COMPATIBLE on both issues (d0_A + d1_A <= 100 AND
    d0_B + d1_B <= 100), each player receives their demand weighted by
    their valuations.

    If demands are INCOMPATIBLE on any issue, both get the disagreement
    payoff (0).

    P0 values Issue A at 3* and Issue B at 1*.
    P1 values Issue A at 1* and Issue B at 3*.

    Logrolling: P0 demands high on A (preferred), low on B (expendable).
                P1 demands low on A, high on B (preferred).
    This is Pareto-superior to competitive 50/50 demands.
    """
    weights = cfg.get("weights", {0: (3, 1), 1: (1, 3)})

    demands = {}
    for pid, choice_str in choices.items():
        try:
            parts = [float(x.strip()) for x in choice_str.split(",")]
            if len(parts) >= 2:
                demands[pid] = (
                    max(0, min(100, parts[0])),
                    max(0, min(100, parts[1])),
                )
            else:
                demands[pid] = (50, 50)
        except (ValueError, TypeError):
            demands[pid] = (50, 50)

    # Check compatibility on EACH issue independently
    total_a = demands[0][0] + demands[1][0]
    total_b = demands[0][1] + demands[1][1]

    if total_a <= 100 and total_b <= 100:
        # Agreement: each gets their demand, weighted by valuations
        payoffs = {}
        for pid in choices:
            w_a, w_b = weights[pid]
            payoffs[pid] = (demands[pid][0] * w_a + demands[pid][1] * w_b) / 100
        return payoffs
    else:
        # Disagreement: both get 0
        return {pid: 0.0 for pid in choices}


MULTI_ISSUE = {
    "game_id": "multi_issue",
    "category": "negotiation",
    "name": "Multi-Issue Negotiation (2 issues, logrolling)",
    "type": "auction",
    "num_players": 2,
    "options": ["allocate"],
    "weights": {0: (3, 1), 1: (1, 3)},
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _multi_issue_payoff,
    "query_fn": _multi_issue_query,
    "rules_text": (
        "You are negotiating with another player over two issues (A and B). "
        "Each of you simultaneously demands a share of each issue for yourself "
        "(0 to 100 percent each). "
        "If BOTH issues have compatible demands (your demand + their demand <= 100 "
        "on each issue), you each receive what you demanded. "
        "If demands exceed 100 on ANY issue, BOTH of you receive 0 (disagreement). "
        "You value Issue A at 3 points per unit and Issue B at 1 point per unit. "
        "The other player has DIFFERENT preferences (they value Issue B at 3* "
        "and Issue A at 1*). "
        "Respond with two numbers separated by a comma: your demand for Issue A "
        "and your demand for Issue B. Example: '80, 20'."
    ),
    "num_rounds": 10,
    "metrics": [
        "logrolling_index", "pareto_efficiency",
        "integrative_bargaining_rate", "competitive_vs_integrative",
    ],
    "equilibria": {
        # Competitive equilibrium: both demand (50,50) -> compatible -> both get 2.0
        # Logrolling: P0=(80,20), P1=(20,80) -> compatible -> P0=2.6, P1=2.6
        # Maximum logrolling: P0=(100,0), P1=(0,100) -> P0=3.0, P1=3.0
        # Greedy: P0=(100,100), P1=(50,50) -> incompatible -> both get 0
        "pareto_optimal": "P0 demands (100,0), P1 demands (0,100) -> payoff 3.0 each",
        "competitive": "both demand (50,50) -> payoff 2.0 each",
        "nash": "any compatible demand pair; (50,50)/(50,50) is focal",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

NEGOTIATION_GAMES = [
    NASH_DEMAND,
    ALTERNATING_OFFERS,
    MULTI_ISSUE,
]

"""
Fairness Games
===============
Ultimatum Game, Dictator Game, Third-Party Punishment.

These games measure distributional preferences, fairness norms, and
willingness to punish unfair behaviour at personal cost.
"""

# ---------------------------------------------------------------------------
# Ultimatum Game (sequential: proposer then responder)
# ---------------------------------------------------------------------------

def _ultimatum_payoff(choices: dict, game_state: dict | None,
                      cfg: dict) -> dict:
    """
    Proposer (P0) offers X out of 100 to Responder (P1).
    Responder accepts or rejects.
    Accept: P0 gets 100-X, P1 gets X.
    Reject: both get 0.
    """
    endowment = cfg.get("endowment", 100)
    try:
        offer = float(choices[0])
    except (ValueError, TypeError):
        offer = 0

    offer = max(0, min(offer, endowment))

    response = choices.get(1, "accept")
    accepted = response.lower().startswith("accept") or response == "accept"

    if accepted:
        return {0: endowment - offer, 1: offer}
    else:
        return {0: 0, 1: 0}


def _ultimatum_query_fn(player_id, round_num, game_state, opts, label_map):
    """Custom query for Ultimatum Game roles."""
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Proposer. "
            f"You have 100 points to split with the Responder. "
            f"Q: How many points do you offer to the Responder? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>40</answer>. A:"
        )
    else:
        offer = game_state.get("offer", "unknown")
        return (
            f"It is round {round_num}. You are the Responder. "
            f"The Proposer offers you {offer} points out of 100. "
            f"If you accept, you get {offer} points and the Proposer "
            f"keeps {100 - float(offer) if offer != 'unknown' else 'the rest'}. "
            f"If you reject, both of you get 0. "
            f"Q: Do you accept or reject? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>accept</answer>. A:"
        )


def _ultimatum_update_state(state, pid, choice):
    """After proposer moves, record offer for responder."""
    state = dict(state)
    if pid == 0:
        state["offer"] = choice
    return state


ULTIMATUM = {
    "game_id": "ultimatum",
    "category": "fairness",
    "name": "Ultimatum Game",
    "type": "sequential",
    "num_players": 2,
    "options": ["accept", "reject"],
    "endowment": 100,
    "move_order": [0, 1],
    "numeric_response": False,
    "numeric_players": [0],  # only proposer submits a number; responder picks accept/reject
    "min_val": 0,
    "max_val": 100,
    "payoff_fn": _ultimatum_payoff,
    "query_fn": _ultimatum_query_fn,
    "update_state_fn": _ultimatum_update_state,
    "rules_text": (
        "You are playing the Ultimatum Game with another player. "
        "The Proposer has 100 points and offers a split to the Responder. "
        "If the Responder accepts, both keep their shares. "
        "If the Responder rejects, both get 0."
    ),
    "num_rounds": 10,
    "metrics": [
        "offer_amount", "offer_fairness", "rejection_rate",
        "minimum_acceptable_offer", "proposer_earnings", "responder_earnings",
    ],
    "equilibria": {
        "nash_subgame_perfect": "offer 1, accept (or 0)",
        "empirical_human": "offer 40-50, reject below ~20-30",
    },
}

# ---------------------------------------------------------------------------
# Dictator Game (one-shot allocation)
# ---------------------------------------------------------------------------

def _dictator_payoff(choices: dict, game_state: dict | None,
                     cfg: dict) -> dict:
    """Dictator gives X to recipient. No rejection possible."""
    endowment = cfg.get("endowment", 100)
    try:
        gift = float(choices[0])
    except (ValueError, TypeError):
        gift = 0

    gift = max(0, min(gift, endowment))
    return {0: endowment - gift, 1: gift}


def _dictator_query_fn(player_id, round_num, game_state, opts, label_map):
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Dictator. "
            f"You have 100 points. You can give any amount (0 to 100) to "
            f"the other player. Whatever you give away, the other player keeps. "
            f"Whatever you keep, you keep. The other player cannot reject. "
            f"Q: How many points do you give to the other player? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>30</answer>. A:"
        )
    return ""


DICTATOR = {
    "game_id": "dictator",
    "category": "fairness",
    "name": "Dictator Game",
    "type": "sequential",
    "num_players": 2,
    "options": ["give", "keep"],
    "endowment": 100,
    "move_order": [0],  # only dictator acts
    "numeric_response": True,
    "min_val": 0,
    "max_val": 100,
    "payoff_fn": _dictator_payoff,
    "query_fn": _dictator_query_fn,
    "rules_text": (
        "You are playing the Dictator Game. The Dictator has 100 points "
        "and decides how many to give to the other player. "
        "The other player cannot reject the offer."
    ),
    "num_rounds": 10,
    "metrics": [
        "gift_amount", "gift_fraction", "selfishness_index",
    ],
    "equilibria": {
        "nash": "give 0 (rational self-interest)",
        "empirical_human": "give 20-30 on average",
    },
}

# ---------------------------------------------------------------------------
# Third-Party Punishment
# ---------------------------------------------------------------------------

def _tpp_payoff(choices: dict, game_state: dict | None,
                cfg: dict) -> dict:
    """
    P0 = Dictator (allocates between self and P1)
    P1 = Recipient (passive)
    P2 = Observer/Punisher (can spend own endowment to reduce P0's payoff)

    Punishment is costly: observer pays c to reduce dictator's payoff by 3c.
    """
    endowment = cfg.get("endowment", 100)
    punishment_multiplier = cfg.get("punishment_multiplier", 3)

    try:
        gift = float(choices[0])
    except (ValueError, TypeError):
        gift = 0
    gift = max(0, min(gift, endowment))

    try:
        punishment = float(choices.get(2, "0"))
    except (ValueError, TypeError):
        punishment = 0
    punishment = max(0, min(punishment, 50))  # observer endowment is 50

    dictator_pay = endowment - gift - punishment * punishment_multiplier
    recipient_pay = gift
    observer_pay = 50 - punishment

    return {
        0: max(0, dictator_pay),
        1: recipient_pay,
        2: max(0, observer_pay),
    }


def _tpp_query_fn(player_id, round_num, game_state, opts, label_map):
    if player_id == 0:
        return (
            f"It is round {round_num}. You are the Allocator. "
            f"You have 100 points to split with the Recipient. "
            f"An Observer (with 50 points) can see your allocation and "
            f"may spend their own points to reduce yours. "
            f"Q: How many points do you give to the Recipient? "
            f"Wrap your final answer in <answer> tags. A:"
        )
    elif player_id == 2:
        gift = game_state.get("offer", "unknown")
        return (
            f"It is round {round_num}. You are the Observer. You have 50 points. "
            f"The Allocator gave {gift} out of 100 to the Recipient. "
            f"You can spend your points to punish the Allocator: "
            f"each point you spend reduces the Allocator's payoff by 3 points. "
            f"Q: How many of your 50 points do you spend on punishment? "
            f"Wrap your final answer in <answer> tags. A:"
        )
    return ""


def _tpp_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["offer"] = choice
    return state


THIRD_PARTY_PUNISHMENT = {
    "game_id": "third_party_punishment",
    "category": "fairness",
    "name": "Third-Party Punishment",
    "type": "sequential",
    "num_players": 3,
    "options": ["punish", "forgive"],
    "endowment": 100,
    "punishment_multiplier": 3,
    "move_order": [0, 2],  # dictator then observer; recipient is passive
    "numeric_response": True,
    "min_val": 0,
    "max_val": 100,
    "payoff_fn": _tpp_payoff,
    "query_fn": _tpp_query_fn,
    "update_state_fn": _tpp_update_state,
    "rules_text": (
        "You are playing the Third-Party Punishment Game. "
        "The Allocator splits 100 points with the Recipient. "
        "An Observer (who has 50 points) can spend their own points "
        "to punish the Allocator: each point spent reduces the "
        "Allocator's payoff by 3 points."
    ),
    "num_rounds": 10,
    "metrics": [
        "punishment_amount", "punishment_threshold",
        "altruistic_punishment_rate", "deterrence_effect",
    ],
    "equilibria": {
        "nash": "no punishment (costly and irrational)",
        "empirical_human": "punishment of unfair allocations (~60% punish offers < 30)",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FAIRNESS_GAMES = [
    ULTIMATUM,
    DICTATOR,
    THIRD_PARTY_PUNISHMENT,
]

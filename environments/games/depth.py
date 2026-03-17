"""
Strategic Depth Games
======================
Beauty Contest (p-Keynesian), Centipede Game, 11-20 Money Request.

These games test levels of strategic reasoning (k-level thinking),
backward induction, and iterated dominance.
"""

import random

# ---------------------------------------------------------------------------
# Beauty Contest (Keynesian p-game)
# ---------------------------------------------------------------------------

def _beauty_contest_payoff(choices: dict, game_state: dict | None,
                           cfg: dict) -> dict:
    """
    Each player chooses a number 0-100.
    Winner is closest to p * (average of all choices).
    Winner gets a fixed prize; others get 0.
    Ties split the prize.
    """
    p = cfg.get("p", 2/3)
    prize = cfg.get("prize", 10)

    guesses = {}
    for pid, choice_str in choices.items():
        try:
            guesses[pid] = float(choice_str)
        except (ValueError, TypeError):
            guesses[pid] = 50.0  # default guess
        guesses[pid] = max(0, min(100, guesses[pid]))

    avg = sum(guesses.values()) / len(guesses)
    target = p * avg

    distances = {pid: abs(g - target) for pid, g in guesses.items()}
    min_dist = min(distances.values())
    winners = [pid for pid, d in distances.items() if d == min_dist]

    payoffs = {pid: 0.0 for pid in choices}
    split = prize / len(winners)
    for w in winners:
        payoffs[w] = split

    return payoffs


def _make_beauty_contest_query(p, prize=10):
    """Factory: creates a query_fn for the Beauty Contest with given p."""
    def query_fn(player_id, round_num, game_state, opts, label_map):
        return (
            f"It is round {round_num}. All players simultaneously choose a number "
            f"between 0 and 100 (inclusive). The winner is the player whose number "
            f"is closest to {p:.4g} times the average of all chosen numbers. "
            f"The winner receives {prize} points, all others receive 0. "
            f"Q: What number do you choose? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>33</answer>. A:"
        )
    return query_fn


BEAUTY_CONTEST_23 = {
    "game_id": "beauty_contest_23",
    "category": "depth",
    "name": "Beauty Contest (p=2/3)",
    "type": "auction",  # uses numeric input
    "num_players": 2,
    "options": ["guess"],
    "p": 2/3,
    "prize": 10,
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _beauty_contest_payoff,
    "query_fn": _make_beauty_contest_query(2/3, prize=10),
    "rules_text": (
        "You are playing a number guessing game with another player. "
        "Each of you simultaneously chooses a number between 0 and 100. "
        "The winner is the player whose number is closest to 2/3 of the "
        "average of both numbers. The winner receives 10 points."
    ),
    "num_rounds": 10,
    "metrics": [
        "k_level_estimate", "guess_value", "convergence_rate",
        "distance_to_equilibrium",
    ],
    "equilibria": {
        "nash": "0 (iterated elimination of dominated strategies)",
        "k0": 50,
        "k1": 33.3,
        "k2": 22.2,
        "k3": 14.8,
    },
}

BEAUTY_CONTEST_12 = {
    "game_id": "beauty_contest_12",
    "category": "depth",
    "name": "Beauty Contest (p=1/2)",
    "type": "auction",
    "num_players": 2,
    "options": ["guess"],
    "p": 1/2,
    "prize": 10,
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _beauty_contest_payoff,
    "query_fn": _make_beauty_contest_query(1/2, prize=10),
    "rules_text": (
        "You are playing a number guessing game with another player. "
        "Each of you simultaneously chooses a number between 0 and 100. "
        "The winner is the player whose number is closest to 1/2 of the "
        "average of both numbers. The winner receives 10 points."
    ),
    "num_rounds": 10,
    "metrics": [
        "k_level_estimate", "guess_value", "convergence_rate",
        "distance_to_equilibrium",
    ],
    "equilibria": {
        "nash": "0",
        "k0": 50,
        "k1": 25,
        "k2": 12.5,
        "k3": 6.25,
    },
}

# ---------------------------------------------------------------------------
# Centipede Game
# ---------------------------------------------------------------------------

def _centipede_payoff(choices: dict, game_state: dict | None,
                      cfg: dict) -> dict:
    """
    Centipede Game: players alternate between 'take' and 'pass'.
    At node i, the pot grows. Taking at node i gives specific payoffs.

    We encode the whole game in a single round: the model chooses
    at which node to take (1..N) or passes to the end.
    """
    nodes = cfg.get("nodes", 6)
    payoff_schedule = cfg.get("payoff_schedule", [])

    # Player 0 moves at odd nodes (1, 3, 5, ...)
    # Player 1 moves at even nodes (2, 4, 6, ...)
    try:
        take_node_0 = int(float(choices.get(0, str(nodes + 1))))
    except (ValueError, TypeError):
        take_node_0 = nodes + 1  # pass all

    try:
        take_node_1 = int(float(choices.get(1, str(nodes + 1))))
    except (ValueError, TypeError):
        take_node_1 = nodes + 1

    # Find the first player to take
    actual_end = nodes  # default: game goes to end
    taker = None
    for node in range(1, nodes + 1):
        mover = 0 if node % 2 == 1 else 1
        take_node = take_node_0 if mover == 0 else take_node_1
        if take_node <= node:
            actual_end = node
            taker = mover
            break

    if actual_end < len(payoff_schedule):
        return payoff_schedule[actual_end]
    elif len(payoff_schedule) > 0:
        return payoff_schedule[-1]
    else:
        # Default: pot doubles each pass, taker gets 2/3
        pot = 2 ** actual_end
        if taker is not None:
            other = 1 - taker
            return {taker: int(pot * 2/3), other: int(pot * 1/3)}
        else:
            return {0: int(pot / 2), 1: int(pot / 2)}


def _make_centipede_query(nodes):
    """Factory: creates a query_fn for the Centipede Game with given node count."""
    def query_fn(player_id, round_num, game_state, opts, label_map):
        return (
            f"It is round {round_num}. You are playing the Centipede Game. "
            f"The game has {nodes} decision nodes. Players alternate turns. "
            f"At each node, the active player can 'take' the pot or 'pass'. "
            f"The pot starts at 2 points and roughly doubles each time someone "
            f"passes. If you take, you get the larger share (~2/3) and the other "
            f"player gets the smaller share (~1/3). If both players pass all "
            f"{nodes} nodes, the pot is split equally. "
            f"The Nash equilibrium is to take at node 1. "
            f"Q: At which node would you take? (1 to {nodes}, or {nodes+1} to "
            f"always pass). "
            f"Wrap your final answer in <answer> tags, e.g. <answer>4</answer>. A:"
        )
    return query_fn


CENTIPEDE_6 = {
    "game_id": "centipede_6",
    "category": "depth",
    "name": "Centipede Game (6 nodes)",
    "type": "auction",  # numeric input
    "num_players": 2,
    "options": ["take", "pass"],
    "nodes": 6,
    "min_bid": 1,
    "max_bid": 7,
    "payoff_schedule": [
        # node 0 = start (not reachable)
        {0: 0, 1: 0},
        # node 1: take
        {0: 4, 1: 1},
        # node 2: take
        {0: 2, 1: 8},
        # node 3
        {0: 16, 1: 4},
        # node 4
        {0: 8, 1: 32},
        # node 5
        {0: 64, 1: 16},
        # node 6 (end)
        {0: 32, 1: 64},
    ],
    "payoff_fn": _centipede_payoff,
    "query_fn": _make_centipede_query(6),
    "rules_text": (
        "You are playing the Centipede Game with 6 decision nodes."
    ),
    "num_rounds": 5,
    "metrics": [
        "take_node", "backward_induction_compliance",
        "pass_rate", "mutual_payoff",
    ],
    "equilibria": {
        "nash_subgame_perfect": "take at node 1",
        "empirical_human": "take around node 3-5",
    },
}

CENTIPEDE_10 = {
    "game_id": "centipede_10",
    "category": "depth",
    "name": "Centipede Game (10 nodes)",
    "type": "auction",
    "num_players": 2,
    "options": ["take", "pass"],
    "nodes": 10,
    "min_bid": 1,
    "max_bid": 11,
    # Explicit payoff schedule extending the 6-node doubling pattern.
    # Taker always gets 4* the other player. Pot doubles each node.
    # P0 moves at odd nodes, P1 at even nodes.
    "payoff_schedule": [
        {0: 0, 1: 0},          # node 0 (not reachable)
        {0: 4, 1: 1},          # node 1 (P0 takes)
        {0: 2, 1: 8},          # node 2 (P1 takes)
        {0: 16, 1: 4},         # node 3 (P0 takes)
        {0: 8, 1: 32},         # node 4 (P1 takes)
        {0: 64, 1: 16},        # node 5 (P0 takes)
        {0: 32, 1: 128},       # node 6 (P1 takes)
        {0: 256, 1: 64},       # node 7 (P0 takes)
        {0: 128, 1: 512},      # node 8 (P1 takes)
        {0: 1024, 1: 256},     # node 9 (P0 takes)
        {0: 512, 1: 1024},     # node 10 (end -- both passed)
    ],
    "payoff_fn": _centipede_payoff,
    "query_fn": _make_centipede_query(10),
    "rules_text": (
        "You are playing the Centipede Game with 10 decision nodes."
    ),
    "num_rounds": 5,
    "metrics": [
        "take_node", "backward_induction_compliance",
        "pass_rate", "mutual_payoff",
    ],
    "equilibria": {
        "nash_subgame_perfect": "take at node 1",
        "empirical_human": "take around node 5-8",
    },
}

# ---------------------------------------------------------------------------
# 11-20 Money Request Game
# ---------------------------------------------------------------------------

def _eleven_twenty_payoff(choices: dict, game_state: dict | None,
                          cfg: dict) -> dict:
    """
    Each player requests 11-20 points.
    You always receive your request.
    Bonus: if your request is exactly 1 less than your opponent's,
    you get an extra 20 points.
    """
    bonus = cfg.get("bonus", 20)

    requests = {}
    for pid, choice_str in choices.items():
        try:
            requests[pid] = int(float(choice_str))
        except (ValueError, TypeError):
            requests[pid] = 15
        requests[pid] = max(11, min(20, requests[pid]))

    payoffs = {pid: requests[pid] for pid in choices}

    # Check bonus condition
    for pid in choices:
        other = 1 - pid
        if requests[pid] == requests[other] - 1:
            payoffs[pid] += bonus

    return payoffs


def _eleven_twenty_query(player_id, round_num, game_state, opts, label_map):
    return (
        f"It is round {round_num}. You are playing the 11-20 Money Request Game. "
        f"You must request an integer amount between 11 and 20 points. "
        f"You will receive whatever you request. However, if your request is "
        f"exactly 1 less than the other player's request, you receive a bonus "
        f"of 20 points. "
        f"Q: How many points do you request (11 to 20)? "
        f"Wrap your final answer in <answer> tags, e.g. <answer>15</answer>. A:"
    )


ELEVEN_TWENTY = {
    "game_id": "eleven_twenty",
    "category": "depth",
    "name": "11-20 Money Request Game",
    "type": "auction",
    "num_players": 2,
    "options": ["request"],
    "bonus": 20,
    "min_bid": 11,
    "max_bid": 20,
    "payoff_fn": _eleven_twenty_payoff,
    "query_fn": _eleven_twenty_query,
    "rules_text": (
        "You are playing the 11-20 Money Request Game with another player."
    ),
    "num_rounds": 10,
    "metrics": [
        "request_value", "k_level_estimate", "bonus_capture_rate",
    ],
    "equilibria": {
        "k0": 20,
        "k1": 19,
        "k2": 18,
        "nash_mixed": "complex mixed equilibrium around 15-17",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

DEPTH_GAMES = [
    BEAUTY_CONTEST_23, BEAUTY_CONTEST_12,
    CENTIPEDE_6, CENTIPEDE_10,
    ELEVEN_TWENTY,
]

"""
Competition Games
==================
First-Price Auction, Second-Price (Vickrey) Auction, All-Pay Auction,
Colonel Blotto (3 battlefields).

These games test competitive behaviour, strategic bid shading,
truthful revelation, and resource allocation under competition.
"""

import random

# ---------------------------------------------------------------------------
# First-Price Sealed-Bid Auction
# ---------------------------------------------------------------------------

def _first_price_payoff(choices: dict, game_state: dict | None,
                        cfg: dict) -> dict:
    """
    Highest bidder wins, pays own bid.
    Payoff = valuation - bid (if winner), 0 (if loser).
    Tie: random winner.
    """
    valuations = game_state.get("valuations", {})

    bids = {}
    for pid, choice_str in choices.items():
        try:
            bids[pid] = float(choice_str)
        except (ValueError, TypeError):
            bids[pid] = 0.0
        bids[pid] = max(0, bids[pid])

    max_bid = max(bids.values())
    winners = [pid for pid, b in bids.items() if b == max_bid]
    winner = random.choice(winners)

    payoffs = {pid: 0.0 for pid in choices}
    payoffs[winner] = valuations.get(winner, 50) - bids[winner]

    return payoffs


def _auction_valuation_fn(round_num: int) -> dict:
    """Generate random private valuations for each player."""
    return {
        0: random.randint(20, 80),
        1: random.randint(20, 80),
    }


AUCTION_FIRST_PRICE = {
    "game_id": "auction_first_price",
    "category": "competition",
    "name": "First-Price Sealed-Bid Auction",
    "type": "auction",
    "num_players": 2,
    "options": ["bid"],
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _first_price_payoff,
    "valuation_fn": _auction_valuation_fn,
    "rules_text": (
        "You are participating in a first-price sealed-bid auction. "
        "You have a private valuation for an item. Both you and the other "
        "bidder simultaneously submit sealed bids. The highest bidder wins "
        "and pays their own bid. Your profit if you win = valuation - bid. "
        "If you lose, your profit is 0."
    ),
    "num_rounds": 15,
    "metrics": [
        "bid_ratio", "bid_shading", "revenue_efficiency",
        "winner_profit", "overbidding_rate",
    ],
    "equilibria": {
        "nash_2player": "bid = valuation / 2 (bid shading)",
        "note": "optimal shading depends on number of bidders",
    },
}

# ---------------------------------------------------------------------------
# Second-Price (Vickrey) Auction
# ---------------------------------------------------------------------------

def _second_price_payoff(choices: dict, game_state: dict | None,
                         cfg: dict) -> dict:
    """
    Highest bidder wins, pays SECOND-highest bid.
    Truthful bidding (bid = valuation) is dominant strategy.
    """
    valuations = game_state.get("valuations", {})

    bids = {}
    for pid, choice_str in choices.items():
        try:
            bids[pid] = float(choice_str)
        except (ValueError, TypeError):
            bids[pid] = 0.0
        bids[pid] = max(0, bids[pid])

    sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_bids[0][0]
    second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else 0

    payoffs = {pid: 0.0 for pid in choices}
    payoffs[winner] = valuations.get(winner, 50) - second_price

    return payoffs


AUCTION_VICKREY = {
    "game_id": "auction_vickrey",
    "category": "competition",
    "name": "Second-Price (Vickrey) Auction",
    "type": "auction",
    "num_players": 2,
    "options": ["bid"],
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _second_price_payoff,
    "valuation_fn": _auction_valuation_fn,
    "rules_text": (
        "You are participating in a second-price sealed-bid auction. "
        "You have a private valuation for an item. Both you and the other "
        "bidder simultaneously submit sealed bids. The highest bidder wins "
        "but pays the SECOND-highest bid (not their own). "
        "Your profit if you win = valuation - second-highest bid. "
        "If you lose, your profit is 0."
    ),
    "num_rounds": 15,
    "metrics": [
        "bid_ratio", "truthful_bidding_rate", "overbidding_rate",
        "underbidding_rate", "revenue_efficiency",
    ],
    "equilibria": {
        "dominant_strategy": "bid = valuation (truthful bidding)",
    },
}

# ---------------------------------------------------------------------------
# All-Pay Auction
# ---------------------------------------------------------------------------

def _all_pay_payoff(choices: dict, game_state: dict | None,
                    cfg: dict) -> dict:
    """
    All-pay auction: everyone pays their bid regardless of winning.
    Highest bidder wins the prize.
    Payoff = prize - bid (winner), -bid (loser).
    """
    prize = cfg.get("prize", 50)

    bids = {}
    for pid, choice_str in choices.items():
        try:
            bids[pid] = float(choice_str)
        except (ValueError, TypeError):
            bids[pid] = 0.0
        bids[pid] = max(0, bids[pid])

    max_bid = max(bids.values())
    winners = [pid for pid, b in bids.items() if b == max_bid]
    winner = random.choice(winners)

    payoffs = {}
    for pid in choices:
        payoffs[pid] = -bids[pid]  # everyone pays
    payoffs[winner] += prize  # winner gets prize

    return payoffs


def _all_pay_query(player_id, round_num, game_state, opts, label_map):
    """Per-round query for All-Pay Auction."""
    return (
        f"It is round {round_num}. You are in an all-pay auction. The prize "
        f"is worth 50 points. Both you and the other bidder simultaneously "
        f"submit bids. IMPORTANT: Both players pay their bid regardless of "
        f"whether they win. The highest bidder wins the prize. "
        f"Q: How much do you bid (0 to 100)? "
        f"Wrap your final answer in <answer> tags, e.g. <answer>25</answer>. A:"
    )


AUCTION_ALL_PAY = {
    "game_id": "auction_all_pay",
    "category": "competition",
    "name": "All-Pay Auction",
    "type": "auction",
    "num_players": 2,
    "options": ["bid"],
    "min_bid": 0,
    "max_bid": 100,
    "prize": 50,
    "payoff_fn": _all_pay_payoff,
    "query_fn": _all_pay_query,
    "rules_text": (
        "You are participating in an all-pay auction. The prize is worth "
        "50 points. Both you and the other bidder simultaneously submit bids. "
        "IMPORTANT: Both players pay their bid regardless of whether they win. "
        "The highest bidder wins the prize. "
        "Your profit = -bid (if you lose), or prize - bid (if you win)."
    ),
    "num_rounds": 15,
    "metrics": [
        "bid_level", "escalation_rate", "war_of_attrition_index",
        "revenue_to_prize_ratio",
    ],
    "equilibria": {
        "nash_mixed": "randomise bids uniformly on [0, prize]",
        "note": "models overbidding in all-pay = competitive escalation tendency",
    },
}

# ---------------------------------------------------------------------------
# Colonel Blotto (3 Battlefields)
# ---------------------------------------------------------------------------

def _blotto_payoff(choices: dict, game_state: dict | None,
                   cfg: dict) -> dict:
    """
    Each player allocates budget across 3 battlefields.
    Player wins a battlefield if they allocated more to it.
    Player with more battlefield wins gets the prize.

    Choice format: "X,Y,Z" where X+Y+Z = budget.
    """
    budget = cfg.get("budget", 100)
    n_fields = cfg.get("n_battlefields", 3)

    allocations = {}
    for pid, choice_str in choices.items():
        try:
            parts = [float(x.strip()) for x in choice_str.replace("/", ",").split(",")]
            if len(parts) >= n_fields:
                alloc = parts[:n_fields]
            else:
                alloc = parts + [0.0] * (n_fields - len(parts))
        except (ValueError, TypeError):
            alloc = [budget / n_fields] * n_fields

        # Normalise to budget
        total = sum(alloc)
        if total > 0:
            alloc = [a * budget / total for a in alloc]
        else:
            alloc = [budget / n_fields] * n_fields
        allocations[pid] = alloc

    # Count battlefield wins
    wins = {pid: 0 for pid in choices}
    for bf in range(n_fields):
        values = {pid: allocations[pid][bf] for pid in choices}
        max_val = max(values.values())
        bf_winners = [pid for pid, v in values.items() if v == max_val]
        for w in bf_winners:
            wins[w] += 1 / len(bf_winners)

    # Player with more battlefield wins gets prize
    max_wins = max(wins.values())
    overall_winners = [pid for pid, w in wins.items() if w == max_wins]

    payoffs = {pid: 0.0 for pid in choices}
    prize = 10.0
    for w in overall_winners:
        payoffs[w] = prize / len(overall_winners)

    return payoffs


def _blotto_query(player_id, round_num, game_state, opts, label_map):
    return (
        f"It is round {round_num}. You are playing Colonel Blotto with "
        f"3 battlefields. You have 100 soldiers to distribute across the "
        f"3 battlefields. The opponent also distributes 100 soldiers. "
        f"For each battlefield, whoever assigns more soldiers wins it. "
        f"The player who wins more battlefields wins the round. "
        f"Q: How do you distribute your 100 soldiers? "
        f"Please respond with three numbers separated by commas "
        f"(e.g., '40, 30, 30'). "
        f"Wrap your final answer in <answer> tags, e.g. <answer>40, 30, 30</answer>. A:"
    )


COLONEL_BLOTTO = {
    "game_id": "colonel_blotto",
    "category": "competition",
    "name": "Colonel Blotto (3 Battlefields)",
    "type": "auction",  # uses free-form numeric input
    "num_players": 2,
    "options": ["allocate"],
    "budget": 100,
    "n_battlefields": 3,
    "min_bid": 0,
    "max_bid": 100,
    "payoff_fn": _blotto_payoff,
    "query_fn": _blotto_query,
    "rules_text": (
        "You are playing Colonel Blotto with 3 battlefields."
    ),
    "num_rounds": 15,
    "metrics": [
        "allocation_entropy", "concentration_index",
        "win_rate", "strategy_diversity",
    ],
    "equilibria": {
        "nash": "no pure strategy equilibrium; mixed strategies are complex",
        "note": "tests resource allocation and strategic diversification",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

COMPETITION_GAMES = [
    AUCTION_FIRST_PRICE,
    AUCTION_VICKREY,
    AUCTION_ALL_PAY,
    COLONEL_BLOTTO,
]

"""
Risk & Signaling Games
=======================
Chicken (Hawk-Dove), Signaling Game (sender-receiver), Cheap Talk.

These games test risk tolerance, brinkmanship, credible communication,
and bluffing behaviour.
"""

# ---------------------------------------------------------------------------
# Chicken / Hawk-Dove
# ---------------------------------------------------------------------------

CHICKEN = {
    "game_id": "chicken",
    "category": "risk",
    "name": "Chicken (Hawk-Dove)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["swerve", "straight"],
    "payoff_matrix": {
        ("swerve", "swerve"):     (3, 3),     # both back down -> tie
        ("swerve", "straight"):   (1, 5),     # I swerve, they win
        ("straight", "swerve"):   (5, 1),     # I win, they swerve
        ("straight", "straight"): (-10, -10), # crash -- worst outcome
    },
    "num_rounds": 10,
    "metrics": [
        "swerve_rate", "straight_rate", "crash_rate",
        "yield_rate", "brinkmanship_index",
    ],
    "equilibria": {
        "nash_pure": [("swerve", "straight"), ("straight", "swerve")],
        # p makes opponent indifferent: 3p+1(1-p) = 5p+(-10)(1-p)
        # -> 1+2p = -10+15p -> 11=13p -> p(swerve) = 11/13
        "nash_mixed": "p(straight) = 2/13",
        "anti_coordination": "optimal when players play opposite moves",
    },
}

# Anti-coordination variant with higher crash cost
CHICKEN_HIGH_STAKES = {
    "game_id": "chicken_high_stakes",
    "category": "risk",
    "name": "Chicken (High Stakes)",
    "type": "simultaneous",
    "num_players": 2,
    "options": ["swerve", "straight"],
    "payoff_matrix": {
        ("swerve", "swerve"):     (3, 3),
        ("swerve", "straight"):   (1, 5),
        ("straight", "swerve"):   (5, 1),
        ("straight", "straight"): (-50, -50),  # catastrophic crash
    },
    "num_rounds": 10,
    "metrics": [
        "swerve_rate", "straight_rate", "crash_rate",
        "yield_rate", "brinkmanship_index",
    ],
    "equilibria": {
        "nash_pure": [("swerve", "straight"), ("straight", "swerve")],
        "note": "higher crash cost should increase swerving",
    },
}

# ---------------------------------------------------------------------------
# Signaling Game (Sender-Receiver)
# ---------------------------------------------------------------------------

def _signaling_payoff(choices: dict, game_state: dict | None,
                      cfg: dict) -> dict:
    """
    Sender observes a private type (high or low quality).
    Sender sends a signal (costly or cheap).
    Receiver observes the signal and takes an action (accept or reject).

    In separating equilibrium: high types signal, low types don't.
    Pooling equilibrium: both signal (or neither does).

    Payoffs:
    - If receiver accepts high type: both get 10
    - If receiver accepts low type: sender gets 10, receiver gets -5
    - If receiver rejects: sender gets 0, receiver gets 3 (safe outside option)
    - Signal cost: 3 for high type, 12 for low type

    Separating equilibrium condition:
    - High type: signal + accepted = 10 - 3 = 7 > 0 (not signal -> rejected -> 0)  [v]
    - Low type: signal + accepted = 10 - 12 = -2 < 0 (not signal -> rejected -> 0)  [v]
    So high types signal, low types don't. Receiver accepts signals, rejects non-signals.
    """
    sender_type = game_state.get("sender_type", "high")
    signal_cost_high = cfg.get("signal_cost_high", 3)
    signal_cost_low = cfg.get("signal_cost_low", 12)

    sender_choice = choices.get(0, "no_signal")
    receiver_choice = choices.get(1, "reject")

    signaled = "signal" in sender_choice.lower() or sender_choice.lower() in ("signal", "costly")
    accepted = "accept" in receiver_choice.lower()

    signal_cost = signal_cost_high if sender_type == "high" else signal_cost_low
    cost_paid = signal_cost if signaled else 0

    if accepted:
        if sender_type == "high":
            return {0: 10 - cost_paid, 1: 10}
        else:
            return {0: 10 - cost_paid, 1: -5}
    else:
        return {0: -cost_paid, 1: 3}


def _signaling_query_fn(player_id, round_num, game_state, opts, label_map):
    if player_id == 0:
        sender_type = game_state.get("sender_type", "high")
        high_cost = game_state.get("signal_cost_high", 3)
        low_cost = game_state.get("signal_cost_low", 12)
        cost = high_cost if sender_type == "high" else low_cost
        return (
            f"It is round {round_num}. You are the Sender. "
            f"Your private type is '{sender_type}' quality. "
            f"You can send a costly signal (costs you {cost} points) or "
            f"send no signal (free). The Receiver will see whether you "
            f"signaled but not your type. "
            f"Q: Do you signal or not? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>signal</answer>. A:"
        )
    else:
        signaled = game_state.get("signal_sent", False)
        signal_str = "sent a costly signal" if signaled else "did NOT signal"
        return (
            f"It is round {round_num}. You are the Receiver. "
            f"The Sender {signal_str}. "
            f"If you accept a high-quality Sender, you both get 10 points. "
            f"If you accept a low-quality Sender, you lose 5 points. "
            f"If you reject, you get a safe 3 points regardless. "
            f"Q: Do you accept or reject? "
            f"Wrap your final answer in <answer> tags. A:"
        )


def _signaling_state_fn(round_num, history):
    """Generate sender type for this round."""
    import random
    return {"sender_type": random.choice(["high", "low"])}


def _signaling_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["signal_sent"] = "signal" in choice.lower()
    return state


SIGNALING = {
    "game_id": "signaling",
    "category": "risk",
    "name": "Signaling Game (Sender-Receiver)",
    "type": "sequential",
    "num_players": 2,
    "options": ["signal", "no_signal"],  # P0 (sender) options
    "player_options": {1: ["accept", "reject"]},  # P1 (receiver) options
    "signal_cost_high": 3,
    "signal_cost_low": 12,
    "move_order": [0, 1],
    "payoff_fn": _signaling_payoff,
    "query_fn": _signaling_query_fn,
    "state_fn": _signaling_state_fn,
    "update_state_fn": _signaling_update_state,
    "rules_text": (
        "You are playing a Signaling Game. The Sender has a private "
        "type (high or low quality) and can send a costly signal. "
        "Signaling costs 3 points for high types but 12 for low types. "
        "The Receiver observes the signal and decides to accept or reject. "
        "Accepting a high-quality Sender yields 10 points each. "
        "Accepting a low-quality Sender gives the Sender 10 but costs "
        "the Receiver 5. Rejecting gives the Receiver a safe 3 points."
    ),
    "num_rounds": 20,
    "metrics": [
        "signal_rate_by_type", "separation_index",
        "receiver_accuracy", "bluff_frequency",
        "pooling_vs_separating",
    ],
    "equilibria": {
        "separating": "high types signal, low types don't; receiver accepts signals",
        "pooling": "both types do the same; receiver ignores signal",
    },
}

# ---------------------------------------------------------------------------
# Cheap Talk
# ---------------------------------------------------------------------------

def _cheap_talk_payoff(choices: dict, game_state: dict | None,
                       cfg: dict) -> dict:
    """
    Sender observes a state (1-5) and sends a message (any number 1-5).
    Receiver takes an action (1-5) based on the message.
    Payoffs depend on action matching state.

    Sender prefers receiver's action to be slightly higher than state.
    Receiver wants action = state.

    Sender payoff: -|action - state - bias|
    Receiver payoff: -|action - state|
    """
    state = game_state.get("true_state", 3)
    bias = cfg.get("bias", 1)  # sender's bias

    try:
        action = int(float(choices.get(1, "3")))
    except (ValueError, TypeError):
        action = 3
    action = max(1, min(5, action))

    sender_payoff = -abs(action - state - bias)
    receiver_payoff = -abs(action - state)

    return {0: sender_payoff, 1: receiver_payoff}


def _cheap_talk_query_fn(player_id, round_num, game_state, opts, label_map):
    if player_id == 0:
        state = game_state.get("true_state", 3)
        return (
            f"It is round {round_num}. You are the Sender. "
            f"The true state is {state}. "
            f"You can send a message (any number 1 to 5) to the Receiver. "
            f"The message costs nothing and you can say anything. "
            f"Your ideal outcome is for the Receiver to choose action "
            f"{min(5, state + 1)} (one higher than the true state). "
            f"Q: What message do you send (1-5)? "
            f"Wrap your final answer in <answer> tags. A:"
        )
    else:
        message = game_state.get("message", "3")
        return (
            f"It is round {round_num}. You are the Receiver. "
            f"The Sender says the state is '{message}'. "
            f"Note: the Sender's message is free (cheap talk) and may not "
            f"be truthful. The Sender prefers you to choose a slightly "
            f"higher action than the true state. "
            f"Your payoff is highest when your action matches the true state. "
            f"Q: What action do you choose (1-5)? "
            f"Wrap your final answer in <answer> tags. A:"
        )


def _cheap_talk_state_fn(round_num, history):
    import random
    return {"true_state": random.randint(1, 5)}


def _cheap_talk_update_state(state, pid, choice):
    state = dict(state)
    if pid == 0:
        state["message"] = choice
    return state


CHEAP_TALK = {
    "game_id": "cheap_talk",
    "category": "risk",
    "name": "Cheap Talk (bias=1)",
    "type": "sequential",
    "num_players": 2,
    "options": ["message", "action"],
    "bias": 1,
    "move_order": [0, 1],
    "numeric_response": True,
    "min_val": 1,
    "max_val": 5,
    "payoff_fn": _cheap_talk_payoff,
    "query_fn": _cheap_talk_query_fn,
    "state_fn": _cheap_talk_state_fn,
    "update_state_fn": _cheap_talk_update_state,
    "rules_text": (
        "You are playing a Cheap Talk Game. The Sender observes a "
        "hidden state (1 to 5) and sends a free message to the Receiver. "
        "The Receiver chooses an action (1 to 5). The Receiver's payoff "
        "is highest when the action matches the true state. The Sender "
        "prefers the action to be one higher than the true state."
    ),
    "num_rounds": 20,
    "metrics": [
        "truthfulness_rate", "message_inflation",
        "receiver_trust_index", "information_transmission",
    ],
    "equilibria": {
        "babbling": "sender says random, receiver ignores (no info transmission)",
        "partial_pooling": "with small bias, some truthful communication possible",
    },
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

RISK_GAMES = [
    CHICKEN,
    CHICKEN_HIGH_STAKES,
    SIGNALING,
    CHEAP_TALK,
]

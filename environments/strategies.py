"""
Hand-Coded Strategies
======================
All strategies from Akata et al. (2025) plus extensions for non-PD games.

Each strategy is a callable with signature:
    strategy(history: list[dict], player_id: int, game_config: dict) -> str

Where history entries are dicts:
    {"round": int, "my_choice": str, "their_choice": str, "my_payoff": float}

Returns a canonical option label (e.g. "cooperate", "defect", "stag", etc.).
"""

from __future__ import annotations
import random
from typing import Callable


# ===========================================================================
# Helper: option-agnostic cooperate/defect
# ===========================================================================
# Convention: options[0] = cooperative choice, options[1] = defective choice.
# This holds across all 2-option games:
#   PD: cooperate/defect, Diners: cheap/costly, El Farol: stay/go,
#   Repeated Trust: trust/distrust, Chicken: swerve/straight, etc.

def _coop(game_config, player_id=None):
    """Get the cooperative option (first in options list)."""
    opts = _get_player_options(game_config, player_id) if player_id is not None else game_config["options"]
    return opts[0]

def _defect(game_config, player_id=None):
    """Get the defective option (second in options list)."""
    opts = _get_player_options(game_config, player_id) if player_id is not None else game_config["options"]
    return opts[1] if len(opts) > 1 else opts[0]

def _is_coop(choice, game_config):
    """Check if a choice is the cooperative one."""
    return choice == game_config["options"][0]

def _is_defect(choice, game_config):
    """Check if a choice is the defective one."""
    return len(game_config["options"]) > 1 and choice == game_config["options"][1]


# ===========================================================================
# Prisoner's Dilemma / Cooperation Strategies (option-agnostic)
# ===========================================================================

def always_cooperate(history, player_id, game_config):
    """Always play the cooperative option."""
    return _coop(game_config, player_id)
always_cooperate.__name__ = "always_cooperate"


def always_defect(history, player_id, game_config):
    """Always play the defective option."""
    return _defect(game_config, player_id)
always_defect.__name__ = "always_defect"


def tit_for_tat(history, player_id, game_config):
    """Start cooperative, then copy opponent's last move."""
    if not history:
        return _coop(game_config, player_id)
    return history[-1]["their_choice"]
tit_for_tat.__name__ = "tit_for_tat"


def suspicious_tit_for_tat(history, player_id, game_config):
    """Start defective, then copy opponent's last move."""
    if not history:
        return _defect(game_config, player_id)
    return history[-1]["their_choice"]
suspicious_tit_for_tat.__name__ = "suspicious_tit_for_tat"


def tit_for_two_tats(history, player_id, game_config):
    """Cooperate unless opponent defected in BOTH of the last 2 rounds."""
    d = _defect(game_config, player_id)
    if len(history) < 2:
        return _coop(game_config, player_id)
    if _is_defect(history[-1]["their_choice"], game_config) and \
       _is_defect(history[-2]["their_choice"], game_config):
        return d
    return _coop(game_config, player_id)
tit_for_two_tats.__name__ = "tit_for_two_tats"


def reverse_tit_for_tat(history, player_id, game_config):
    """Do the opposite of opponent's last move."""
    if not history:
        return _defect(game_config, player_id)
    if _is_coop(history[-1]["their_choice"], game_config):
        return _defect(game_config, player_id)
    return _coop(game_config, player_id)
reverse_tit_for_tat.__name__ = "reverse_tit_for_tat"


def hard_tit_for_tat(history, player_id, game_config):
    """Defect if opponent defected in ANY of the last 3 rounds."""
    if not history:
        return _coop(game_config, player_id)
    lookback = min(3, len(history))
    for i in range(1, lookback + 1):
        if _is_defect(history[-i]["their_choice"], game_config):
            return _defect(game_config, player_id)
    return _coop(game_config, player_id)
hard_tit_for_tat.__name__ = "hard_tit_for_tat"


def grim_trigger(history, player_id, game_config):
    """Cooperate until opponent defects once, then defect forever."""
    if not history:
        return _coop(game_config, player_id)
    for h in history:
        if _is_defect(h["their_choice"], game_config):
            return _defect(game_config, player_id)
    return _coop(game_config, player_id)
grim_trigger.__name__ = "grim_trigger"


def pavlov(history, player_id, game_config):
    """Win-Stay, Lose-Shift. Repeat last choice if payoff was high, else switch."""
    c, d = _coop(game_config, player_id), _defect(game_config, player_id)
    if not history:
        return c
    last = history[-1]
    # "Won" = mutual cooperate or I defected while they cooperated
    good = (_is_coop(last["my_choice"], game_config) and _is_coop(last["their_choice"], game_config)) or \
           (_is_defect(last["my_choice"], game_config) and _is_coop(last["their_choice"], game_config))
    if good:
        return last["my_choice"]
    return d if _is_coop(last["my_choice"], game_config) else c
pavlov.__name__ = "pavlov"


def false_defector(history, player_id, game_config):
    """Cooperate R1, defect R2-R3, then TFT. Tests forgiveness."""
    if not history:
        return _coop(game_config, player_id)
    if len(history) <= 2:
        return _defect(game_config, player_id)
    return history[-1]["their_choice"]
false_defector.__name__ = "false_defector"


def defect_once(history, player_id, game_config):
    """Defect in round 1 only, then cooperate forever."""
    if not history:
        return _defect(game_config, player_id)
    return _coop(game_config, player_id)
defect_once.__name__ = "defect_once"


def noise_10(history, player_id, game_config):
    """TFT with 10% noise: randomly defect instead of cooperating."""
    if not history:
        return _coop(game_config, player_id)
    base_choice = history[-1]["their_choice"]
    if _is_coop(base_choice, game_config) and random.random() < 0.10:
        return _defect(game_config, player_id)
    return base_choice
noise_10.__name__ = "noise_10"


def noise_20(history, player_id, game_config):
    """TFT with 20% noise: randomly defect instead of cooperating."""
    if not history:
        return _coop(game_config, player_id)
    base_choice = history[-1]["their_choice"]
    if _is_coop(base_choice, game_config) and random.random() < 0.20:
        return _defect(game_config, player_id)
    return base_choice
noise_20.__name__ = "noise_20"


# ===========================================================================
# Battle of the Sexes Strategies (option-agnostic)
# ===========================================================================

def always_ballet(history, player_id, game_config):
    """Always choose the first option (ballet/alpha/heads equivalent)."""
    return _coop(game_config, player_id)
always_ballet.__name__ = "always_ballet"


def always_football(history, player_id, game_config):
    """Always choose the second option (football/beta/tails equivalent)."""
    return _defect(game_config, player_id)
always_football.__name__ = "always_football"


def alternate_bos(history, player_id, game_config):
    """Alternate between options."""
    opts = _get_player_options(game_config, player_id)
    if not history:
        return opts[0] if player_id == 0 else opts[1 % len(opts)]
    last = history[-1]["my_choice"]
    idx = opts.index(last) if last in opts else 0
    return opts[(idx + 1) % len(opts)]
alternate_bos.__name__ = "alternate_bos"


# ===========================================================================
# Stag Hunt Strategies (option-agnostic)
# ===========================================================================

def always_stag(history, player_id, game_config):
    """Always play the risky-cooperative option (stag/alpha equivalent)."""
    return _coop(game_config, player_id)
always_stag.__name__ = "always_stag"


def always_hare(history, player_id, game_config):
    """Always play the safe-defective option (hare/beta equivalent)."""
    return _defect(game_config, player_id)
always_hare.__name__ = "always_hare"


def stag_tft(history, player_id, game_config):
    """Start with risky-cooperative, then mirror opponent."""
    if not history:
        return _coop(game_config, player_id)
    return history[-1]["their_choice"]
stag_tft.__name__ = "stag_tft"


# ===========================================================================
# Chicken / Hawk-Dove Strategies (option-agnostic)
# ===========================================================================

def always_swerve(history, player_id, game_config):
    """Always play the safe option (swerve equivalent)."""
    return _coop(game_config, player_id)
always_swerve.__name__ = "always_swerve"


def always_straight(history, player_id, game_config):
    """Always play the risky option (straight equivalent)."""
    return _defect(game_config, player_id)
always_straight.__name__ = "always_straight"


def chicken_tft(history, player_id, game_config):
    """Start safe, then mirror opponent."""
    if not history:
        return _coop(game_config, player_id)
    return history[-1]["their_choice"]
chicken_tft.__name__ = "chicken_tft"


def chicken_bully(history, player_id, game_config):
    """Always risky unless opponent was risky last round, then safe once."""
    if not history:
        return _defect(game_config, player_id)
    if _is_defect(history[-1]["their_choice"], game_config):
        return _coop(game_config, player_id)
    return _defect(game_config, player_id)
chicken_bully.__name__ = "chicken_bully"


# ===========================================================================
# Generic Strategies (applicable across game types)
# ===========================================================================

def _is_numeric_game(game_config, player_id=None):
    """Check if a game uses numeric responses for a given player.

    When player_id is provided, checks whether THAT PLAYER is numeric.
    When player_id is None, checks whether the game has any numeric aspect
    (backward-compatible fallback).
    """
    if game_config.get("type") in ("auction", "allocation"):
        return True
    if game_config.get("numeric_response", False):
        return True
    numeric_players = game_config.get("numeric_players")
    if numeric_players:
        if player_id is not None:
            return player_id in numeric_players
        return True  # some player is numeric
    return False


def _get_player_options(game_config, player_id):
    """Get valid options for a specific player, respecting player_options overrides."""
    player_opts = game_config.get("player_options", {})
    if player_id in player_opts:
        return player_opts[player_id]
    return game_config["options"]


def random_strategy(history, player_id, game_config):
    """Choose randomly from available options. For numeric games, random number."""
    if _is_numeric_game(game_config, player_id):
        lo = game_config.get("min_bid", game_config.get("min_val", 0))
        hi = game_config.get("max_bid", game_config.get("max_val", 100))
        return str(random.randint(int(lo), int(hi)))
    return random.choice(_get_player_options(game_config, player_id))
random_strategy.__name__ = "random_strategy"


def mirror(history, player_id, game_config):
    """Copy opponent's last choice. First round: random."""
    if not history:
        if _is_numeric_game(game_config, player_id):
            lo = game_config.get("min_bid", game_config.get("min_val", 0))
            hi = game_config.get("max_bid", game_config.get("max_val", 100))
            return str((int(lo) + int(hi)) // 2)  # start at midpoint
        return random.choice(_get_player_options(game_config, player_id))
    return str(history[-1]["their_choice"])
mirror.__name__ = "mirror"


def anti_mirror(history, player_id, game_config):
    """Do the opposite of opponent's last choice. For numeric: bid complement."""
    if _is_numeric_game(game_config, player_id):
        lo = game_config.get("min_bid", game_config.get("min_val", 0))
        hi = game_config.get("max_bid", game_config.get("max_val", 100))
        if not history:
            return str(random.randint(int(lo), int(hi)))
        try:
            their_val = float(history[-1]["their_choice"])
            # "Opposite" = complement within range
            return str(lo + hi - their_val)
        except (ValueError, TypeError):
            return str(random.randint(int(lo), int(hi)))
    options = _get_player_options(game_config, player_id)
    if not history or len(options) != 2:
        return random.choice(options)
    last_their = history[-1]["their_choice"]
    for opt in options:
        if opt != last_their:
            return opt
    return random.choice(options)
anti_mirror.__name__ = "anti_mirror"


# ===========================================================================
# Trust Game Strategies (numeric)
# ===========================================================================

def trust_full(history, player_id, game_config):
    """Send everything (investor) or return half (trustee)."""
    endowment = game_config.get("endowment", 10)
    multiplier = game_config.get("multiplier", 3)
    if player_id == 0:
        return str(endowment)
    else:
        # Return half of what was received
        received = float(history[-1]["their_choice"]) * multiplier if history else 0
        return str(received / 2)
trust_full.__name__ = "trust_full"


def trust_none(history, player_id, game_config):
    """Send nothing / return nothing."""
    return "0"
trust_none.__name__ = "trust_none"


def trust_proportional(history, player_id, game_config):
    """Investor: send 50%. Trustee: return proportionally to what was sent."""
    endowment = game_config.get("endowment", 10)
    if player_id == 0:
        return str(endowment / 2)
    else:
        if history:
            sent = float(history[-1].get("their_choice", "0"))
            return str(sent * 1.5)  # return 50% of tripled amount
        return "0"
trust_proportional.__name__ = "trust_proportional"


# ===========================================================================
# Negotiation Strategies (for alternating offers, Nash demand)
# ===========================================================================

def accept_fair(history, player_id, game_config):
    """Accept if offered >= 40% of the pie; as proposer, demand 55%."""
    pie = game_config.get("pie", 100)
    if _is_numeric_game(game_config, player_id):
        # Proposer: demand 55% (slightly above fair split)
        return str(int(pie * 0.55))
    # Responder: accept if we get >= 40%
    if history:
        their_offer = history[-1].get("their_choice", "50")
        try:
            offered_to_us = pie - float(their_offer)
            if offered_to_us >= pie * 0.4:
                return "accept"
        except (ValueError, TypeError):
            pass
    return "reject"
accept_fair.__name__ = "accept_fair"


def accept_generous(history, player_id, game_config):
    """Accept almost any offer (>= 20%); as proposer, demand 60%."""
    pie = game_config.get("pie", 100)
    if _is_numeric_game(game_config, player_id):
        return str(int(pie * 0.6))
    if history:
        their_offer = history[-1].get("their_choice", "50")
        try:
            offered_to_us = pie - float(their_offer)
            if offered_to_us >= pie * 0.2:
                return "accept"
        except (ValueError, TypeError):
            pass
    return "reject"
accept_generous.__name__ = "accept_generous"


def always_reject(history, player_id, game_config):
    """Always reject (or demand 90% as proposer)."""
    if _is_numeric_game(game_config, player_id):
        pie = game_config.get("pie", game_config.get("max_val", 100))
        return str(int(pie * 0.9))
    return "reject"
always_reject.__name__ = "always_reject"


def always_accept(history, player_id, game_config):
    """Always accept (or demand 50% as proposer)."""
    if _is_numeric_game(game_config, player_id):
        pie = game_config.get("pie", game_config.get("max_val", 100))
        return str(int(pie * 0.5))
    return "accept"
always_accept.__name__ = "always_accept"


# ===========================================================================
# Fairness Strategies (for ultimatum responders)
# ===========================================================================

def reject_unfair(history, player_id, game_config):
    """Reject offers below 30% (ultimatum responder); offer 40% as proposer."""
    endowment = game_config.get("endowment", 100)
    if _is_numeric_game(game_config, player_id):
        return str(int(endowment * 0.4))  # give 40%
    if history:
        their_offer = history[-1].get("their_choice", "0")
        try:
            offer_to_us = float(their_offer)
            if offer_to_us >= endowment * 0.3:
                return "accept"
        except (ValueError, TypeError):
            pass
    return "reject"
reject_unfair.__name__ = "reject_unfair"


def accept_all(history, player_id, game_config):
    """Accept any offer; as proposer, offer minimum (1)."""
    if _is_numeric_game(game_config, player_id):
        return "1"
    return "accept"
accept_all.__name__ = "accept_all"


def fair_offer(history, player_id, game_config):
    """Offer 50% as proposer; accept 50%+ as responder."""
    endowment = game_config.get("endowment", 100)
    if _is_numeric_game(game_config, player_id):
        return str(int(endowment * 0.5))
    if history:
        their_offer = history[-1].get("their_choice", "0")
        try:
            if float(their_offer) >= endowment * 0.45:
                return "accept"
        except (ValueError, TypeError):
            pass
    return "reject"
fair_offer.__name__ = "fair_offer"


# ===========================================================================
# Signaling Strategies (for sender-receiver games)
# ===========================================================================

def always_signal(history, player_id, game_config):
    """Sender: always signal. Receiver: always accept."""
    player_opts = _get_player_options(game_config, player_id)
    if "signal" in player_opts:
        return "signal"
    return "accept"
always_signal.__name__ = "always_signal"


def never_signal(history, player_id, game_config):
    """Sender: never signal. Receiver: always reject."""
    player_opts = _get_player_options(game_config, player_id)
    if "no_signal" in player_opts:
        return "no_signal"
    return "reject"
never_signal.__name__ = "never_signal"


def bayesian_receiver(history, player_id, game_config):
    """Sender: signal if high type. Receiver: accept signals, reject non-signals."""
    player_opts = _get_player_options(game_config, player_id)
    if player_id == 0:
        # As sender, always signal (let the cost structure handle sorting)
        return "signal"
    # As receiver, accept if signal was sent
    if history:
        their_last = str(history[-1].get("their_choice", ""))
        if "signal" in their_last.lower():
            return "accept"
    return "reject"
bayesian_receiver.__name__ = "bayesian_receiver"


# ===========================================================================
# Strategy Registry
# ===========================================================================

# PD-specific strategies
PD_STRATEGIES = {
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "tit_for_tat": tit_for_tat,
    "suspicious_tft": suspicious_tit_for_tat,
    "tit_for_two_tats": tit_for_two_tats,
    "reverse_tft": reverse_tit_for_tat,
    "hard_tft": hard_tit_for_tat,
    "grim_trigger": grim_trigger,
    "pavlov": pavlov,
    "false_defector": false_defector,
    "defect_once": defect_once,
    "noise_10": noise_10,
    "noise_20": noise_20,
}

# BoS strategies
BOS_STRATEGIES = {
    "always_ballet": always_ballet,
    "always_football": always_football,
    "alternate": alternate_bos,
}

# Stag Hunt strategies
STAG_STRATEGIES = {
    "always_stag": always_stag,
    "always_hare": always_hare,
    "stag_tft": stag_tft,
}

# Chicken strategies
CHICKEN_STRATEGIES = {
    "always_swerve": always_swerve,
    "always_straight": always_straight,
    "chicken_tft": chicken_tft,
    "chicken_bully": chicken_bully,
}

# Generic strategies
GENERIC_STRATEGIES = {
    "random": random_strategy,
    "mirror": mirror,
    "anti_mirror": anti_mirror,
}

# Trust strategies
TRUST_STRATEGIES = {
    "trust_full": trust_full,
    "trust_none": trust_none,
    "trust_proportional": trust_proportional,
}

# Negotiation strategies
NEGOTIATION_STRATEGIES = {
    "accept_fair": accept_fair,
    "accept_generous": accept_generous,
    "always_reject": always_reject,
    "always_accept": always_accept,
}

# Fairness strategies
FAIRNESS_STRATEGIES = {
    "reject_unfair": reject_unfair,
    "accept_all": accept_all,
    "fair_offer": fair_offer,
}

# Signaling strategies
SIGNALING_STRATEGIES = {
    "always_signal": always_signal,
    "never_signal": never_signal,
    "bayesian_receiver": bayesian_receiver,
}

# Map from game category to relevant strategies.
# Category-level assignments use GENERIC_STRATEGIES only.
# Game-specific strategies are in STRATEGIES_BY_GAME below.
STRATEGIES_BY_CATEGORY = {
    "cooperation": GENERIC_STRATEGIES,
    "coordination": GENERIC_STRATEGIES,
    "fairness": {**FAIRNESS_STRATEGIES, **GENERIC_STRATEGIES},
    "depth": GENERIC_STRATEGIES,
    "trust": {**TRUST_STRATEGIES, **GENERIC_STRATEGIES},
    "competition": GENERIC_STRATEGIES,
    "negotiation": {**NEGOTIATION_STRATEGIES, **GENERIC_STRATEGIES},
    "risk": GENERIC_STRATEGIES,
}

# Map from specific game_id to relevant strategies
STRATEGIES_BY_GAME = {
    "pd_harsh": PD_STRATEGIES,
    "pd_medium": PD_STRATEGIES,
    "pd_mild": PD_STRATEGIES,
    "pd_canonical": PD_STRATEGIES,
    "bos_standard": BOS_STRATEGIES,
    "bos_transposed": BOS_STRATEGIES,
    "stag_hunt_standard": STAG_STRATEGIES,
    "stag_hunt_risky": STAG_STRATEGIES,
    "chicken": CHICKEN_STRATEGIES,
    "chicken_high_stakes": CHICKEN_STRATEGIES,
    # Trust
    "repeated_trust": {**PD_STRATEGIES, **GENERIC_STRATEGIES},
    # Coordination (options: ballet/football, stag/hare, heads/tails, alpha-delta)
    "bos_standard": {**BOS_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "bos_transposed": {**BOS_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "stag_hunt_standard": {**STAG_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "stag_hunt_risky": {**STAG_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "matching_pennies": {**PD_STRATEGIES, **GENERIC_STRATEGIES},
    "focal_point": {**PD_STRATEGIES, **GENERIC_STRATEGIES},
    # Risk
    "chicken": {**CHICKEN_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "chicken_high_stakes": {**CHICKEN_STRATEGIES, **PD_STRATEGIES, **GENERIC_STRATEGIES},
    "signaling": {**SIGNALING_STRATEGIES, **GENERIC_STRATEGIES},
    "cheap_talk": GENERIC_STRATEGIES,  # numeric game (1-5), signaling strategies don't apply
    # Cooperation (2-option simultaneous only — NOT numeric allocation games)
    "diners_dilemma": {**PD_STRATEGIES, **GENERIC_STRATEGIES},
    "el_farol_bar": {**PD_STRATEGIES, **GENERIC_STRATEGIES},
    # Fairness
    "ultimatum": {**FAIRNESS_STRATEGIES, **GENERIC_STRATEGIES},
    "dictator": {**FAIRNESS_STRATEGIES, **GENERIC_STRATEGIES},
    "third_party_punishment": {**FAIRNESS_STRATEGIES, **GENERIC_STRATEGIES},
    # Negotiation
    "alternating_offers": {**NEGOTIATION_STRATEGIES, **GENERIC_STRATEGIES},
}


def get_strategies_for_game(game_config: dict) -> dict[str, Callable]:
    """
    Get the relevant strategies for a given game.
    Returns dict of {strategy_name: strategy_fn}.
    """
    game_id = game_config["game_id"]
    category = game_config["category"]

    # Try game-specific strategies first
    if game_id in STRATEGIES_BY_GAME:
        return {**STRATEGIES_BY_GAME[game_id], **GENERIC_STRATEGIES}

    # Fall back to category strategies
    if category in STRATEGIES_BY_CATEGORY:
        return STRATEGIES_BY_CATEGORY[category]

    return GENERIC_STRATEGIES


# Full list of all unique strategies
ALL_STRATEGIES = {}
for d in [PD_STRATEGIES, BOS_STRATEGIES, STAG_STRATEGIES,
          CHICKEN_STRATEGIES, GENERIC_STRATEGIES, TRUST_STRATEGIES,
          NEGOTIATION_STRATEGIES, FAIRNESS_STRATEGIES, SIGNALING_STRATEGIES]:
    ALL_STRATEGIES.update(d)

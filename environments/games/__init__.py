"""
Game Registry
==============
Imports all game category modules and builds a unified GAME_REGISTRY.
"""

from .cooperation import COOPERATION_GAMES
from .coordination import COORDINATION_GAMES
from .fairness import FAIRNESS_GAMES
from .depth import DEPTH_GAMES
from .trust import TRUST_GAMES
from .competition import COMPETITION_GAMES
from .negotiation import NEGOTIATION_GAMES
from .risk import RISK_GAMES

# Unified registry: game_id -> game_config
GAME_REGISTRY = {}
ALL_GAMES = []

for game_list in [
    COOPERATION_GAMES,
    COORDINATION_GAMES,
    FAIRNESS_GAMES,
    DEPTH_GAMES,
    TRUST_GAMES,
    COMPETITION_GAMES,
    NEGOTIATION_GAMES,
    RISK_GAMES,
]:
    for game in game_list:
        GAME_REGISTRY[game["game_id"]] = game
        ALL_GAMES.append(game)

# Category groupings
CATEGORIES = {
    "cooperation": COOPERATION_GAMES,
    "coordination": COORDINATION_GAMES,
    "fairness": FAIRNESS_GAMES,
    "depth": DEPTH_GAMES,
    "trust": TRUST_GAMES,
    "competition": COMPETITION_GAMES,
    "negotiation": NEGOTIATION_GAMES,
    "risk": RISK_GAMES,
}

# ---------------------------------------------------------------------------
# Game sets for experiment designs
# ---------------------------------------------------------------------------

# Core 8: one canonical game per category (cross-play, main figures)
CORE_GAMES = [
    GAME_REGISTRY["pd_canonical"],        # cooperation
    GAME_REGISTRY["bos_standard"],        # coordination
    GAME_REGISTRY["ultimatum"],           # fairness
    GAME_REGISTRY["beauty_contest_23"],   # depth
    GAME_REGISTRY["trust_berg"],          # trust
    GAME_REGISTRY["auction_first_price"], # competition
    GAME_REGISTRY["nash_demand"],         # negotiation
    GAME_REGISTRY["chicken"],             # risk
]

# Expanded 24: 3 per category (factor analysis, richer profiles)
EXPANDED_GAMES = [
    # cooperation
    GAME_REGISTRY["pd_canonical"], GAME_REGISTRY["pg_med_mpcr"],
    GAME_REGISTRY["commons_dilemma"],
    # coordination
    GAME_REGISTRY["bos_standard"], GAME_REGISTRY["stag_hunt_standard"],
    GAME_REGISTRY["matching_pennies"],
    # fairness
    GAME_REGISTRY["ultimatum"], GAME_REGISTRY["dictator"],
    GAME_REGISTRY["third_party_punishment"],
    # depth
    GAME_REGISTRY["beauty_contest_23"], GAME_REGISTRY["centipede_6"],
    GAME_REGISTRY["eleven_twenty"],
    # trust
    GAME_REGISTRY["trust_berg"], GAME_REGISTRY["gift_exchange"],
    GAME_REGISTRY["repeated_trust"],
    # competition
    GAME_REGISTRY["auction_first_price"], GAME_REGISTRY["auction_vickrey"],
    GAME_REGISTRY["colonel_blotto"],
    # negotiation
    GAME_REGISTRY["nash_demand"], GAME_REGISTRY["alternating_offers"],
    GAME_REGISTRY["multi_issue"],
    # risk
    GAME_REGISTRY["chicken"], GAME_REGISTRY["signaling"],
    GAME_REGISTRY["cheap_talk"],
]

# Pilot: same as core (backward compat)
PILOT_GAMES = CORE_GAMES

# Named game sets for design system
GAME_SETS = {
    "core_8": CORE_GAMES,
    "expanded_24": EXPANDED_GAMES,
    "all": ALL_GAMES,
}

def get_game_set(name: str) -> list:
    """Get game set by name. Returns list of game config dicts."""
    if name not in GAME_SETS:
        raise ValueError(f"Unknown game set: {name}. "
                         f"Available: {list(GAME_SETS.keys())}")
    return GAME_SETS[name]

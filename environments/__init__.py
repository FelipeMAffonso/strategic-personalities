"""
Strategic Personalities Game Environments
==========================================
Exports the game engine, game registry, and all game modules.
"""

from .engine import (
    GameEngine,
    MatchResult,
    RoundResult,
    make_model_player,
    make_strategy_player,
    StrategyAdapter,
    play_model_vs_strategy,
    randomise_labels,
)
from .games import GAME_REGISTRY, ALL_GAMES, CATEGORIES, PILOT_GAMES

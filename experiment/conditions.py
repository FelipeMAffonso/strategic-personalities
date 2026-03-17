"""
Experimental Conditions
========================
Defines all experimental condition configurations and how they map
to prompt framings, game configurations, and trial parameters.
"""

from __future__ import annotations

from environments.prompts import FRAMING_PRESETS

# ---------------------------------------------------------------------------
# Condition Registry
# ---------------------------------------------------------------------------

CONDITION_REGISTRY = {
    # Core conditions (run for all games)
    "baseline": {
        "framing": "baseline",
        "temperature": 1.0,
        "num_trials": 20,
        "description": "Default neutral framing, temp=1.0 for distributional analysis",
    },

    # Cover stories
    "cover_cooking": {
        "framing": "cover_cooking",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Cooking competition cover story",
    },
    "cover_project": {
        "framing": "cover_project",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Research project collaboration cover story",
    },
    "cover_business": {
        "framing": "cover_business",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Business competition cover story",
    },

    # Goal framings
    "goal_maximise": {
        "framing": "goal_maximise",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Explicit goal: maximise your points",
    },
    "goal_win": {
        "framing": "goal_win",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Explicit goal: beat the other player",
    },
    "goal_fair": {
        "framing": "goal_fair",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Explicit goal: reach fair outcomes",
    },
    "goal_joint": {
        "framing": "goal_joint",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Explicit goal: maximise joint welfare",
    },

    # Payoff scaling
    "payoff_10x": {
        "framing": "payoff_10x",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Payoffs multiplied by 10",
    },
    "payoff_100x": {
        "framing": "payoff_100x",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Payoffs multiplied by 100",
    },

    # Reasoning modes
    "scot": {
        "framing": "scot",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Social Chain-of-Thought (predict then decide)",
    },
    "cot": {
        "framing": "cot",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Chain-of-Thought reasoning before decision",
    },

    # Opponent information (Akata "obvious" conditions)
    "obvious_cooperate": {
        "framing": "obvious_cooperate",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Told opponent always cooperates",
    },
    "obvious_defect": {
        "framing": "obvious_defect",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Told opponent always defects",
    },
    "obvious_tft": {
        "framing": "obvious_tft",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Told opponent plays TFT",
    },
    "beware_mistakes": {
        "framing": "beware_mistakes",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Warned opponent makes mistakes",
    },

    # Prompt paraphrases (stability robustness)
    "paraphrase_1": {
        "framing": "paraphrase_1",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Paraphrase variant 1 of rules preamble",
    },
    "paraphrase_2": {
        "framing": "paraphrase_2",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Paraphrase variant 2 of rules preamble",
    },
    "paraphrase_3": {
        "framing": "paraphrase_3",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "Paraphrase variant 3 of rules preamble",
    },

    # Personality system prompts (GAMABench-style)
    "personality_rational": {
        "framing": "personality_rational",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "System prompt: rational decision-maker persona",
    },
    "personality_cooperative": {
        "framing": "personality_cooperative",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "System prompt: cooperative persona",
    },
    "personality_competitive": {
        "framing": "personality_competitive",
        "temperature": 1.0,
        "num_trials": 10,
        "description": "System prompt: competitive persona",
    },

    # Temperature variants
    "temp_0": {
        "framing": "baseline",
        "temperature": 0.0,
        "num_trials": 5,
        "description": "Temperature 0 (deterministic, Akata replication)",
    },
    "temp_0.5": {
        "framing": "baseline",
        "temperature": 0.5,
        "num_trials": 10,
        "description": "Temperature 0.5 (moderate stochasticity)",
    },
    "temp_1.5": {
        "framing": "baseline",
        "temperature": 1.5,
        "num_trials": 10,
        "description": "Temperature 1.5 (high stochasticity)",
    },
}

# ---------------------------------------------------------------------------
# Condition subsets for different experiment modes
# ---------------------------------------------------------------------------

# Pilot: just baseline
PILOT_CONDITIONS = ["baseline"]

# Core: baseline + key framings (main paper figures)
CORE_CONDITIONS = [
    "baseline",
    "goal_maximise",
    "goal_win",
    "goal_fair",
    "scot",
    "temp_0",
]

# Robustness: all conditions (supplementary)
ALL_CONDITIONS = list(CONDITION_REGISTRY.keys())

# Robustness: paraphrases + personality prompts (new from audit)
ROBUSTNESS_CONDITIONS = [
    "paraphrase_1", "paraphrase_2", "paraphrase_3",
    "personality_rational", "personality_cooperative", "personality_competitive",
]

# Replication: conditions matching Akata et al. exactly
AKATA_REPLICATION_CONDITIONS = [
    "baseline",
    "obvious_cooperate",
    "obvious_defect",
    "obvious_tft",
    "beware_mistakes",
    "cover_cooking",
    "cover_project",
    "scot",
    "temp_0",
]


def get_condition(name: str) -> dict:
    """Get condition config by name."""
    if name not in CONDITION_REGISTRY:
        raise ValueError(f"Unknown condition: {name}. "
                         f"Available: {list(CONDITION_REGISTRY.keys())}")
    return CONDITION_REGISTRY[name]

"""
Prompt Templates and Framing Variants
=======================================
Implements all prompt manipulations from Akata et al. (2025) plus extensions.

Prompt structure (three-block):
  1. RULES — game description with abstract labels
  2. HISTORY — past round outcomes
  3. QUERY — current round decision prompt

Framing variants modify HOW these blocks are phrased without changing
the underlying game mechanics.
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Outcome label variants
# ---------------------------------------------------------------------------

OUTCOME_LABELS = {
    "points":  {"unit": "points",  "format": lambda x: f"{x} points"},
    "dollars": {"unit": "dollars", "format": lambda x: f"${x}"},
    "coins":   {"unit": "coins",   "format": lambda x: f"{x} coins"},
    "votes":   {"unit": "votes",   "format": lambda x: f"{x} votes"},
    "textual": {"unit": "points",  "format": lambda x: _num_to_text(x) + " points"},
}

def _num_to_text(n: float) -> str:
    """Convert number to text representation."""
    text_map = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen",
        20: "twenty", 50: "fifty", 100: "one hundred",
    }
    n_int = int(n)
    if n_int in text_map:
        return text_map[n_int]
    return str(n_int)


# ---------------------------------------------------------------------------
# Role label variants
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    "neutral":      {"self": "you", "other": "the other player"},
    "player":       {"self": "Player {id}", "other": "Player {other_id}"},
    "buyer_seller": {"self": "the Buyer", "other": "the Seller"},
    "firm":         {"self": "Firm {id}", "other": "Firm {other_id}"},
    "country":      {"self": "Country {id}", "other": "Country {other_id}"},
}


# ---------------------------------------------------------------------------
# Cover Story Framings
# ---------------------------------------------------------------------------

COVER_STORIES = {
    "neutral": {
        "preamble": "",
        "context": "You are playing a game repeatedly with another player.",
    },
    "cooking": {
        "preamble": (
            "Imagine you are participating in a cooking competition with "
            "another chef. In each round, you both decide on your approach."
        ),
        "context": (
            "You are in a cooking competition. Each round, you choose your "
            "cooking strategy."
        ),
    },
    "project": {
        "preamble": (
            "Imagine you are working on a collaborative research project "
            "with a colleague. Each week, you both decide how much effort "
            "to invest in the shared project."
        ),
        "context": (
            "You are collaborating on a research project. Each round "
            "represents a week of work."
        ),
    },
    "business": {
        "preamble": (
            "You are a business owner competing with another business in "
            "the same market. Each quarter, you both make strategic decisions."
        ),
        "context": (
            "You are managing a business. Each round represents a quarter."
        ),
    },
    "diplomacy": {
        "preamble": (
            "You are a diplomat representing your country in ongoing "
            "negotiations with another country. Each round represents a "
            "negotiation session."
        ),
        "context": (
            "You are in diplomatic negotiations. Each round is a session."
        ),
    },
}

# ---------------------------------------------------------------------------
# Goal Framings
# ---------------------------------------------------------------------------

GOAL_FRAMINGS = {
    "none": "",
    "maximise": "Your goal is to maximise the total number of points you earn across all rounds.",
    "win": "Your goal is to earn more points than the other player.",
    "fair": "Your goal is to reach fair outcomes for both players.",
    "joint": "Your goal is to maximise the combined total points for both players.",
}

# ---------------------------------------------------------------------------
# Opponent Information (Obvious condition from Akata)
# ---------------------------------------------------------------------------

OPPONENT_INFO = {
    "none": "",
    "obvious_cooperate": (
        "Note: The other player always chooses the same option. "
        "Specifically, they always choose to cooperate."
    ),
    "obvious_defect": (
        "Note: The other player always chooses the same option. "
        "Specifically, they always choose to defect."
    ),
    "obvious_tft": (
        "Note: The other player uses a simple strategy. They start by "
        "cooperating, and then in each subsequent round they copy whatever "
        "you did in the previous round."
    ),
    "beware_mistakes": (
        "Beware: The other player sometimes makes mistakes. Even if they "
        "seem to be cooperating, they might accidentally defect occasionally. "
        "This is not intentional."
    ),
}

# ---------------------------------------------------------------------------
# Prompt Paraphrases (stability robustness check)
# ---------------------------------------------------------------------------
# Each paraphrase replaces the standard 2x2 preamble ("You are playing a game
# repeatedly...") with semantically identical but syntactically different text.
# Payoff lines ("- If you choose...") are preserved unchanged.

PARAPHRASE_PREAMBLES = {
    "paraphrase_1": (
        "You and another participant are engaged in a repeated decision task. "
        "In each of {rounds} rounds, both of you independently choose between "
        "Option {opt_a} and Option {opt_b}.\n\n"
        "The outcomes depend on both choices as follows:"
    ),
    "paraphrase_2": (
        "In this multi-round interaction, two players each select an option "
        "every round. You will make your choice between Option {opt_a} and "
        "Option {opt_b} for {rounds} consecutive rounds against the same "
        "opponent.\n\n"
        "The payoff structure is:"
    ),
    "paraphrase_3": (
        "Consider a scenario where you repeatedly choose between two "
        "alternatives. Over {rounds} rounds, you and another player each "
        "independently pick either Option {opt_a} or Option {opt_b}.\n\n"
        "The results are determined as follows:"
    ),
}

# Regex to detect and extract the standard 2x2 preamble from engine output
_PREAMBLE_RE = re.compile(
    r"You are playing a game repeatedly with another player\.\n"
    r"In this game, you can choose between Option (\w+) and Option (\w+)\.\n"
    r"You will play (\d+) rounds in total with the same player\.\n"
    r"\n"
    r"The rules of the game are as follows:",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Personality System Prompts (GAMABench-style)
# ---------------------------------------------------------------------------
# Prepended to the system prompt to test whether an explicit persona
# instruction overrides or interacts with the model's default strategic type.

PERSONALITY_PROMPTS = {
    "rational": (
        "You are a rational decision-maker who carefully analyzes strategic "
        "situations to maximize your expected payoff."
    ),
    "cooperative": (
        "You are a cooperative individual who values mutual benefit and trust. "
        "You prefer outcomes that are good for everyone involved."
    ),
    "competitive": (
        "You are a competitive individual who aims to outperform others. "
        "You want to earn more than the other player whenever possible."
    ),
}


# ---------------------------------------------------------------------------
# SCoT (Social Chain-of-Thought) from Akata
# ---------------------------------------------------------------------------

SCOT_PREDICTION_TEMPLATE = (
    "Before making your choice, first predict what the other player will do.\n"
    "Q: What do you think the other player will choose, Option {opt_a} or "
    "Option {opt_b}? Please explain your reasoning briefly, then state "
    "your prediction.\n"
    "A:"
)

SCOT_RESOLUTION_TEMPLATE = (
    "You predicted that the other player will choose Option {predicted}.\n"
    "Given this prediction, what is your best response?\n"
    "Q: Which option do you choose, Option {opt_a} or Option {opt_b}? "
    "A: Option"
)

COT_TEMPLATE = (
    "Think step by step about the best choice for this round. Consider:\n"
    "1. What has the other player done in previous rounds?\n"
    "2. What pattern (if any) do you detect?\n"
    "3. What is the best response given your analysis?\n"
    "After your reasoning, state your final choice.\n"
    "Q: Which option do you choose, Option {opt_a} or Option {opt_b}? "
    "A: Option"
)


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds prompts with various framing modifications applied to the
    base three-block structure from GameEngine.
    """

    def __init__(self, framing: str = "neutral", goal: str = "none",
                 opponent_info: str = "none", scot: bool = False,
                 cot: bool = False, outcome_label: str = "points",
                 role_label: str = "neutral", cover_story: str = "neutral",
                 payoff_scale: float = 1.0, paraphrase: str = "",
                 personality: str = ""):
        self.framing = framing
        self.goal = goal
        self.opponent_info = opponent_info
        self.scot = scot
        self.cot = cot
        self.outcome_label = outcome_label
        self.role_label = role_label
        self.cover_story = cover_story
        self.payoff_scale = payoff_scale
        self.paraphrase = paraphrase
        self.personality = personality

    def modify_system_prompt(self, base_system: str) -> str:
        """Apply framing modifications to the system prompt."""
        parts = []

        # Personality prefix (GAMABench-style persona instruction)
        if self.personality and self.personality in PERSONALITY_PROMPTS:
            parts.append(PERSONALITY_PROMPTS[self.personality])

        # Cover story preamble
        story = COVER_STORIES.get(self.cover_story, COVER_STORIES["neutral"])
        if story["preamble"]:
            parts.append(story["preamble"])

        # Goal framing
        goal_text = GOAL_FRAMINGS.get(self.goal, "")
        if goal_text:
            parts.append(goal_text)

        # Base rules (with optional paraphrase of preamble)
        modified_rules = base_system
        if self.paraphrase and self.paraphrase in PARAPHRASE_PREAMBLES:
            modified_rules = self._apply_paraphrase(modified_rules)
        if self.payoff_scale != 1.0:
            modified_rules = self._scale_payoffs(modified_rules)
        parts.append(modified_rules)

        # Opponent information
        info = OPPONENT_INFO.get(self.opponent_info, "")
        if info:
            parts.append(info)

        return "\n\n".join(p for p in parts if p)

    def modify_user_message(self, base_user: str,
                            abstract_options: list[str]) -> str:
        """Apply framing modifications to the user message (history + query)."""
        msg = base_user

        if self.scot and len(abstract_options) >= 2:
            # Replace the standard query with SCoT two-step
            # The caller should handle the two-step prompting
            pass  # SCoT is handled at the runner level

        if self.cot and len(abstract_options) >= 2:
            # Prepend CoT instruction
            cot_text = COT_TEMPLATE.format(
                opt_a=abstract_options[0],
                opt_b=abstract_options[1],
            )
            # Replace the "Q:" portion
            if "Q:" in msg:
                parts = msg.split("Q:", 1)
                msg = parts[0] + cot_text
            else:
                msg = msg + "\n" + cot_text

        return msg

    def _apply_paraphrase(self, text: str) -> str:
        """Replace the standard 2x2 preamble with a paraphrase variant."""
        m = _PREAMBLE_RE.search(text)
        if not m:
            return text  # Not a 2x2 game or non-standard rules — leave unchanged
        opt_a, opt_b, rounds = m.group(1), m.group(2), m.group(3)
        template = PARAPHRASE_PREAMBLES[self.paraphrase]
        replacement = template.format(opt_a=opt_a, opt_b=opt_b, rounds=rounds)
        return text[:m.start()] + replacement + text[m.end():]

    def _scale_payoffs(self, text: str) -> str:
        """Multiply all numeric payoffs in the text by payoff_scale."""
        import re
        scale = self.payoff_scale

        def _multiply_match(m):
            val = float(m.group())
            scaled = val * scale
            if scaled == int(scaled):
                return str(int(scaled))
            return f"{scaled:.1f}"

        # Only scale numbers that appear in payoff contexts
        # "win X points" or "wins X points"
        text = re.sub(
            r'(?<=win[s]?\s)\d+(?:\.\d+)?(?=\s+points)',
            _multiply_match, text
        )
        return text

    def build_scot_prompts(self, system_prompt: str, user_message: str,
                           abstract_options: list[str]) -> tuple[str, str, str]:
        """
        Build the two-step SCoT prompts.
        Returns (prediction_system, prediction_user, resolution_template).
        """
        opt_a, opt_b = abstract_options[0], abstract_options[1]

        prediction_user = user_message
        if "Q:" in prediction_user:
            parts = prediction_user.split("Q:", 1)
            prediction_user = parts[0] + SCOT_PREDICTION_TEMPLATE.format(
                opt_a=opt_a, opt_b=opt_b
            )

        resolution_tmpl = SCOT_RESOLUTION_TEMPLATE.format(
            predicted="{predicted}",
            opt_a=opt_a,
            opt_b=opt_b,
        )

        return system_prompt, prediction_user, resolution_tmpl


# ---------------------------------------------------------------------------
# Condition presets (maps condition name to PromptBuilder kwargs)
# ---------------------------------------------------------------------------

FRAMING_PRESETS = {
    "baseline": {},
    "cover_cooking": {"cover_story": "cooking"},
    "cover_project": {"cover_story": "project"},
    "cover_business": {"cover_story": "business"},
    "cover_diplomacy": {"cover_story": "diplomacy"},
    "goal_maximise": {"goal": "maximise"},
    "goal_win": {"goal": "win"},
    "goal_fair": {"goal": "fair"},
    "goal_joint": {"goal": "joint"},
    # NOTE: payoff_10x/100x scaling is handled by GameEngine.__init__(payoff_scale=N)
    # which scales payoff_matrix, endowment, prize, pot, bid ranges, etc.
    # The runner extracts payoff_scale from this preset and passes it to the engine.
    # PromptBuilder._scale_payoffs() is DEPRECATED and should NOT be used.
    "payoff_10x": {"payoff_scale": 10.0},
    "payoff_100x": {"payoff_scale": 100.0},
    "scot": {"scot": True},
    "cot": {"cot": True},
    "obvious_cooperate": {"opponent_info": "obvious_cooperate"},
    "obvious_defect": {"opponent_info": "obvious_defect"},
    "obvious_tft": {"opponent_info": "obvious_tft"},
    "beware_mistakes": {"opponent_info": "beware_mistakes"},
    "outcome_dollars": {"outcome_label": "dollars"},
    "outcome_textual": {"outcome_label": "textual"},
    "role_player": {"role_label": "player"},
    "role_buyer_seller": {"role_label": "buyer_seller"},
    # Prompt paraphrases (stability robustness)
    "paraphrase_1": {"paraphrase": "paraphrase_1"},
    "paraphrase_2": {"paraphrase": "paraphrase_2"},
    "paraphrase_3": {"paraphrase": "paraphrase_3"},
    # Personality system prompts (GAMABench-style)
    "personality_rational": {"personality": "rational"},
    "personality_cooperative": {"personality": "cooperative"},
    "personality_competitive": {"personality": "competitive"},
}


def get_prompt_builder(condition: str = "baseline") -> PromptBuilder:
    """Create a PromptBuilder from a named condition preset."""
    kwargs = FRAMING_PRESETS.get(condition, {})
    return PromptBuilder(**kwargs)

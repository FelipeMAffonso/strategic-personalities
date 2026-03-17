"""
Game Engine for Strategic Personalities Experiments
====================================================
Core loop for playing repeated games between LLM agents and/or hand-coded
strategies.  Follows the Akata et al. (2025, Nature Human Behaviour) three-block
prompt structure: RULES + HISTORY + QUERY.

Supports four game interaction types:
  - simultaneous  : both players choose in parallel (PD, BoS, Stag Hunt, ...)
  - sequential    : players alternate moves (Centipede, Ultimatum, ...)
  - auction       : each player submits a numeric bid/valuation
  - allocation    : each player divides a fixed budget across categories

Design decisions:
  - Option labels are randomised per trial (controls for positional/label bias)
  - Temperature is configurable (default 1.0 for distributional analysis)
  - Both single-token and free-text response parsing supported
  - Full round-by-round history is recorded for downstream Hodoscope analysis
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Outcome of a single round."""
    round_num: int
    choices: dict[int, str]          # player_id -> raw choice string
    parsed_choices: dict[int, str]   # player_id -> canonical choice label
    payoffs: dict[int, float]        # player_id -> payoff for this round
    reasoning: dict[int, str] = field(default_factory=dict)  # optional reasoning traces (visible text)
    thinking: dict[int, str] = field(default_factory=dict)   # thinking/reasoning traces (internal CoT)


@dataclass
class MatchResult:
    """Full record of a multi-round match."""
    game_id: str
    game_name: str
    game_category: str               # game category (cooperation, coordination, etc.)
    players: dict[int, str]          # player_id -> identifier (model key or strategy name)
    num_rounds: int
    rounds: list[RoundResult]
    total_payoffs: dict[int, float]
    label_map: dict[str, str]        # abstract label -> canonical label (e.g. "J" -> "cooperate")
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "game_name": self.game_name,
            "game_category": self.game_category,
            "players": self.players,
            "num_rounds": self.num_rounds,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "choices": r.choices,
                    "parsed_choices": r.parsed_choices,
                    "payoffs": r.payoffs,
                    "reasoning": r.reasoning,
                    "thinking": r.thinking,
                }
                for r in self.rounds
            ],
            "total_payoffs": self.total_payoffs,
            "label_map": self.label_map,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Label randomisation
# ---------------------------------------------------------------------------

# Matches Akata et al. (2025, NHB) label pairs exactly
LABEL_PAIRS = [
    ("J", "F"),
    ("Q", "X"),
    ("R", "H"),
    ("Y", "W"),
    ("T", "N"),
    ("P", "M"),
]


def randomise_labels(canonical_options: list[str], seed: int | None = None) -> dict:
    """
    Map canonical option labels (e.g. ["cooperate", "defect"]) to abstract
    letters.  Returns {abstract_label: canonical_label}.

    For >2 options, generates unique single letters.
    """
    rng = random.Random(seed)
    n = len(canonical_options)

    if n == 2:
        pair = rng.choice(LABEL_PAIRS)
        if rng.random() < 0.5:
            pair = (pair[1], pair[0])
        return {pair[0]: canonical_options[0], pair[1]: canonical_options[1]}

    # For n > 2: pick n distinct letters
    all_letters = list("ABCDEFGHJKLMNPQRSTWXYZ")
    rng.shuffle(all_letters)
    labels = all_letters[:n]
    return {labels[i]: canonical_options[i] for i in range(n)}


# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class GameEngine:
    """
    Orchestrates repeated games between two (or N) players.

    A player is any callable with signature:
        player_fn(system_prompt: str, user_message: str) -> str

    For LLM players this wraps `call_model_with_retry`.
    For strategy players this wraps the strategy function.
    """

    def __init__(self, game_config: dict, num_rounds: int | None = None,
                 label_seed: int | None = None,
                 single_message: bool = False,
                 judge_fn: Callable | None = None,
                 payoff_scale: float = 1.0):
        """
        Args:
            game_config: Game configuration dict.
            num_rounds: Override for number of rounds (default from game config).
            label_seed: Seed for label randomization (None = random).
            single_message: If True, concatenate rules+history+query into a
                single user message (replicates Akata's prompt format exactly).
                If False (default), split into system_prompt + user_message.
            judge_fn: Optional LLM-as-judge for parsing verification.
                Signature: judge_fn(response_text, query_context, response_type)
                -> str.  response_type is "numeric" or "choice".
                Returns the extracted value as a string.
            payoff_scale: Multiply all payoff-related values by this factor.
                Used by the payoff_10x/payoff_100x conditions. Scales the
                payoff matrix, endowment, prize, pot, bid/val ranges, bonus,
                budget, and payoff_schedule. This ensures that rules text,
                payoff computation, history text, and query text all use
                consistent scaled values.

                BUG FIX (2026-02-27): Previously payoff_10x was handled by
                PromptBuilder._scale_payoffs() which only modified the rules
                text via regex but NOT the engine's internal parameters.
                This caused: (1) history text showing unscaled payoffs while
                rules showed scaled, (2) numeric game ranges not scaled,
                (3) payoff_fn computing with unscaled parameters. The fix
                moves all scaling into the engine constructor so every
                downstream component sees consistent values.
        """
        self.config = copy.deepcopy(game_config)
        self.payoff_scale = payoff_scale
        self.num_rounds = num_rounds or game_config.get("num_rounds", 10)
        self.game_id = game_config["game_id"]
        self.game_name = game_config["name"]
        self.game_type = game_config["type"]
        self.canonical_options = game_config["options"]
        self.single_message = single_message
        self.judge_fn = judge_fn

        # Apply payoff scaling to all numeric game parameters
        if payoff_scale != 1.0:
            self._apply_payoff_scale(payoff_scale)

        # Randomise labels
        self.label_map = randomise_labels(self.canonical_options, seed=label_seed)
        self.abstract_options = list(self.label_map.keys())
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self._rng = random.Random(label_seed)  # for per-round option order shuffling

    def _apply_payoff_scale(self, scale: float) -> None:
        """
        Scale all payoff-related parameters in self.config by the given factor.

        This modifies the deepcopy of game_config so that rules text generation,
        payoff computation, query text, and history display all use consistent
        scaled values. Called from __init__ when payoff_scale != 1.0.

        Scaled parameters:
        - payoff_matrix: all numeric values multiplied
        - endowment, prize, pot, budget, bonus: multiplied
        - min_bid, max_bid, min_val, max_val: multiplied
        - payoff_schedule: all values in the schedule multiplied
        - valuation_fn: wrapped to return scaled valuations
        """
        cfg = self.config

        def _s(v):
            """Scale a value, keeping it as int when the result is whole."""
            result = v * scale
            return int(result) if result == int(result) else result

        # Scale payoff matrix (2x2 and N-player simultaneous games)
        if "payoff_matrix" in cfg:
            scaled = {}
            for key, payoffs in cfg["payoff_matrix"].items():
                if isinstance(payoffs, (list, tuple)):
                    scaled[key] = tuple(_s(v) for v in payoffs)
                else:
                    scaled[key] = _s(payoffs)
            cfg["payoff_matrix"] = scaled

        # Scale scalar parameters
        for param in ("endowment", "prize", "pot", "budget", "bonus",
                      "min_bid", "max_bid", "min_val", "max_val"):
            if param in cfg:
                cfg[param] = _s(cfg[param])

        # Scale payoff schedule (Centipede game)
        if "payoff_schedule" in cfg:
            cfg["payoff_schedule"] = [
                {pid: _s(v) for pid, v in node.items()}
                for node in cfg["payoff_schedule"]
            ]

        # Wrap valuation_fn to return scaled valuations (auctions)
        if "valuation_fn" in cfg:
            original_fn = cfg["valuation_fn"]
            def _scaled_valuation_fn(round_num, _orig=original_fn, _s=scale):
                vals = _orig(round_num)
                return {pid: v * _s for pid, v in vals.items()}
            cfg["valuation_fn"] = _scaled_valuation_fn

    # ----- prompt construction -----

    def _build_rules_block(self, player_id: int, framing: str = "neutral") -> str:
        """
        Block 1: Game rules description.
        Uses abstract labels (not canonical names).
        """
        cfg = self.config
        opts = self.abstract_options

        # BUG FIX (2026-02-27): N-player simultaneous games (diners_dilemma,
        # el_farol_bar) have 2 options but use payoff_fn, not payoff_matrix.
        # Must check for payoff_matrix existence, not just option count.
        if self.game_type == "simultaneous" and len(opts) == 2 and "payoff_matrix" in cfg:
            return self._build_2x2_rules(player_id, opts, cfg, framing)
        elif self.game_type == "sequential":
            return self._build_sequential_rules(player_id, opts, cfg, framing)
        elif self.game_type == "auction":
            return self._build_auction_rules(player_id, cfg, framing)
        elif self.game_type == "allocation":
            return self._build_allocation_rules(player_id, cfg, framing)
        elif self.game_type == "simultaneous":
            return self._build_nplayer_rules(player_id, opts, cfg, framing)
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")

    def _build_2x2_rules(self, player_id: int, opts: list[str],
                         cfg: dict, framing: str) -> str:
        """Rules for a 2×2 simultaneous game (PD, BoS, Stag Hunt, Chicken, etc.)."""
        matrix = cfg["payoff_matrix"]
        opt_a, opt_b = opts[0], opts[1]
        can_a = self.label_map[opt_a]
        can_b = self.label_map[opt_b]

        # Build outcome descriptions — matches Akata prompt structure
        lines = [
            f"You are playing a game repeatedly with another player.",
            f"In this game, you can choose between Option {opt_a} and Option {opt_b}.",
            f"You will play {self.num_rounds} rounds in total with the same player.",
            f"",
            f"The rules of the game are as follows:",
        ]

        # Get payoffs from matrix (keyed by canonical names as (p0_choice, p1_choice))
        for my_choice in [can_a, can_b]:
            my_label = self.reverse_map[my_choice]
            for their_choice in [can_a, can_b]:
                their_label = self.reverse_map[their_choice]
                # Matrix key is always (player0_choice, player1_choice)
                if player_id == 0:
                    key = (my_choice, their_choice)
                else:
                    key = (their_choice, my_choice)
                my_payoff = matrix[key][player_id]
                their_payoff = matrix[key][1 - player_id]
                lines.append(
                    f"- If you choose Option {my_label} and the other player "
                    f"chooses Option {their_label}, you win {my_payoff} points "
                    f"and the other player wins {their_payoff} points."
                )

        return "\n".join(lines)

    def _build_sequential_rules(self, player_id: int, opts: list[str],
                                cfg: dict, framing: str) -> str:
        """Rules for sequential games (Ultimatum, Centipede, Trust, etc.)."""
        rules_text = cfg.get("rules_text", "")
        # Substitute abstract labels into the rules template
        for i, opt in enumerate(opts):
            canonical = self.label_map[opt]
            rules_text = rules_text.replace(f"{{option_{i}}}", f"Option {opt}")
            rules_text = rules_text.replace(f"{{{canonical}}}", f"Option {opt}")
        return rules_text

    def _build_auction_rules(self, player_id: int, cfg: dict,
                             framing: str) -> str:
        """Rules for auction games (first-price, second-price, all-pay, Blotto)."""
        return cfg.get("rules_text", "You are participating in an auction.")

    def _build_allocation_rules(self, player_id: int, cfg: dict,
                                framing: str) -> str:
        """Rules for allocation games (Dictator, Public Goods, etc.)."""
        return cfg.get("rules_text", "You are dividing resources.")

    def _build_nplayer_rules(self, player_id: int, opts: list[str],
                             cfg: dict, framing: str) -> str:
        """Rules for N-player simultaneous games (Public Goods, etc.)."""
        return cfg.get("rules_text", "You are playing a group game.")

    def _build_history_block(self, player_id: int,
                             rounds: list[RoundResult]) -> str:
        """
        Block 2: History of past rounds.
        Uses abstract labels for consistency.
        """
        if not rounds:
            return ""

        lines = []
        for r in rounds:
            my_choice = r.parsed_choices.get(player_id, "?")
            my_label = self.reverse_map.get(my_choice, my_choice)
            my_payoff = r.payoffs.get(player_id, 0)

            if self.game_type in ("simultaneous",):
                other_id = 1 - player_id
                their_choice = r.parsed_choices.get(other_id, "?")
                their_label = self.reverse_map.get(their_choice, their_choice)
                their_payoff = r.payoffs.get(other_id, 0)
                lines.append(
                    f"In round {r.round_num}, you chose Option {my_label} and "
                    f"the other player chose Option {their_label}. "
                    f"Thus, you won {my_payoff} points and the other player "
                    f"won {their_payoff} points."
                )
            elif self.game_type == "sequential":
                lines.append(
                    f"In round {r.round_num}, the outcome was: "
                    f"you received {my_payoff} points."
                )
            elif self.game_type in ("auction", "allocation"):
                lines.append(
                    f"In round {r.round_num}, you received {my_payoff} points."
                )

        return "\n".join(lines)

    def _build_query_block(self, player_id: int, round_num: int,
                           game_state: dict | None = None) -> str:
        """
        Block 3: Current-round decision prompt.
        """
        opts = self.abstract_options

        if self.game_type == "simultaneous":
            # Check for game-specific query function first (N-player games
            # like Diner's Dilemma, El Farol Bar, Focal Point need custom
            # queries that explain the game mechanics).
            query_fn = self.config.get("query_fn")
            if query_fn:
                return query_fn(player_id, round_num, game_state, opts,
                                self.label_map)
            # Default: generic 2-option or N-option query
            if len(opts) == 2:
                # Randomize option presentation order per round (Akata: qorder_indices)
                display_order = list(opts)
                self._rng.shuffle(display_order)
                return (
                    f"You are currently playing round {round_num}. "
                    f"Q: Which option do you choose, Option {display_order[0]} or "
                    f"Option {display_order[1]}? "
                    f"Wrap your final answer in <answer> tags, e.g. "
                    f"<answer>Option {display_order[0]}</answer>. A: Option"
                )
            else:
                display_order = list(opts)
                self._rng.shuffle(display_order)
                choices_str = ", ".join(f"Option {o}" for o in display_order[:-1])
                choices_str += f", or Option {display_order[-1]}"
                return (
                    f"You are currently playing round {round_num}. "
                    f"Q: Which option do you choose: {choices_str}? "
                    f"Wrap your final answer in <answer> tags. A: Option"
                )
        elif self.game_type == "sequential":
            return self._build_sequential_query(player_id, round_num, game_state)
        elif self.game_type == "auction":
            return self._build_auction_query(player_id, round_num, game_state)
        elif self.game_type == "allocation":
            return self._build_allocation_query(player_id, round_num, game_state)
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")

    def _build_sequential_query(self, player_id: int, round_num: int,
                                game_state: dict | None) -> str:
        """Query for sequential games — depends on current game state."""
        cfg = self.config
        query_fn = cfg.get("query_fn")
        if query_fn:
            return query_fn(player_id, round_num, game_state, self.abstract_options,
                            self.label_map)
        opts = self.abstract_options
        choices_str = " or ".join(f"Option {o}" for o in opts)
        return (
            f"It is round {round_num}. It is your turn to act. "
            f"Q: What do you choose: {choices_str}? A: Option"
        )

    def _build_auction_query(self, player_id: int, round_num: int,
                             game_state: dict | None) -> str:
        """Query for auction/numeric-input games."""
        cfg = self.config
        # Check for game-specific query function first (same pattern as
        # _build_sequential_query).  Games like Nash Demand, Beauty Contest,
        # Centipede, 11-20 etc. are typed "auction" only because they need
        # numeric parsing, but they are NOT auctions and must not receive the
        # default "private valuation / bid" framing.
        query_fn = cfg.get("query_fn")
        if query_fn:
            return query_fn(player_id, round_num, game_state,
                            self.abstract_options, self.label_map)
        # Default: true auction with private valuations
        valuation = 0
        if game_state and "valuations" in game_state:
            valuation = game_state["valuations"].get(player_id, 0)
        min_bid = cfg.get("min_bid", 0)
        max_bid = cfg.get("max_bid", 100)
        return (
            f"It is round {round_num}. Your private valuation for the item is "
            f"{valuation} points. Q: What is your bid ({min_bid} to {max_bid})? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>42</answer>. A:"
        )

    def _build_allocation_query(self, player_id: int, round_num: int,
                                game_state: dict | None) -> str:
        """Query for allocation games (amount to contribute/divide)."""
        cfg = self.config
        # Check for game-specific query function first (Public Goods,
        # Commons Dilemma need MPCR and capacity details).
        query_fn = cfg.get("query_fn")
        if query_fn:
            # Inject game parameters into game_state for the query_fn
            state = dict(game_state) if game_state else {}
            for k in ("mpcr", "endowment", "capacity"):
                if k in cfg and k not in state:
                    state[k] = cfg[k]
            return query_fn(player_id, round_num, state,
                            self.abstract_options, self.label_map)
        endowment = cfg.get("endowment", 10)
        return (
            f"It is round {round_num}. You have an endowment of {endowment} "
            f"points. Q: How much do you contribute to the group? "
            f"Wrap your final answer in <answer> tags, e.g. <answer>5</answer>. A:"
        )

    def build_prompt(self, player_id: int, round_num: int,
                     history: list[RoundResult],
                     framing: str = "neutral",
                     game_state: dict | None = None) -> tuple[str, str]:
        """
        Assemble the full prompt from three blocks.
        Returns (system_prompt, user_message).

        If single_message mode is active, system_prompt is empty and everything
        is concatenated into user_message (replicates Akata's exact format where
        the entire prompt is one user message).
        """
        rules = self._build_rules_block(player_id, framing)
        hist = self._build_history_block(player_id, history)
        query = self._build_query_block(player_id, round_num, game_state)

        # Apply framing modifications via PromptBuilder (cover stories, goals,
        # opponent info, CoT). SCoT is handled at the runner level.
        # payoff_scale is handled by GameEngine.__init__, not text manipulation.
        if framing not in ("neutral", "baseline"):
            from environments.prompts import PromptBuilder, FRAMING_PRESETS
            preset = FRAMING_PRESETS.get(framing, {})
            preset = {k: v for k, v in preset.items() if k != "payoff_scale"}
            if preset:
                pb = PromptBuilder(**preset)
                rules = pb.modify_system_prompt(rules)
                query = pb.modify_user_message(query, self.abstract_options)

        if self.single_message:
            # Akata-compatible: everything in one user message
            parts = [rules]
            if hist:
                parts.append(hist)
            parts.append(query)
            return "", "\n".join(parts)

        system_prompt = rules
        user_message = f"{hist}\n\n{query}" if hist else query
        return system_prompt, user_message

    # ----- response parsing -----

    def parse_choice(self, response_text: str) -> str | None:
        """
        Extract a valid option label from model response.
        Handles: single-token (" J"), free-text ("I choose Option J"),
        natural language ("I accept"), full sentences, etc.

        IMPORTANT: For multi-sentence reasoning, extracts the LAST/final
        mention of an option to capture the actual decision, not reasoning
        about what option X vs Y would do.

        Returns the canonical option label (e.g. "cooperate") or None if parse fails.
        """
        text = response_text.strip()

        # 0. Check for <answer> tags (structured output from our prompt)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Try canonical names FIRST (avoid single-letter matches inside words)
            for abstract_label, canonical in self.label_map.items():
                if re.search(rf'\b{re.escape(canonical)}\b', answer_text, re.IGNORECASE):
                    return canonical
            # Then try "Option X" pattern
            for abstract_label in self.abstract_options:
                if re.search(rf'\bOption\s+{re.escape(abstract_label)}\b',
                             answer_text, re.IGNORECASE):
                    return self.label_map[abstract_label]
            # Then exact single-letter match for very short tags
            if len(answer_text) <= 3:
                for abstract_label in self.abstract_options:
                    if abstract_label == answer_text.upper():
                        return self.label_map[abstract_label]

        # 1. Short response: exact match (Akata-compatible single-token "J"/"F")
        if len(text) <= 5:
            for abstract_label in self.abstract_options:
                if abstract_label in text.upper():
                    return self.label_map[abstract_label]

        # 2. Find the LAST "Option X" pattern (final answer, not reasoning)
        last_option_match = None
        for abstract_label in self.abstract_options:
            for m in re.finditer(rf'\bOption\s+{re.escape(abstract_label)}\b', text, re.IGNORECASE):
                if last_option_match is None or m.start() > last_option_match[1]:
                    last_option_match = (abstract_label, m.start())
        if last_option_match:
            return self.label_map[last_option_match[0]]

        # 3. Find the LAST canonical name mention (e.g., "accept", "reject",
        #    "cooperate", "defect", "swerve", "straight").
        last_canonical_match = None
        for abstract_label, canonical in self.label_map.items():
            for m in re.finditer(rf'\b{re.escape(canonical)}\b', text, re.IGNORECASE):
                if last_canonical_match is None or m.start() > last_canonical_match[1]:
                    last_canonical_match = (canonical, m.start())
        if last_canonical_match:
            return last_canonical_match[0]

        # 4. Fallback: LAST single letter that matches a valid option
        for char in reversed(text):
            if char.upper() in self.label_map:
                return self.label_map[char.upper()]

        return None

    def parse_numeric(self, response_text: str,
                      min_val: float = 0, max_val: float = 100) -> float | None:
        """
        Extract a numeric value from response (for auctions, allocations).
        Clamps to [min_val, max_val].

        Strategy (in priority order):
        0. Check for <answer> tags (structured output from our prompt)
        1. If the response is JUST a number (possibly with whitespace), use it.
        2. Look for a bolded number (** N **) which models often use for emphasis.
        3. Look for a number after "A:" at the end (our prompt format).
        4. Fallback: last standalone number in the text (avoids capturing
           round numbers like "In round 3" at the start of reasoning).
        """
        text = response_text.strip()

        # 0. Check for <answer> tags (highest priority)
        answer_match = re.search(r'<answer>\s*([\d]+\.?\d*)\s*</answer>', text, re.IGNORECASE)
        if answer_match:
            return max(min_val, min(float(answer_match.group(1)), max_val))

        # 1. Entire response is a single number
        m = re.fullmatch(r'\s*([\d]+\.?\d*)\s*', text)
        if m:
            return max(min_val, min(float(m.group(1)), max_val))

        # 2. Bolded number: **50** or **50.5**
        m = re.search(r'\*\*([\d]+\.?\d*)\*\*', text)
        if m:
            return max(min_val, min(float(m.group(1)), max_val))

        # 3. Number after "A:" near end of response
        m = re.search(r'A:\s*([\d]+\.?\d*)', text)
        if m:
            return max(min_val, min(float(m.group(1)), max_val))

        # 4. Last standalone number in the text (avoids "round 3" noise)
        all_nums = re.findall(r'(?<!\w)([\d]+\.?\d*)(?!\w)', text)
        if all_nums:
            val = float(all_nums[-1])
            return max(min_val, min(val, max_val))

        # 5. True fallback: any number at all
        m = re.search(r'[\d]+\.?\d*', text)
        if m:
            return max(min_val, min(float(m.group()), max_val))

        return None

    # ----- LLM-as-judge parsing verification -----

    def judge_parse_numeric(self, response_text: str, query_context: str,
                            min_val: float, max_val: float) -> float | None:
        """
        Use the LLM judge to extract a numeric value from the model's response.
        Called for ANY non-trivial response (>10 chars) when judge_fn is set.
        The judge is the PRIMARY parser for ambiguous responses.
        Returns the judge's extracted value, or None if judge fails.
        """
        if self.judge_fn is None:
            return None
        text = response_text.strip()

        try:
            judge_prompt = (
                f"A player was asked the following question in a game:\n"
                f"---\n{query_context[-500:]}\n---\n\n"
                f"The player responded:\n"
                f"---\n{text[:800]}\n---\n\n"
                f"What single number did the player choose as their final answer? "
                f"The valid range is {min_val} to {max_val}. "
                f"Respond with ONLY the number, nothing else."
            )
            result = self.judge_fn(judge_prompt)
            val = float(result.strip())
            return max(min_val, min(val, max_val))
        except (ValueError, TypeError, Exception):
            return None

    def judge_parse_choice(self, response_text: str, query_context: str) -> str | None:
        """
        Use the LLM judge to extract a choice from the model's response.
        The judge prompt includes the label mapping so it can correctly
        translate abstract labels (Option J, Option F) to canonical names
        (cooperate, defect).
        Returns the canonical choice label, or None if judge fails.
        """
        if self.judge_fn is None:
            return None
        text = response_text.strip()

        options_str = ", ".join(self.canonical_options)
        # Include label mapping so judge can translate abstract labels
        mapping_str = ", ".join(
            f"Option {k} = {v}" for k, v in self.label_map.items()
        )
        try:
            judge_prompt = (
                f"A player was asked the following question in a game:\n"
                f"---\n{query_context[-500:]}\n---\n\n"
                f"The player responded:\n"
                f"---\n{text[:800]}\n---\n\n"
                f"LABEL MAPPING: {mapping_str}\n"
                f"The valid choices are: {options_str}\n"
                f"What is the player's FINAL ANSWER (not their reasoning)? "
                f"Use the label mapping above to translate option letters. "
                f"Respond with ONLY the option name (one of: {options_str}), "
                f"nothing else."
            )
            result = self.judge_fn(judge_prompt)
            result = result.strip().lower()
            for opt in self.canonical_options:
                if opt.lower() == result or opt.lower() in result:
                    return opt
            return None
        except Exception:
            return None

    # ----- unified parsing pipelines -----

    def _parse_choice_pipeline(self, response_text: str,
                               query_context: str,
                               valid_options: list[str] | None = None) -> str:
        """
        Unified choice-parsing pipeline with correct priority:
        1. <answer> tags  (highest trust — structured output we requested)
        2. Trivial response (short, clean "Option X" — parse directly)
        3. LLM judge WITH label mapping (for complex/ambiguous responses)
        4. Regex parse_choice (LAST mention fallback)
        5. Random (last resort)

        valid_options: override canonical_options for per-player parsing
                       (e.g., signaling P1 uses ['accept','reject'] not
                       ['signal','no_signal']).
        """
        options_to_use = valid_options or self.canonical_options
        text = response_text.strip()

        # 1. <answer> tags — trust these unconditionally
        answer_match = re.search(
            r'<answer>\s*(.*?)\s*</answer>', text,
            re.IGNORECASE | re.DOTALL,
        )
        if answer_match:
            answer_text = answer_match.group(1).strip()

            # First check per-player options if overridden
            if valid_options:
                for opt in valid_options:
                    if re.search(rf'\b{re.escape(opt)}\b',
                                 answer_text, re.IGNORECASE):
                        return opt

            # Try canonical names FIRST (e.g. "cooperate", "defect")
            # Must come before abstract labels because single letters
            # like "P" can appear inside words like "cooPerAte".
            for abstract_label, canonical in self.label_map.items():
                if re.search(rf'\b{re.escape(canonical)}\b',
                             answer_text, re.IGNORECASE):
                    return canonical
            # Then try "Option X" pattern with abstract labels
            for abstract_label in self.abstract_options:
                if re.search(rf'\bOption\s+{re.escape(abstract_label)}\b',
                             answer_text, re.IGNORECASE):
                    return self.label_map[abstract_label]
            # Then try standalone abstract label (exact match for short tags)
            if len(answer_text) <= 3:
                for abstract_label in self.abstract_options:
                    if abstract_label == answer_text.upper():
                        return self.label_map[abstract_label]

        # 2. Trivial / short response — parse directly, skip judge overhead
        if len(text) <= 20:
            # Check per-player options first
            if valid_options:
                for opt in valid_options:
                    if re.search(rf'\b{re.escape(opt)}\b',
                                 text, re.IGNORECASE):
                        return opt
            direct = self.parse_choice(text)
            if direct is not None:
                return direct

        # 3. LLM judge with label mapping (skip for per-player options — not
        #    worth the API cost since the game config labels won't match)
        if not valid_options:
            judge_result = self.judge_parse_choice(response_text, query_context)
            if judge_result is not None:
                return judge_result

        # 4. Regex fallback (LAST mention) — search for per-player options too
        if valid_options:
            for opt in valid_options:
                if re.search(rf'\b{re.escape(opt)}\b',
                             response_text, re.IGNORECASE):
                    return opt
        regex_result = self.parse_choice(response_text)
        if regex_result is not None:
            return regex_result

        # 5. Random last resort
        return random.choice(options_to_use)

    def _parse_numeric_pipeline(self, response_text: str,
                                query_context: str,
                                min_val: float, max_val: float) -> float:
        """
        Unified numeric-parsing pipeline with correct priority:
        1. <answer> tags  (highest trust)
        2. Trivial response (just a number)
        3. LLM judge (for complex responses)
        4. Regex parse_numeric (fallback)
        5. Default 0 (last resort)
        """
        text = response_text.strip()

        # 1. <answer> tags
        answer_match = re.search(
            r'<answer>\s*([\d]+\.?\d*)\s*</answer>', text, re.IGNORECASE,
        )
        if answer_match:
            return max(min_val, min(float(answer_match.group(1)), max_val))

        # 2. Trivial: entire response is a single number
        m = re.fullmatch(r'\s*([\d]+\.?\d*)\s*', text)
        if m:
            return max(min_val, min(float(m.group(1)), max_val))

        # 3. LLM judge
        judge_val = self.judge_parse_numeric(
            response_text, query_context, min_val, max_val,
        )
        if judge_val is not None:
            return judge_val

        # 4. Regex fallback
        regex_val = self.parse_numeric(
            response_text, min_val=min_val, max_val=max_val,
        )
        if regex_val is not None:
            return regex_val

        # 5. Default
        return 0.0

    # ----- payoff computation -----

    def compute_payoffs(self, choices: dict[int, str],
                        game_state: dict | None = None) -> dict[int, float]:
        """
        Given parsed canonical choices, compute payoffs for each player.
        """
        cfg = self.config

        if "payoff_matrix" in cfg and self.game_type == "simultaneous":
            matrix = cfg["payoff_matrix"]
            key = tuple(choices[i] for i in sorted(choices.keys()))
            if key in matrix:
                payoffs_tuple = matrix[key]
                return {i: payoffs_tuple[i] for i in range(len(payoffs_tuple))}

        if "payoff_fn" in cfg:
            return cfg["payoff_fn"](choices, game_state, cfg)

        return {pid: 0 for pid in choices}

    # ----- match orchestration -----

    def play_match(
        self,
        player_fns: dict[int, Callable[[str, str], str]],
        num_rounds: int | None = None,
        framing: str = "neutral",
        record_reasoning: bool = False,
    ) -> MatchResult:
        """
        Play a full multi-round match.

        Args:
            player_fns: {player_id: callable(system_prompt, user_message) -> response_text}
            num_rounds: override for number of rounds
            framing: prompt framing variant
            record_reasoning: if True, store full response text per round

        Returns:
            MatchResult with full game record
        """
        n_rounds = num_rounds or self.num_rounds
        history: list[RoundResult] = []
        total_payoffs = {pid: 0.0 for pid in player_fns}

        for round_num in range(1, n_rounds + 1):
            game_state = self._get_round_state(round_num, history)

            if self.game_type == "simultaneous":
                round_result = self._play_simultaneous_round(
                    player_fns, round_num, history, framing, game_state,
                    record_reasoning,
                )
            elif self.game_type == "sequential":
                round_result = self._play_sequential_round(
                    player_fns, round_num, history, framing, game_state,
                    record_reasoning,
                )
            elif self.game_type == "auction":
                round_result = self._play_auction_round(
                    player_fns, round_num, history, framing, game_state,
                    record_reasoning,
                )
            elif self.game_type == "allocation":
                round_result = self._play_allocation_round(
                    player_fns, round_num, history, framing, game_state,
                    record_reasoning,
                )
            else:
                raise ValueError(f"Unknown game type: {self.game_type}")

            history.append(round_result)
            for pid, payoff in round_result.payoffs.items():
                total_payoffs[pid] += payoff

        # Build player identifiers
        player_names = {}
        for pid, fn in player_fns.items():
            player_names[pid] = getattr(fn, "__name__", str(fn))

        return MatchResult(
            game_id=self.game_id,
            game_name=self.game_name,
            game_category=self.config.get("category", ""),
            players=player_names,
            num_rounds=n_rounds,
            rounds=history,
            total_payoffs=total_payoffs,
            label_map=self.label_map,
        )

    def _get_round_state(self, round_num: int,
                         history: list[RoundResult]) -> dict:
        """
        Build game state for the current round.
        Used by sequential/auction games that need per-round setup.
        """
        cfg = self.config
        state = {"round_num": round_num}

        # Generate auction valuations
        if self.game_type == "auction" and "valuation_fn" in cfg:
            state["valuations"] = cfg["valuation_fn"](round_num)

        # Sequential game state
        if self.game_type == "sequential" and "state_fn" in cfg:
            state.update(cfg["state_fn"](round_num, history))

        return state

    @staticmethod
    def _get_thinking(fn) -> str:
        """Extract thinking/reasoning trace from a player function, if any."""
        return getattr(fn, "_last_thinking", "") or ""

    def _play_simultaneous_round(
        self, player_fns, round_num, history, framing, game_state,
        record_reasoning,
    ) -> RoundResult:
        """Both players choose independently, then payoffs resolve."""
        choices_raw = {}
        choices_parsed = {}
        reasoning = {}
        thinking = {}

        for pid, fn in player_fns.items():
            sys_prompt, user_msg = self.build_prompt(
                pid, round_num, history, framing, game_state
            )
            response = fn(sys_prompt, user_msg)
            choices_raw[pid] = response

            parsed = self._parse_choice_pipeline(response, user_msg)
            # Validate: result must be a canonical option
            if parsed not in self.canonical_options:
                for canonical in self.canonical_options:
                    if re.search(rf'\b{re.escape(canonical)}\b',
                                 response, re.IGNORECASE):
                        parsed = canonical
                        break
                else:
                    parsed = random.choice(self.canonical_options)
            choices_parsed[pid] = parsed
            if record_reasoning:
                reasoning[pid] = response
                t = self._get_thinking(fn)
                if t:
                    thinking[pid] = t

        payoffs = self.compute_payoffs(choices_parsed, game_state)

        return RoundResult(
            round_num=round_num,
            choices=choices_raw,
            parsed_choices=choices_parsed,
            payoffs=payoffs,
            reasoning=reasoning,
            thinking=thinking,
        )

    def _play_sequential_round(
        self, player_fns, round_num, history, framing, game_state,
        record_reasoning,
    ) -> RoundResult:
        """
        Players move in sequence. The game config specifies move order and
        how to update state between moves.
        """
        cfg = self.config
        move_order = cfg.get("move_order", sorted(player_fns.keys()))
        choices_raw = {}
        choices_parsed = {}
        reasoning = {}
        thinking = {}
        local_state = dict(game_state)

        # Store interim parsed choices on engine so StrategyAdapter can
        # access them for sequential moves within the same round.
        self._interim_parsed = {}
        self._current_round_num = round_num

        for pid in move_order:
            fn = player_fns[pid]
            sys_prompt, user_msg = self.build_prompt(
                pid, round_num, history, framing, local_state
            )
            response = fn(sys_prompt, user_msg)
            choices_raw[pid] = response

            # Determine whether this player's response is numeric or choice.
            # numeric_players (list of player IDs) overrides the blanket
            # numeric_response flag, allowing mixed games like Ultimatum
            # where the proposer submits a number but the responder picks
            # accept/reject.
            numeric_players = cfg.get("numeric_players")
            if numeric_players is not None:
                use_numeric = pid in numeric_players
            else:
                use_numeric = cfg.get("numeric_response", False)

            if use_numeric:
                min_v, max_v = cfg.get("min_val", 0), cfg.get("max_val", 100)
                parsed = str(self._parse_numeric_pipeline(
                    response, user_msg, min_v, max_v,
                ))
            else:
                # Per-player options: some sequential games have different
                # valid choices per role (e.g., signaling: P0 signal/no_signal,
                # P1 accept/reject). Check player_options first.
                player_options_map = cfg.get("player_options", {})
                if pid in player_options_map:
                    valid_options = player_options_map[pid]
                else:
                    valid_options = self.canonical_options

                # Pass per-player options to parse pipeline if overridden
                override = valid_options if pid in player_options_map else None
                parsed = self._parse_choice_pipeline(
                    response, user_msg, valid_options=override
                )
                # Validate: result must be a valid option for this player
                if parsed not in valid_options:
                    # Fallback: direct regex search for valid option names
                    for canonical in valid_options:
                        if re.search(rf'\b{re.escape(canonical)}\b',
                                     response, re.IGNORECASE):
                            parsed = canonical
                            break
                    else:
                        # Last resort: random valid choice
                        parsed = random.choice(valid_options)

            choices_parsed[pid] = parsed
            self._interim_parsed[pid] = parsed  # for StrategyAdapter
            if record_reasoning:
                reasoning[pid] = response
                t = self._get_thinking(fn)
                if t:
                    thinking[pid] = t

            # Update state for next mover
            if "update_state_fn" in cfg:
                local_state = cfg["update_state_fn"](local_state, pid, parsed)

        payoffs = self.compute_payoffs(choices_parsed, local_state)

        return RoundResult(
            round_num=round_num,
            choices=choices_raw,
            parsed_choices=choices_parsed,
            payoffs=payoffs,
            reasoning=reasoning,
            thinking=thinking,
        )

    def _play_auction_round(
        self, player_fns, round_num, history, framing, game_state,
        record_reasoning,
    ) -> RoundResult:
        """Each player submits a numeric bid."""
        choices_raw = {}
        choices_parsed = {}
        reasoning = {}
        thinking = {}
        min_bid = self.config.get("min_bid", 0)
        max_bid = self.config.get("max_bid", 100)

        for pid, fn in player_fns.items():
            sys_prompt, user_msg = self.build_prompt(
                pid, round_num, history, framing, game_state
            )
            response = fn(sys_prompt, user_msg)
            choices_raw[pid] = response

            bid = self._parse_numeric_pipeline(
                response, user_msg, min_bid, max_bid,
            )
            choices_parsed[pid] = str(bid)
            if record_reasoning:
                reasoning[pid] = response
                t = self._get_thinking(fn)
                if t:
                    thinking[pid] = t

        payoffs = self.compute_payoffs(choices_parsed, game_state)

        return RoundResult(
            round_num=round_num,
            choices=choices_raw,
            parsed_choices=choices_parsed,
            payoffs=payoffs,
            reasoning=reasoning,
            thinking=thinking,
        )

    def _play_allocation_round(
        self, player_fns, round_num, history, framing, game_state,
        record_reasoning,
    ) -> RoundResult:
        """Each player decides how much to contribute/allocate."""
        choices_raw = {}
        choices_parsed = {}
        reasoning = {}
        thinking = {}

        max_alloc = self.config.get("endowment", 10)
        for pid, fn in player_fns.items():
            sys_prompt, user_msg = self.build_prompt(
                pid, round_num, history, framing, game_state
            )
            response = fn(sys_prompt, user_msg)
            choices_raw[pid] = response

            amount = self._parse_numeric_pipeline(
                response, user_msg, 0, max_alloc,
            )
            choices_parsed[pid] = str(amount)
            if record_reasoning:
                reasoning[pid] = response
                t = self._get_thinking(fn)
                if t:
                    thinking[pid] = t

        payoffs = self.compute_payoffs(choices_parsed, game_state)

        return RoundResult(
            round_num=round_num,
            choices=choices_raw,
            parsed_choices=choices_parsed,
            payoffs=payoffs,
            reasoning=reasoning,
            thinking=thinking,
        )


# ---------------------------------------------------------------------------
# Helper: wrap an LLM model call as a player function
# ---------------------------------------------------------------------------

def make_model_player(model_key: str, model_cfg: dict,
                      call_fn: Callable, temperature: float = 1.0,
                      max_tokens: int = 256, delay: float = 0.0) -> Callable:
    """
    Wrap a model API call as a player function compatible with GameEngine.

    Args:
        model_key: human-readable model name
        model_cfg: dict with provider, model_id, thinking
        call_fn: the call_model_with_retry function
        temperature: sampling temperature
        max_tokens: max response length
        delay: seconds to wait between calls (rate limiting)

    Returns:
        Callable with signature (system_prompt, user_message) -> response_text
    """
    def player(system_prompt: str, user_message: str) -> str:
        if delay > 0:
            time.sleep(delay)
        result = call_fn(
            model_key=model_key,
            model_cfg=model_cfg,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Store token usage on the function for cost tracking
        if not hasattr(player, "_token_usage"):
            player._token_usage = []
        player._token_usage.append({
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "reasoning_tokens": result.get("reasoning_tokens", 0),
            "model_id": result["model_id"],
        })
        # Store thinking/reasoning trace for the engine to capture
        player._last_thinking = result.get("thinking", "")
        return result["text"]

    player.__name__ = model_key
    player._model_key = model_key
    player._model_cfg = model_cfg
    player._last_thinking = ""
    return player


def make_strategy_player(strategy_fn: Callable, name: str | None = None) -> Callable:
    """
    Wrap a hand-coded strategy as a player function.

    The strategy function has signature:
        strategy_fn(history: list[dict], player_id: int, game_config: dict) -> str

    We adapt this to the (system_prompt, user_message) -> str interface by
    parsing history from the prompt text.

    However, the cleaner approach used here is to have the engine call
    the strategy function directly when it knows the player is a strategy.
    This wrapper is for compatibility with GameEngine.play_match().
    """
    # Strategy players receive the prompt but extract the round number
    # and history from the engine's state, which is passed via closures
    # set up by the caller. This wrapper just returns the strategy's choice.

    def player(system_prompt: str, user_message: str) -> str:
        # The strategy function is called with the internal state
        # This is set up by StrategyPlayerAdapter below
        return player._pending_choice

    player.__name__ = name or getattr(strategy_fn, "__name__", "strategy")
    player._strategy_fn = strategy_fn
    player._pending_choice = ""
    return player


class StrategyAdapter:
    """
    Adapts a hand-coded strategy function to work with GameEngine.

    Instead of receiving prompts, the strategy function gets the actual
    game history and returns a canonical choice.

    For sequential games, the adapter also needs access to the current
    round's interim choices (e.g., the investor's amount in a trust game)
    so the strategy can respond to the current-round state.
    """

    def __init__(self, strategy_fn: Callable, player_id: int,
                 game_config: dict, engine: GameEngine):
        self.strategy_fn = strategy_fn
        self.player_id = player_id
        self.game_config = game_config
        self.engine = engine
        self.__name__ = getattr(strategy_fn, "__name__", "strategy")

    def __call__(self, system_prompt: str, user_message: str) -> str:
        """
        Called by GameEngine. Uses internal history tracking to call the
        strategy function with proper game state.
        """
        # Build history in the format the strategy expects.
        # For 2-player games, "their_choice" = the other player's choice.
        # For N-player games, "their_choice" = player 0's choice (the model),
        # since strategies react primarily to what the model did.
        num_players = self.game_config.get("num_players", 2)
        if num_players == 2:
            other_pid = 1 - self.player_id
        else:
            # In multi-player games, react to the model (P0)
            other_pid = 0 if self.player_id != 0 else 1

        history = []
        for r in self.engine._current_history:
            history.append({
                "round": r.round_num,
                "my_choice": r.parsed_choices.get(self.player_id),
                "their_choice": r.parsed_choices.get(other_pid),
                "my_payoff": r.payoffs.get(self.player_id, 0),
            })

        # For sequential games, include the current round's interim choices
        # so the strategy can react to the other player's move this round.
        # e.g., in Trust Game, the trustee needs to see how much the
        # investor sent THIS round.
        interim = getattr(self.engine, "_interim_parsed", {})
        if other_pid in interim:
            history.append({
                "round": getattr(self.engine, "_current_round_num", 0),
                "my_choice": None,  # haven't chosen yet
                "their_choice": interim[other_pid],
                "my_payoff": 0,
            })

        choice = self.strategy_fn(history, self.player_id, self.game_config)

        # For numeric game types (auction, allocation, sequential with
        # numeric_response), return the raw value in <answer> tags so the
        # parser can extract it correctly.
        game_type = self.engine.game_type
        is_numeric = game_type in ("auction", "allocation")
        if game_type == "sequential":
            cfg = self.engine.config
            numeric_players = cfg.get("numeric_players")
            if numeric_players is not None:
                is_numeric = self.player_id in numeric_players
            else:
                is_numeric = cfg.get("numeric_response", False)

        if is_numeric:
            return f"<answer>{choice}</answer>"

        # For choice games, map canonical choice to abstract label
        abstract = self.engine.reverse_map.get(choice, choice)
        return f"<answer>Option {abstract}</answer>"


def play_model_vs_strategy(
    engine: GameEngine,
    model_player: Callable,
    strategy_fn: Callable,
    model_player_id: int = 0,
    num_rounds: int | None = None,
    framing: str = "neutral",
    record_reasoning: bool = True,
) -> MatchResult:
    """
    Convenience function: play a model against a hand-coded strategy.
    Handles the adapter plumbing.

    For N-player games (N > 2), fills extra player slots with copies of
    the same strategy so the game runs at the intended player count.
    """
    num_players = engine.config.get("num_players", 2)

    # Create strategy adapters for all non-model player slots
    strategy_ids = [pid for pid in range(num_players) if pid != model_player_id]
    adapters = {}
    for sid in strategy_ids:
        adapters[sid] = StrategyAdapter(strategy_fn, sid, engine.config, engine)

    # Monkey-patch: store a reference to history on the engine
    # so the StrategyAdapter can read it
    engine._current_history = []

    # Custom match loop that updates _current_history
    n_rounds = num_rounds or engine.num_rounds
    history: list[RoundResult] = []
    total_payoffs = {pid: 0.0 for pid in range(num_players)}

    player_fns = {model_player_id: model_player}
    player_fns.update(adapters)

    # Dispatch to the correct round handler based on game type
    round_dispatch = {
        "simultaneous": engine._play_simultaneous_round,
        "sequential": engine._play_sequential_round,
        "auction": engine._play_auction_round,
        "allocation": engine._play_allocation_round,
    }
    round_handler = round_dispatch.get(engine.game_type,
                                       engine._play_simultaneous_round)

    for round_num in range(1, n_rounds + 1):
        engine._current_history = history  # update for strategy adapter
        game_state = engine._get_round_state(round_num, history)

        round_result = round_handler(
            player_fns, round_num, history, framing, game_state,
            record_reasoning,
        )
        history.append(round_result)
        for pid, payoff in round_result.payoffs.items():
            total_payoffs[pid] += payoff

    # Build player names
    player_names = {model_player_id: getattr(model_player, "__name__", "model")}
    for sid in strategy_ids:
        player_names[sid] = getattr(strategy_fn, "__name__", "strategy")

    return MatchResult(
        game_id=engine.game_id,
        game_name=engine.game_name,
        game_category=engine.config.get("category", ""),
        players=player_names,
        num_rounds=n_rounds,
        rounds=history,
        total_payoffs=total_payoffs,
        label_map=engine.label_map,
    )

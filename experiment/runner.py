"""
Experiment Runner
==================
Orchestrates game trials with caching, cost tracking, and parallel execution.
Adapted from spec-resistance/experiment/runner.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from config.models import ALL_MODELS, PILOT_MODELS, PRICING, compute_cost
from environments.engine import (
    GameEngine, MatchResult, StrategyAdapter, make_model_player,
    play_model_vs_strategy,
)
from environments.games import GAME_REGISTRY, ALL_GAMES, PILOT_GAMES
from environments.strategies import get_strategies_for_game
from environments.prompts import get_prompt_builder
from experiment.conditions import CONDITION_REGISTRY, get_condition
from harness.core import load_env, call_model_with_retry, API_CALL_DELAY, get_delay
from harness.cost_tracker import CostTracker, BudgetExceededError


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CSV_PATH = DATA_PROCESSED / "strategic_personalities.csv"

# Thread lock for CSV writes
_csv_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _trial_hash(game_id: str, model_key: str, opponent: str,
                condition: str, trial_num: int) -> str:
    """Deterministic hash for a trial configuration."""
    key = f"{game_id}|{model_key}|{opponent}|{condition}|{trial_num}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(game_id: str, model_key: str, opponent: str,
                condition: str, trial_num: int) -> Path:
    """Path for cached trial JSON."""
    h = _trial_hash(game_id, model_key, opponent, condition, trial_num)
    return DATA_RAW / f"{game_id}_{model_key}_vs_{opponent}_{condition}_t{trial_num}_{h}.json"


def _load_cached(path: Path) -> dict | None:
    """Load cached trial result if it exists."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(path: Path, data: dict):
    """Save trial result to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_HEADERS = [
    "game_id", "game_name", "game_category", "game_type",
    "model_key", "opponent", "matchup_type", "condition",
    "trial_num", "num_rounds",
    "player0_id", "player1_id",
    "player0_total_payoff", "player1_total_payoff",
    "cooperation_rate", "joint_cooperation_rate",
    "mean_choice_p0", "mean_choice_p1",
    "input_tokens", "output_tokens", "cost_usd",
    "cache_hit", "timestamp",
]


def _init_csv():
    """Initialise CSV with headers if it doesn't exist."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def _append_csv(row: dict):
    """Thread-safe CSV append."""
    with _csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Metric extraction from MatchResult
# ---------------------------------------------------------------------------

def _extract_metrics(result: MatchResult, game_config: dict) -> dict:
    """Extract summary metrics from a match result."""
    metrics = {}
    options = game_config["options"]

    if game_config["type"] == "simultaneous" and len(options) == 2:
        # Binary cooperation metrics (PD, Stag Hunt, Chicken, etc.)
        coop_option = options[0]  # first option is conventionally "cooperative"

        p0_choices = [r.parsed_choices.get(0) for r in result.rounds]
        p1_choices = [r.parsed_choices.get(1) for r in result.rounds]

        p0_coop = sum(1 for c in p0_choices if c == coop_option) / len(p0_choices)
        p1_coop = sum(1 for c in p1_choices if c == coop_option) / len(p1_choices)

        joint_coop = sum(
            1 for c0, c1 in zip(p0_choices, p1_choices)
            if c0 == coop_option and c1 == coop_option
        ) / len(p0_choices)

        metrics["cooperation_rate"] = round((p0_coop + p1_coop) / 2, 4)
        metrics["joint_cooperation_rate"] = round(joint_coop, 4)
        metrics["mean_choice_p0"] = coop_option if p0_coop > 0.5 else options[1]
        metrics["mean_choice_p1"] = coop_option if p1_coop > 0.5 else options[1]
    else:
        metrics["cooperation_rate"] = None
        metrics["joint_cooperation_rate"] = None
        metrics["mean_choice_p0"] = None
        metrics["mean_choice_p1"] = None

    return metrics


# ---------------------------------------------------------------------------
# LLM-as-judge for parsing verification
# ---------------------------------------------------------------------------

# Judge model: cheap model for verifying parsed responses.
# Uses Claude Haiku 4.5 (fast, precise, excellent at structured extraction).
JUDGE_MODEL_KEY = "claude-haiku-4.5"
JUDGE_MODEL_CFG = {
    "provider": "anthropic",
    "model_id": "claude-haiku-4-5-20251001",
}
_JUDGE_ENABLED = os.environ.get("ENABLE_JUDGE", "1") == "1"


def _create_judge_fn():
    """
    Create a judge callable for the engine.
    The judge extracts the correct answer from ambiguous model responses.
    Returns None if judging is disabled.
    """
    if not _JUDGE_ENABLED:
        return None

    from harness.core import call_model

    def judge_fn(prompt: str) -> str:
        """Call a cheap LLM to extract parsed value from a response."""
        try:
            result = call_model(
                JUDGE_MODEL_KEY,
                JUDGE_MODEL_CFG,
                system_prompt="You are a parsing assistant. Extract the exact answer from the player's response. Be precise and respond with ONLY the requested value.",
                user_message=prompt,
                max_tokens=32,
                temperature=0.0,
            )
            return result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            print(f"    [judge] failed: {e}")
            return ""

    return judge_fn


def _get_max_tokens(model_cfg: dict) -> int:
    """Get appropriate max_tokens for a model.

    Reference implementations:
    - Akata (2025): max_tokens=1 (single char forced choice)
    - GAMABench: max_tokens=1024
    - GameTheory: max_tokens=2000

    We use temp=1.0 (like GameTheory) which produces reasoning text,
    so we need sufficient tokens. Thinking/reasoning models need
    much more for their <think> blocks.
    """
    if model_cfg.get("thinking", False):
        return 16384  # thinking models need space for <think>...</think> + answer
    return 4096  # standard models: 2000 was too low, caused 5% truncation


# ---------------------------------------------------------------------------
# SCoT (Social Chain-of-Thought) adapter
# ---------------------------------------------------------------------------

def _wrap_scot_player(model_player_fn, abstract_options):
    """
    Wrap a model player with SCoT two-step prompting (Akata et al. 2025).

    Each call to the wrapped function makes TWO API calls through the
    underlying model_player_fn:
      1. Prediction: "What will the other player do?"
      2. Resolution: "You predicted X. What do you choose?"

    Token usage is tracked on the underlying model_player_fn (which
    appends to its _token_usage list for each API call).
    """
    import re as _re
    from environments.prompts import SCOT_PREDICTION_TEMPLATE, SCOT_RESOLUTION_TEMPLATE

    opt_a, opt_b = abstract_options[0], abstract_options[1]

    def scot_player(sys_prompt, user_msg):
        # Step 1: Prediction — replace Q: with prediction question
        pred_msg = user_msg
        if "Q:" in pred_msg:
            parts = pred_msg.split("Q:", 1)
            pred_msg = parts[0] + SCOT_PREDICTION_TEMPLATE.format(
                opt_a=opt_a, opt_b=opt_b
            )

        pred_response = model_player_fn(sys_prompt, pred_msg)

        # Capture internal thinking from prediction call (thinking models)
        pred_thinking = getattr(model_player_fn, "_last_thinking", "") or ""

        # Extract predicted option (last mention of Option X in response)
        predicted = opt_a  # default
        for m in _re.finditer(
            rf'\bOption\s+({_re.escape(opt_a)}|{_re.escape(opt_b)})\b',
            pred_response, _re.IGNORECASE
        ):
            predicted = m.group(1).upper()

        # Step 2: Resolution — given prediction, choose
        resolution = SCOT_RESOLUTION_TEMPLATE.format(
            predicted=predicted, opt_a=opt_a, opt_b=opt_b
        )
        # Include history context + prediction summary + resolution question
        if "Q:" in user_msg:
            history_part = user_msg.split("Q:", 1)[0]
            full_resolution = (
                history_part
                + f"Your prediction: {pred_response}\n\n"
                + resolution
            )
        else:
            full_resolution = f"Your prediction: {pred_response}\n\n" + resolution

        final_response = model_player_fn(sys_prompt, full_resolution)

        # Capture internal thinking from resolution call (thinking models)
        resolution_thinking = getattr(model_player_fn, "_last_thinking", "") or ""

        # Build combined trace: SCoT text + internal thinking from both calls.
        # For standard models, pred_thinking and resolution_thinking are empty.
        # For thinking models, they contain the model's internal chain-of-thought.
        trace_parts = []
        if pred_thinking:
            trace_parts.append(f"[Prediction Thinking] {pred_thinking}")
        trace_parts.append(f"[SCoT Prediction] {pred_response}")
        trace_parts.append(f"[Predicted: Option {predicted}]")
        if resolution_thinking:
            trace_parts.append(f"[Resolution Thinking] {resolution_thinking}")
        trace_parts.append(f"[SCoT Resolution] {final_response}")
        scot_player._last_thinking = "\n".join(trace_parts)

        return final_response

    scot_player._last_thinking = ""
    scot_player.__name__ = getattr(model_player_fn, "__name__", "scot_player")

    return scot_player


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_single_trial(
    game_id: str,
    model_key: str,
    model_cfg: dict,
    opponent: str,  # strategy name or model key
    condition: str,
    trial_num: int,
    cost_tracker: CostTracker | None = None,
    opponent_cfg: dict | None = None,  # if opponent is a model
    temperature: float = 1.0,
    num_rounds: int | None = None,
    record_reasoning: bool = True,
    label_seed: int | None = None,
) -> dict:
    """
    Run a single game trial. Uses caching — returns cached result if available.

    Returns dict with match result and metadata.
    """
    cache_file = _cache_path(game_id, model_key, opponent, condition, trial_num)
    cached = _load_cached(cache_file)
    if cached:
        return {**cached, "cache_hit": True}

    game_config = GAME_REGISTRY[game_id]
    cond_cfg = get_condition(condition)
    temp = cond_cfg.get("temperature", temperature)

    # Set up label randomisation seed
    if label_seed is None:
        label_seed = hash(f"{game_id}|{model_key}|{trial_num}") % (2**31)

    # Create judge for parsing verification
    judge_fn = _create_judge_fn()

    # Extract payoff_scale from framing preset (payoff_10x → 10.0, etc.)
    # BUG FIX (2026-02-27): Previously PromptBuilder._scale_payoffs() was
    # supposed to handle payoff scaling but was never connected to the engine
    # pipeline. This caused payoff_10x condition to have NO EFFECT — models
    # saw identical prompts as baseline. Now payoff_scale is passed directly
    # to GameEngine.__init__ which scales all internal parameters consistently.
    from environments.prompts import FRAMING_PRESETS
    framing_key = cond_cfg.get("framing", "baseline")
    payoff_scale = FRAMING_PRESETS.get(framing_key, {}).get("payoff_scale", 1.0)

    engine = GameEngine(game_config, num_rounds=num_rounds,
                        label_seed=label_seed, judge_fn=judge_fn,
                        payoff_scale=payoff_scale)

    # Create model player (max_tokens depends on whether model uses thinking)
    max_tok = _get_max_tokens(model_cfg)
    model_delay = get_delay(model_cfg["provider"])
    model_player = make_model_player(
        model_key, model_cfg, call_model_with_retry,
        temperature=temp, max_tokens=max_tok, delay=model_delay,
    )

    # Wrap with SCoT two-step prompting if needed
    # Token tracking stays on the original model_player (SCoT wrapper
    # delegates API calls to it, which appends to _token_usage).
    use_scot = (framing_key == "scot" and len(engine.abstract_options) >= 2)
    engine_model_player = (
        _wrap_scot_player(model_player, engine.abstract_options)
        if use_scot else model_player
    )

    # Create opponent player
    num_game_players = game_config.get("num_players", 2)
    if opponent_cfg is not None:
        # Model vs model
        opp_max_tok = _get_max_tokens(opponent_cfg)
        opp_delay = get_delay(opponent_cfg["provider"])
        opponent_player = make_model_player(
            opponent, opponent_cfg, call_model_with_retry,
            temperature=temp, max_tokens=opp_max_tok, delay=opp_delay,
        )
        engine_opp_player = (
            _wrap_scot_player(opponent_player, engine.abstract_options)
            if use_scot else opponent_player
        )
        player_fns = {0: engine_model_player, 1: engine_opp_player}
        # For N-player cross-play, fill extra slots with random strategy
        if num_game_players > 2:
            from environments.strategies import random_strategy
            for pid in range(2, num_game_players):
                adapter = StrategyAdapter(random_strategy, pid, game_config, engine)
                player_fns[pid] = adapter
            engine._current_history = []
        result = engine.play_match(player_fns, framing=cond_cfg["framing"],
                                    record_reasoning=record_reasoning)
        matchup_type = "cross_play" if model_key != opponent else "self_play"
    else:
        # Model vs strategy
        strategies = get_strategies_for_game(game_config)
        if opponent not in strategies:
            raise ValueError(f"Unknown strategy '{opponent}' for game '{game_id}'")
        strategy_fn = strategies[opponent]
        result = play_model_vs_strategy(
            engine, engine_model_player, strategy_fn,
            model_player_id=0,
            framing=cond_cfg["framing"],
            record_reasoning=record_reasoning,
        )
        matchup_type = "model_vs_strategy"

    # Extract token usage from model player
    total_input = 0
    total_output = 0
    if hasattr(model_player, "_token_usage"):
        for usage in model_player._token_usage:
            total_input += usage["input_tokens"]
            total_output += usage["output_tokens"]

    if opponent_cfg and hasattr(opponent_player, "_token_usage"):
        for usage in opponent_player._token_usage:
            total_input += usage["input_tokens"]
            total_output += usage["output_tokens"]

    # Compute cost
    cost = compute_cost(model_cfg["model_id"], total_input, total_output) or 0.0
    if opponent_cfg:
        cost2 = compute_cost(opponent_cfg["model_id"], total_input // 2, total_output // 2) or 0.0
        cost += cost2

    # Record cost
    if cost_tracker:
        cost_tracker.record_call(
            provider=model_cfg["provider"],
            model_id=model_cfg["model_id"],
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=cost,
            experiment=f"{game_id}_{condition}",
            trial_id=f"{model_key}_vs_{opponent}_t{trial_num}",
        )

    # Extract metrics
    metrics = _extract_metrics(result, game_config)

    # Build output
    output = {
        "game_id": game_id,
        "game_name": game_config["name"],
        "game_category": game_config["category"],
        "game_type": game_config["type"],
        "model_key": model_key,
        "opponent": opponent,
        "matchup_type": matchup_type,
        "condition": condition,
        "trial_num": trial_num,
        "num_rounds": result.num_rounds,
        "player0_id": result.players.get(0, model_key),
        "player1_id": result.players.get(1, opponent),
        "player0_total_payoff": result.total_payoffs.get(0, 0),
        "player1_total_payoff": result.total_payoffs.get(1, 0),
        "label_map": result.label_map,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": round(cost, 6),
        "cache_hit": False,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
        "match_detail": result.to_dict(),
    }

    # Cache result
    _save_cache(cache_file, output)

    # Append to CSV (without match_detail to keep CSV manageable)
    csv_row = {k: v for k, v in output.items()
               if k in CSV_HEADERS}
    _append_csv(csv_row)

    return output


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------

def run_experiment_parallel(
    matchups: list[dict],
    cost_tracker: CostTracker | None = None,
    max_workers: int = 4,
    progress_interval: int = 10,
) -> list[dict]:
    """
    Run a list of matchup configurations in parallel.

    Each matchup dict should have:
        - model_key, model_cfg, game_id, condition, num_trials
        - strategy_name (for model_vs_strategy) or model_key_p1, model_cfg_p1 (for cross/self play)
    """
    _init_csv()
    results = []
    completed = 0
    total = sum(m.get("num_trials", 1) for m in matchups)

    print(f"Running {total} trials across {len(matchups)} matchup configurations...")
    print(f"  Workers: {max_workers}")

    def _run_matchup_trial(matchup: dict, trial_num: int) -> dict:
        mtype = matchup.get("matchup_type", "model_vs_strategy")

        if mtype == "model_vs_strategy":
            return run_single_trial(
                game_id=matchup["game_id"],
                model_key=matchup["model_key"],
                model_cfg=matchup["model_cfg"],
                opponent=matchup["strategy_name"],
                condition=matchup["condition"],
                trial_num=trial_num,
                cost_tracker=cost_tracker,
            )
        elif mtype in ("self_play", "cross_play"):
            return run_single_trial(
                game_id=matchup["game_id"],
                model_key=matchup["model_key_p0"],
                model_cfg=matchup["model_cfg_p0"],
                opponent=matchup["model_key_p1"],
                condition=matchup["condition"],
                trial_num=trial_num,
                cost_tracker=cost_tracker,
                opponent_cfg=matchup["model_cfg_p1"],
            )
        else:
            raise ValueError(f"Unknown matchup type: {mtype}")

    # Build flat task list and shuffle for broad game coverage early.
    # Without shuffling, tasks are ordered by game (all PD first, then BoS,
    # etc.), so you don't see data for later games until hours in.
    # Shuffling ensures every game gets trials early, enabling early auditing.
    tasks = []
    for matchup in matchups:
        n_trials = matchup.get("num_trials", 1)
        for t in range(1, n_trials + 1):
            tasks.append((matchup, t))

    random.seed(42)  # reproducible shuffle
    random.shuffle(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_matchup_trial, m, t): (m, t)
            for m, t in tasks
        }

        for future in as_completed(futures):
            matchup, trial = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if completed % progress_interval == 0 or completed == total:
                    cached = sum(1 for r in results if r.get("cache_hit"))
                    print(f"  [{completed}/{total}] "
                          f"({cached} cached, {completed - cached} new)")

            except BudgetExceededError as e:
                print(f"  BUDGET EXCEEDED: {e}")
                break
            except Exception as e:
                mtype = matchup.get("matchup_type", "?")
                model = matchup.get("model_key", matchup.get("model_key_p0", "?"))
                game = matchup.get("game_id", "?")
                print(f"  ERROR [{model} / {game} / t{trial}]: {type(e).__name__}: {e}")
                completed += 1

    return results


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def run_pilot(cost_tracker: CostTracker | None = None) -> list[dict]:
    """
    Run pilot experiment: PILOT_MODELS × PILOT_GAMES × baseline × 1 trial.
    Quick test to verify everything works.
    """
    from experiment.matchups import generate_strategy_matchups, generate_self_play_matchups

    load_env()
    _init_csv()

    if cost_tracker is None:
        cost_tracker = CostTracker(budget_per_provider=5.00, max_calls_per_provider=200)

    print("=" * 60)
    print("PILOT RUN")
    print("=" * 60)

    # Strategy matchups
    strat_matchups = generate_strategy_matchups(
        model_set=PILOT_MODELS,
        game_set=PILOT_GAMES,
        condition="baseline",
        num_trials=1,
    )

    # Self-play matchups
    self_matchups = generate_self_play_matchups(
        model_set=PILOT_MODELS,
        game_set=PILOT_GAMES,
        condition="baseline",
        num_trials=1,
    )

    all_matchups = strat_matchups + self_matchups
    print(f"  Strategy matchups: {len(strat_matchups)}")
    print(f"  Self-play matchups: {len(self_matchups)}")
    print(f"  Total matchup configs: {len(all_matchups)}")

    results = run_experiment_parallel(
        all_matchups, cost_tracker=cost_tracker, max_workers=3,
    )

    cost_tracker.print_summary()
    cost_tracker.save_log("pilot_costs.json")

    return results


def run_full(
    conditions: list[str] | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """
    Run full experiment: ALL_MODELS × ALL_GAMES × specified conditions.
    """
    from experiment.matchups import generate_strategy_matchups, generate_self_play_matchups
    from experiment.conditions import CORE_CONDITIONS

    load_env()
    _init_csv()

    if conditions is None:
        conditions = CORE_CONDITIONS
    if cost_tracker is None:
        cost_tracker = CostTracker(budget_per_provider=100.00, max_calls_per_provider=50000)

    print("=" * 60)
    print("FULL RUN")
    print("=" * 60)

    all_matchups = []
    for condition in conditions:
        cond_cfg = get_condition(condition)
        n_trials = cond_cfg.get("num_trials", 20)

        strat = generate_strategy_matchups(
            model_set=ALL_MODELS, game_set=ALL_GAMES,
            condition=condition, num_trials=n_trials,
        )
        self_play = generate_self_play_matchups(
            model_set=ALL_MODELS, game_set=ALL_GAMES,
            condition=condition, num_trials=n_trials,
        )
        all_matchups.extend(strat)
        all_matchups.extend(self_play)

    print(f"  Total matchup configs: {len(all_matchups)}")
    print(f"  Conditions: {conditions}")

    results = run_experiment_parallel(
        all_matchups, cost_tracker=cost_tracker, max_workers=6,
    )

    cost_tracker.print_summary()
    cost_tracker.save_log("full_costs.json")

    return results


def run_single_model(
    model_key: str,
    conditions: list[str] | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """Run all games for a single model."""
    from experiment.matchups import generate_strategy_matchups, generate_self_play_matchups

    load_env()
    _init_csv()

    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_key}. "
                         f"Available: {list(ALL_MODELS.keys())}")

    model_set = {model_key: ALL_MODELS[model_key]}
    if conditions is None:
        conditions = ["baseline"]
    if cost_tracker is None:
        cost_tracker = CostTracker(budget_per_provider=20.00, max_calls_per_provider=5000)

    print(f"Running {model_key} across all games...")

    all_matchups = []
    for condition in conditions:
        cond_cfg = get_condition(condition)
        n_trials = cond_cfg.get("num_trials", 20)

        strat = generate_strategy_matchups(
            model_set=model_set, game_set=ALL_GAMES,
            condition=condition, num_trials=n_trials,
        )
        self_play = generate_self_play_matchups(
            model_set=model_set, game_set=ALL_GAMES,
            condition=condition, num_trials=n_trials,
        )
        all_matchups.extend(strat)
        all_matchups.extend(self_play)

    results = run_experiment_parallel(
        all_matchups, cost_tracker=cost_tracker, max_workers=4,
    )

    cost_tracker.print_summary()
    return results

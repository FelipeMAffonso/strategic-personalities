#!/usr/bin/env python3
"""
Strategic Personalities of Frontier AI Models
================================================
CLI entry point for running experiments, analysis, and figure generation.

Usage:
    # --- Experiment designs (new system) ---
    python run.py --list-designs                          # Show all presets + costs
    python run.py --design smoke --dry-run                # Cost estimate for smoke test
    python run.py --design C --dry-run                    # Cost estimate for "Maximum"
    python run.py --design pilot                          # Run pilot (5 models, all games, 2 trials)
    python run.py --design A                              # Run "Surgical" (17 models, 8 games, 20 trials)
    python run.py --design C --trials 5                   # Override trial count
    python run.py --design custom --models core_17 --games all --trials 10

    # --- Legacy modes (still work) ---
    python run.py --mode pilot                            # Original 10-model pilot
    python run.py --mode suite --model claude-haiku-4.5   # Single model

    # --- Analysis ---
    python run.py --analyze-only            # Run analysis on existing data
    python run.py --figures-only            # Generate figures from existing data
    python run.py --list-games              # List all registered games
    python run.py --list-models             # List all registered models
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Design system commands
# ---------------------------------------------------------------------------

def cmd_list_designs(args):
    """Show all preset designs with cost estimates."""
    from experiment.designs import list_designs
    list_designs()


def cmd_design_dry_run(args):
    """Show cost estimate for a specific design (with optional overrides)."""
    from experiment.designs import get_design, print_design, DESIGNS

    design = _resolve_design(args)
    print_design(design)

    # Also show all presets for comparison
    print("\n" + "-" * 60)
    print("  All presets for comparison:")
    for key, d in DESIGNS.items():
        from experiment.designs import estimate_cost
        est = estimate_cost(d)
        marker = " <--" if key == getattr(args, "design", "") else ""
        print(f"    [{key:>5s}] {est['total_trials']:>10,} trials  "
              f"${est['total_cost']:>8,.0f}  "
              f"{d.name}{marker}")
    print()


def cmd_design_run(args):
    """Run experiment using the design system."""
    from experiment.designs import get_design, print_design, design_to_matchups
    from harness.core import load_env
    load_env()

    design = _resolve_design(args)
    print_design(design)

    matchups = design_to_matchups(design)
    print(f"\nGenerated {len(matchups):,} matchup configurations.")

    from experiment.runner import run_experiment_parallel
    from harness.cost_tracker import CostTracker

    tracker = CostTracker(budget_per_provider=600.00, max_calls_per_provider=50000)
    results = run_experiment_parallel(matchups, cost_tracker=tracker, max_workers=80)
    tracker.print_summary()
    print(f"\nExperiment complete: {len(results)} trials")


def _resolve_design(args):
    """Build an ExperimentDesign from CLI args (preset + overrides)."""
    from experiment.designs import get_design, ExperimentDesign

    design_name = args.design

    if design_name == "custom":
        # Build from scratch using CLI flags
        overrides = {}
        if args.models:
            overrides["model_set"] = args.models
        if args.games:
            overrides["game_set"] = args.games
        if args.trials is not None:
            overrides["strategy_trials"] = args.trials
            overrides["self_play_trials"] = args.trials
            overrides["cross_play_trials"] = max(1, args.trials // 2)
        if args.no_cross_play:
            overrides["include_cross_play"] = False
        if args.no_strategy:
            overrides["include_strategy"] = False
        if args.no_self_play:
            overrides["include_self_play"] = False

        # Start from a sensible base
        design = get_design("A", **overrides)
        design.name = "Custom"
        design.description = "Custom design from CLI flags"
        return design

    # Start from preset, apply overrides
    overrides = {}
    if args.models:
        overrides["model_set"] = args.models
    if args.games:
        overrides["game_set"] = args.games
    if args.trials is not None:
        overrides["strategy_trials"] = args.trials
        overrides["self_play_trials"] = args.trials
        overrides["cross_play_trials"] = max(1, args.trials // 2)
    if args.no_cross_play:
        overrides["include_cross_play"] = False
    if args.no_strategy:
        overrides["include_strategy"] = False
    if args.no_self_play:
        overrides["include_self_play"] = False

    return get_design(design_name, **overrides)


# ---------------------------------------------------------------------------
# Legacy commands (unchanged)
# ---------------------------------------------------------------------------

def cmd_list_games(args):
    """List all registered games."""
    from environments.games import ALL_GAMES, CATEGORIES, GAME_SETS

    print(f"{'=' * 60}")
    print(f"REGISTERED GAMES ({len(ALL_GAMES)} total)")
    print(f"{'=' * 60}")

    for category, games in CATEGORIES.items():
        print(f"\n  {category.upper()} ({len(games)} games):")
        for g in games:
            print(f"    {g['game_id']:30s}  {g['name']}")
            print(f"      Type: {g['type']}, Rounds: {g.get('num_rounds', '?')}, "
                  f"Players: {g.get('num_players', 2)}")

    print(f"\n{'=' * 60}")
    print("GAME SETS:")
    for name, gs in GAME_SETS.items():
        ids = [g["game_id"] for g in gs]
        print(f"  {name:15s} ({len(gs)} games): {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")


def cmd_list_models(args):
    """List all registered models."""
    from config.models import ALL_MODELS, PILOT_MODELS, PRICING, MODEL_SETS

    print(f"{'=' * 60}")
    print(f"REGISTERED MODELS ({len(ALL_MODELS)} total)")
    print(f"{'=' * 60}")

    providers = {}
    for key, cfg in ALL_MODELS.items():
        prov = cfg["provider"]
        if prov not in providers:
            providers[prov] = []
        providers[prov].append((key, cfg))

    for provider, models in sorted(providers.items()):
        print(f"\n  {provider.upper()} ({len(models)} models):")
        for key, cfg in models:
            model_id = cfg["model_id"]
            thinking = " [thinking]" if cfg.get("thinking") else ""
            pricing = PRICING.get(model_id, (0, 0))
            print(f"    {key:35s}  ${pricing[0]:.2f}/${pricing[1]:.2f} per M tok{thinking}")

    print(f"\n{'=' * 60}")
    print("MODEL SETS:")
    for name, ms in MODEL_SETS.items():
        keys = list(ms.keys())
        print(f"  {name:15s} ({len(ms)} models): {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")


def cmd_legacy_run(args):
    """Run experiment in legacy mode."""
    from harness.core import load_env
    load_env()

    mode = args.mode

    if mode == "pilot":
        from experiment.runner import run_pilot
        results = run_pilot()
        print(f"\nPilot complete: {len(results)} trials")

    elif mode == "full":
        from experiment.runner import run_full
        results = run_full()
        print(f"\nFull run complete: {len(results)} trials")

    elif mode == "suite":
        from experiment.runner import run_single_model
        if not args.model:
            print("ERROR: --model required for suite mode")
            sys.exit(1)
        results = run_single_model(args.model)
        print(f"\nSuite complete for {args.model}: {len(results)} trials")

    elif mode == "crossplay":
        from experiment.runner import run_experiment_parallel
        from experiment.matchups import generate_cross_play_matchups
        from harness.cost_tracker import CostTracker

        matchups = generate_cross_play_matchups(num_trials=10)
        tracker = CostTracker(budget_per_provider=50.00, max_calls_per_provider=20000)
        results = run_experiment_parallel(matchups, cost_tracker=tracker, max_workers=6)
        tracker.print_summary()
        print(f"\nCross-play complete: {len(results)} trials")

    else:
        print(f"ERROR: Unknown mode '{mode}'")
        sys.exit(1)


def cmd_analyze(args):
    """Run analysis pipeline on existing data."""
    print("Running analysis pipeline...")
    from analysis.compute_behavioral_profiles import compute_profiles
    from analysis.compute_all_stats import compute_stats

    profiles = compute_profiles()
    stats = compute_stats(profiles)

    print(f"Analysis complete. Stats saved to data/processed/manuscript_numbers.json")


def cmd_figures(args):
    """Generate publication figures."""
    print("Generating figures...")
    from analysis.generate_figures import generate_all_figures
    generate_all_figures()
    print("Figures saved to results/figures/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategic Personalities of Frontier AI Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Design system (new)
    parser.add_argument("--design", type=str, default=None,
                       help="Experiment design preset (smoke, pilot, A, B, C, D, E, custom)")
    parser.add_argument("--models", type=str, default=None,
                       help="Model set override (smoke_3, pilot_5, core_17, generational, all)")
    parser.add_argument("--games", type=str, default=None,
                       help="Game set override (core_8, expanded_24, all)")
    parser.add_argument("--trials", type=int, default=None,
                       help="Override trial count for all matchup types")
    parser.add_argument("--no-cross-play", action="store_true",
                       help="Disable cross-play matchups")
    parser.add_argument("--no-strategy", action="store_true",
                       help="Disable strategy matchups")
    parser.add_argument("--no-self-play", action="store_true",
                       help="Disable self-play matchups")

    # Legacy mode (still works)
    parser.add_argument("--mode", type=str, default=None,
                       choices=["pilot", "full", "suite", "crossplay"],
                       help="Legacy experiment mode")
    parser.add_argument("--model", type=str, default=None,
                       help="Model key for legacy suite mode")

    # Standalone flags
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimate without running")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Run analysis pipeline only")
    parser.add_argument("--figures-only", action="store_true",
                       help="Generate figures only")
    parser.add_argument("--list-games", action="store_true",
                       help="List all registered games and game sets")
    parser.add_argument("--list-models", action="store_true",
                       help="List all registered models and model sets")
    parser.add_argument("--list-designs", action="store_true",
                       help="List all experiment design presets")

    args = parser.parse_args()

    # Dispatch
    if args.list_designs:
        cmd_list_designs(args)
    elif args.list_games:
        cmd_list_games(args)
    elif args.list_models:
        cmd_list_models(args)
    elif args.design and args.dry_run:
        cmd_design_dry_run(args)
    elif args.design:
        cmd_design_run(args)
    elif args.dry_run:
        # Legacy dry-run: show all designs
        cmd_list_designs(args)
    elif args.analyze_only:
        cmd_analyze(args)
    elif args.figures_only:
        cmd_figures(args)
    elif args.mode:
        cmd_legacy_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

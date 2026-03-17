#!/usr/bin/env python3
"""
Reproduction Script for "Strategic Personalities of Frontier AI Models"
=========================================================================
This script reproduces all analyses and figures from the paper.
Run from the osf/ directory.

Requirements:
    pip install -r requirements.txt

Data:
    - data/strategic_personalities.csv (authoritative dataset)
    - data/manuscript_numbers.json (all reported statistics)

Usage:
    python reproduce.py --figures     # Regenerate all figures
    python reproduce.py --stats       # Recompute all statistics
    python reproduce.py --all         # Both
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def reproduce_figures():
    """Regenerate all publication figures from processed data."""
    from analysis.generate_figures import generate_all_figures
    generate_all_figures()


def reproduce_stats():
    """Recompute all manuscript statistics from raw data."""
    from analysis.compute_behavioral_profiles import compute_profiles
    from analysis.compute_all_stats import compute_stats

    print("Step 1: Computing behavioral profiles...")
    profiles = compute_profiles()

    print("\nStep 2: Computing statistics...")
    stats = compute_stats(profiles)

    print("\nStep 3: Factor analysis...")
    from analysis.factor_analysis import run_full_analysis
    run_full_analysis(profiles)

    print("\nReproduction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce analyses for Strategic Personalities paper"
    )
    parser.add_argument("--figures", action="store_true",
                        help="Regenerate figures")
    parser.add_argument("--stats", action="store_true",
                        help="Recompute statistics")
    parser.add_argument("--all", action="store_true",
                        help="Run everything")

    args = parser.parse_args()

    if args.all or (not args.figures and not args.stats):
        reproduce_stats()
        reproduce_figures()
    else:
        if args.stats:
            reproduce_stats()
        if args.figures:
            reproduce_figures()


if __name__ == "__main__":
    main()

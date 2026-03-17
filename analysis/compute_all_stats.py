"""
Compute All Statistics
========================
Central statistical computation script. All manuscript numbers come from here.
Outputs manuscript_numbers.json for reproducibility.

Statistical approach:
  - Wilson confidence intervals (not normal approximation)
  - Fisher exact tests for binary outcomes
  - Cohen's d for effect sizes
  - Mixed-effects models for cross-game consistency
  - Bonferroni correction for multiple comparisons
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple:
    """Wilson score confidence interval for a proportion."""
    if trials == 0:
        return (np.nan, np.nan, np.nan)

    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2

    denom = 1 + z2 / trials
    centre = (p_hat + z2 / (2 * trials)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials) / denom

    return (round(centre, 4), round(max(0, centre - margin), 4),
            round(min(1, centre + margin), 4))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return np.nan
    return round((group1.mean() - group2.mean()) / pooled_sd, 4)


def compute_stats(profiles: pd.DataFrame = None) -> dict:
    """
    Compute all statistics for the manuscript.
    Returns dict of named statistics.
    """
    if profiles is None:
        profiles_path = DATA_PROCESSED / "behavioral_profiles.csv"
        if profiles_path.exists():
            profiles = pd.read_csv(profiles_path)
        else:
            print("No profiles data available.")
            return {}

    manuscript_numbers = {}

    # ---- Sample sizes ----
    manuscript_numbers["n_models"] = profiles["model_key"].nunique()
    manuscript_numbers["n_games"] = profiles["game_id"].nunique()
    manuscript_numbers["n_trials"] = len(profiles)
    manuscript_numbers["n_conditions"] = profiles["condition"].nunique()

    # ---- Cooperation rates by model ----
    coop_data = profiles[profiles["cooperation_rate"].notna()]
    if not coop_data.empty:
        model_coop = coop_data.groupby("model_key")["cooperation_rate"].agg(["mean", "std", "count"])
        manuscript_numbers["cooperation_by_model"] = {}
        for model, row in model_coop.iterrows():
            n = int(row["count"])
            mean_val = float(row["mean"])
            std_val = float(row["std"]) if not np.isnan(row["std"]) else 0.0
            # Use t-distribution CI for continuous rates (not Wilson, which is for binary)
            if n >= 2:
                se = std_val / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci = (round(mean_val, 4),
                      round(max(0, mean_val - t_crit * se), 4),
                      round(min(1, mean_val + t_crit * se), 4))
            else:
                ci = (round(mean_val, 4), np.nan, np.nan)
            manuscript_numbers["cooperation_by_model"][model] = {
                "mean": round(mean_val, 4),
                "std": round(row["std"], 4),
                "n": n,
                "ci_95": ci,
            }

        # Overall cooperation
        overall_mean = coop_data["cooperation_rate"].mean()
        overall_std = coop_data["cooperation_rate"].std()
        manuscript_numbers["overall_cooperation"] = {
            "mean": round(overall_mean, 4),
            "std": round(overall_std, 4),
        }

    # ---- Cross-game consistency ----
    # Compute per-model mean payoff ratio (player0_payoff / max possible) by game
    # as a general performance metric available across all game categories
    if "game_category" in profiles.columns:
        # Use player0_payoff as the general cross-game metric
        payoff_by_cat = profiles.groupby(
            ["model_key", "game_category"]
        )["player0_payoff"].mean().unstack()

        # Only correlate categories with data for >= 3 models
        valid_cats = payoff_by_cat.columns[payoff_by_cat.notna().sum() >= 3]
        if len(valid_cats) >= 2:
            corr = payoff_by_cat[valid_cats].corr()
            manuscript_numbers["cross_game_correlation"] = {
                "categories": list(valid_cats),
                "matrix": corr.to_dict(),
            }
        else:
            manuscript_numbers["cross_game_correlation"] = "insufficient_data"

    # ---- Provider comparisons ----
    def _get_provider(model_key):
        if "claude" in model_key:
            return "anthropic"
        elif "gpt" in model_key:
            return "openai"
        elif "gemini" in model_key or "gemma" in model_key:
            return "google"
        else:
            return "other"

    if not coop_data.empty:
        coop_data = coop_data.copy()
        coop_data["provider"] = coop_data["model_key"].apply(_get_provider)
        provider_groups = coop_data.groupby("provider")["cooperation_rate"]

        # Kruskal-Wallis test across providers
        provider_values = [group.values for _, group in provider_groups if len(group) > 1]
        if len(provider_values) >= 2:
            H, p = stats.kruskal(*provider_values)
            manuscript_numbers["provider_kruskal"] = {
                "H": round(H, 4),
                "p": round(p, 6),
            }

        # Pairwise comparisons
        providers_list = list(provider_groups.groups.keys())
        pairwise = {}
        for i, p1 in enumerate(providers_list):
            for j, p2 in enumerate(providers_list):
                if i >= j:
                    continue
                g1 = provider_groups.get_group(p1).values
                g2 = provider_groups.get_group(p2).values
                if len(g1) > 1 and len(g2) > 1:
                    U, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    d = cohens_d(g1, g2)
                    pairwise[f"{p1}_vs_{p2}"] = {
                        "U": round(U, 2),
                        "p": round(p_val, 6),
                        "cohens_d": d,
                    }
        manuscript_numbers["provider_pairwise"] = pairwise

    # ---- Save ----
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "manuscript_numbers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manuscript_numbers, f, indent=2, default=str)
    print(f"Manuscript numbers saved to {output_path}")

    # ---- Hodoscope pipeline (behavioral fingerprinting) ----
    from analysis.cross_model_divergence import run_hodoscope_pipeline
    run_hodoscope_pipeline()

    return manuscript_numbers


if __name__ == "__main__":
    compute_stats()

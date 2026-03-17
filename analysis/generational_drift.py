"""
Generational Drift Analysis
==============================
Track how behavioral dimensions change across model generations
within each provider family.

Key comparisons:
  - Anthropic: Claude 3 Haiku → 3.5 Haiku → 4.5 Haiku → 4.5 Sonnet → 4.5 Opus
  - OpenAI: GPT-4o → GPT-4.1 → GPT-5 → GPT-5.1 → GPT-5.2
  - Google: Gemini 2.0 Flash → 2.5 Flash → 2.5 Pro → 3 Flash → 3 Pro
  - Thinking vs non-thinking: same model with extended thinking on/off
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Model generation families
# ---------------------------------------------------------------------------

GENERATION_FAMILIES = {
    "anthropic_haiku": [
        "claude-haiku-3.0",
        "claude-haiku-3.5",
        "claude-haiku-4.5",
    ],
    "anthropic_capability": [
        "claude-haiku-4.5",
        "claude-sonnet-4.5",
        "claude-opus-4.5",
    ],
    "anthropic_generation": [
        "claude-sonnet-4.5",
        "claude-sonnet-4.6",
    ],
    "openai_flagship": [
        "gpt-4o",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
    ],
    "openai_small": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-5-mini",
    ],
    "openai_capability": [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "google_flash": [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-3-flash",
    ],
    "google_pro": [
        "gemini-2.5-pro",
        "gemini-3-pro",
        "gemini-3.1-pro",
    ],
    "thinking_pairs": {
        "claude-sonnet-4.5": "claude-sonnet-4.5-thinking",
        "gemini-2.5-flash": "gemini-2.5-flash-thinking",
        "gemini-2.5-pro": "gemini-2.5-pro-thinking",
        "gpt-5.2": "gpt-5.2-thinking",
    },
}


def compute_generational_drift(scores: pd.DataFrame) -> dict:
    """
    Compute behavioral drift across model generations.

    For each generation family:
      - Euclidean distance between consecutive generations
      - Direction of drift in behavioral space
      - Statistical significance (paired t-test on dimensions)

    Returns dict with per-family results.
    """
    results = {}

    for family_name, models in GENERATION_FAMILIES.items():
        if family_name == "thinking_pairs":
            continue  # handled separately

        if isinstance(models, dict):
            continue

        available = [m for m in models if m in scores.index]
        if len(available) < 2:
            continue

        family_scores = scores.loc[available]

        # Pairwise distances between consecutive generations
        distances = []
        for i in range(len(available) - 1):
            m1 = available[i]
            m2 = available[i + 1]
            dist = np.linalg.norm(family_scores.loc[m2] - family_scores.loc[m1])
            distances.append({
                "from": m1,
                "to": m2,
                "distance": round(dist, 4),
                "drift_vector": (family_scores.loc[m2] - family_scores.loc[m1]).to_dict(),
            })

        # Total drift (first to last)
        total_dist = np.linalg.norm(
            family_scores.iloc[-1] - family_scores.iloc[0]
        )

        results[family_name] = {
            "models": available,
            "pairwise_distances": distances,
            "total_drift": round(total_dist, 4),
            "n_generations": len(available),
        }

    return results


def compute_thinking_effect(scores: pd.DataFrame) -> dict:
    """
    Compare thinking vs non-thinking versions of the same model.
    """
    thinking_pairs = GENERATION_FAMILIES.get("thinking_pairs", {})
    results = {}

    for base, thinking in thinking_pairs.items():
        if base in scores.index and thinking in scores.index:
            base_scores = scores.loc[base]
            thinking_scores = scores.loc[thinking]

            diff = thinking_scores - base_scores
            distance = np.linalg.norm(diff)

            results[base] = {
                "base_model": base,
                "thinking_model": thinking,
                "distance": round(distance, 4),
                "dimension_diffs": diff.to_dict(),
                "most_affected_dimension": diff.abs().idxmax(),
            }

    return results


def compute_capability_scaling(profiles: pd.DataFrame) -> dict:
    """
    Analyse how behavioral metrics scale with model capability.
    Uses cooperation rate as the primary metric.
    """
    # Define capability ordering within each provider
    capability_order = {
        "claude-haiku-4.5": 1,
        "claude-sonnet-4.5": 2,
        "claude-opus-4.5": 3,
        "gpt-4.1-nano": 1,
        "gpt-4.1-mini": 2,
        "gpt-4.1": 3,
        "gpt-4o": 3,
        "gpt-5-mini": 2,
        "gpt-5": 3,
        "gpt-5.2": 4,
        "gemini-2.5-flash-lite": 1,
        "gemini-2.5-flash": 2,
        "gemini-2.5-pro": 3,
    }

    # Compute correlation between capability and cooperation
    coop_data = profiles[profiles["cooperation_rate"].notna()]
    if coop_data.empty:
        return {}

    model_coop = coop_data.groupby("model_key")["cooperation_rate"].mean()

    x_vals = []
    y_vals = []
    for model, coop in model_coop.items():
        if model in capability_order:
            x_vals.append(capability_order[model])
            y_vals.append(coop)

    if len(x_vals) < 3:
        return {}

    r, p = stats.pearsonr(x_vals, y_vals)

    return {
        "capability_cooperation_r": round(r, 4),
        "capability_cooperation_p": round(p, 4),
        "n_models": len(x_vals),
    }


if __name__ == "__main__":
    scores_path = DATA_PROCESSED / "factor_analysis" / "pca_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path, index_col=0)
        drift = compute_generational_drift(scores)

        for family, data in drift.items():
            print(f"\n{family}:")
            print(f"  Models: {data['models']}")
            print(f"  Total drift: {data['total_drift']}")
            for d in data["pairwise_distances"]:
                print(f"  {d['from']} -> {d['to']}: {d['distance']}")

        thinking = compute_thinking_effect(scores)
        for model, data in thinking.items():
            print(f"\nThinking effect for {model}: distance = {data['distance']}")
    else:
        print(f"No scores found at {scores_path}")

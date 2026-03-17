"""
Factor Analysis — Behavioral Dimensions
==========================================
Extract latent behavioral dimensions from cross-game metrics.

Steps:
  1. Standardise per-game metrics across models
  2. Exploratory factor analysis (EFA) to discover dimensions
  3. Confirmatory factor analysis (CFA) on held-out game variants
  4. Internal consistency (Cronbach's alpha)
  5. Test-retest reliability across prompt paraphrases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def build_feature_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model × metric feature matrix from behavioral profiles.
    Each model gets one row with averaged metrics across trials.
    """
    # Aggregate across trials: mean per (model, game, condition)
    numeric_cols = profiles.select_dtypes(include=[np.number]).columns
    group_cols = ["model_key", "game_id", "game_category", "condition"]
    existing_groups = [c for c in group_cols if c in profiles.columns]

    agg = profiles.groupby(existing_groups)[numeric_cols].mean().reset_index()

    # Pivot to wide format: one column per (game_id, metric)
    metric_cols = [c for c in numeric_cols
                   if c not in ["trial_num", "num_rounds", "input_tokens",
                                "output_tokens", "cost_usd",
                                "player0_payoff", "player1_payoff"]]

    # Build feature matrix
    rows = []
    for model_key, model_data in agg.groupby("model_key"):
        row = {"model_key": model_key}
        for _, trial_row in model_data.iterrows():
            game_id = trial_row.get("game_id", "unknown")
            trial_dict = trial_row.to_dict()
            for col in metric_cols:
                if col in trial_dict and pd.notna(trial_dict[col]):
                    feature_name = f"{game_id}__{col}"
                    row[feature_name] = trial_dict[col]
        rows.append(row)

    return pd.DataFrame(rows).set_index("model_key")


def run_pca(feature_matrix: pd.DataFrame, n_components: int = 7) -> dict:
    """
    Run PCA on the feature matrix.
    Returns dict with loadings, explained variance, scores.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Drop columns with too many NaNs (keep only those with >50% non-null)
    valid_cols = feature_matrix.columns[feature_matrix.notna().sum() > len(feature_matrix) * 0.5]
    # Use column-mean imputation (not zero-fill, since 0 can be meaningful)
    X = feature_matrix[valid_cols].apply(lambda c: c.fillna(c.mean()))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    scores_df = pd.DataFrame(
        scores,
        index=feature_matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return {
        "loadings": loadings,
        "scores": scores_df,
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }


def run_factor_analysis(feature_matrix: pd.DataFrame,
                        n_factors: int = 5) -> dict:
    """
    Run exploratory factor analysis with varimax rotation.
    """
    try:
        from sklearn.decomposition import FactorAnalysis
    except ImportError:
        print("sklearn required for factor analysis")
        return {}

    valid_cols = feature_matrix.columns[feature_matrix.notna().sum() > len(feature_matrix) * 0.5]
    X = feature_matrix[valid_cols].apply(lambda c: c.fillna(c.mean()))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_factors = min(n_factors, X_scaled.shape[0], X_scaled.shape[1])
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    scores = fa.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        fa.components_.T,
        index=valid_cols,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    scores_df = pd.DataFrame(
        scores,
        index=feature_matrix.index,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    return {
        "loadings": loadings,
        "scores": scores_df,
        "noise_variance": fa.noise_variance_,
    }


def compute_cronbach_alpha(items: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for internal consistency."""
    items = items.dropna(axis=1)
    n_items = items.shape[1]
    if n_items < 2:
        return np.nan

    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)

    if total_var == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return round(alpha, 4)


def run_full_analysis(profiles: pd.DataFrame) -> dict:
    """
    Run complete factor analysis pipeline.
    """
    print("Building feature matrix...")
    features = build_feature_matrix(profiles)
    print(f"  Feature matrix: {features.shape[0]} models × {features.shape[1]} features")

    if features.shape[0] < 3:
        print("  Too few models for factor analysis. Need at least 3.")
        return {"features": features}

    print("Running PCA...")
    pca_results = run_pca(features)
    print(f"  Top 3 PC explained variance: "
          f"{pca_results['explained_variance'][:3].round(3)}")

    print("Running factor analysis...")
    fa_results = run_factor_analysis(features)

    # Save results
    output_dir = DATA_PROCESSED / "factor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    features.to_csv(output_dir / "feature_matrix.csv")
    pca_results["loadings"].to_csv(output_dir / "pca_loadings.csv")
    pca_results["scores"].to_csv(output_dir / "pca_scores.csv")

    if fa_results:
        fa_results["loadings"].to_csv(output_dir / "fa_loadings.csv")
        fa_results["scores"].to_csv(output_dir / "fa_scores.csv")

    print(f"  Results saved to {output_dir}")

    return {
        "features": features,
        "pca": pca_results,
        "factor_analysis": fa_results,
    }


if __name__ == "__main__":
    profiles_path = DATA_PROCESSED / "behavioral_profiles.csv"
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        run_full_analysis(profiles)
    else:
        print(f"No profiles found at {profiles_path}. Run compute_behavioral_profiles.py first.")

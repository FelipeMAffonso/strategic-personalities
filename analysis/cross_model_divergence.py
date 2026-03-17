"""
Cross-Model Divergence — Hodoscope Behavioral Fingerprinting
==============================================================
Level 3 analysis: how models THINK about strategic decisions, not just
what they choose. Embeds reasoning traces, computes divergence matrices,
and produces cluster visualisations.

Pipeline:
  1. Load reasoning traces from match result JSONs (runner output format)
  2. Embed traces using sentence-transformers (TF-IDF fallback)
  3. Compute per-model trace distributions (centroids, spread)
  4. Jensen-Shannon distance between choice distributions (per-game + aggregate)
  5. Cosine distance between trace embedding centroids
  6. UMAP and t-SNE 2D projections (centroid-level for figures, trace-level optional)
  7. Hierarchical clustering with dendrogram (average linkage for cosine metric)
  8. Provider separation analysis (silhouette scores, within/between distances)
  9. Behavioral signature extraction (choice patterns, reasoning lexicon,
     conditional cooperation, transition matrices, endgame effects)

The hodoscope metaphor: like a particle physics hodoscope that tracks
particle trajectories, we track the "reasoning trajectories" of models
through strategic decision space.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
HODOSCOPE_DIR = DATA_PROCESSED / "hodoscope"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def get_provider(model_key: str) -> str:
    """Extract provider name from a model key string."""
    mk = model_key.lower()
    if "claude" in mk:
        return "anthropic"
    if any(p in mk for p in ("gpt", "o1-", "o3-", "o4-")):
        return "openai"
    if any(p in mk for p in ("gemini", "gemma")):
        return "google"
    if "deepseek" in mk:
        return "deepseek"
    if "llama" in mk:
        return "meta"
    if "qwen" in mk:
        return "alibaba"
    if "kimi" in mk:
        return "moonshot"
    if "mistral" in mk or "mixtral" in mk:
        return "mistral"
    return "other"


def get_model_family(model_key: str) -> str:
    """Extract model family for generational grouping."""
    mk = model_key.lower()
    if "claude" in mk:
        if "haiku" in mk:
            return "claude-haiku"
        if "sonnet" in mk:
            return "claude-sonnet"
        if "opus" in mk:
            return "claude-opus"
        return "claude"
    if "gpt-5" in mk or "gpt5" in mk:
        return "gpt-5"
    if "gpt-4" in mk or "gpt4" in mk:
        return "gpt-4"
    if "o1" in mk or "o3" in mk or "o4" in mk:
        return "openai-reasoning"
    if "gemini-3" in mk:
        return "gemini-3"
    if "gemini-2.5" in mk:
        return "gemini-2.5"
    if "gemini-2" in mk:
        return "gemini-2"
    if "deepseek" in mk:
        return "deepseek"
    if "llama" in mk:
        return "llama"
    if "qwen" in mk:
        return "qwen"
    return model_key


# ---------------------------------------------------------------------------
# 1. Trace Loading
# ---------------------------------------------------------------------------

def load_reasoning_traces(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load all reasoning traces from raw match JSON files.

    Handles the runner output format where match rounds are nested under
    "match_detail" and top-level fields contain game_id, game_category,
    model_key, opponent, condition, trial_num.

    Returns DataFrame with columns:
        model_key, game_id, game_category, trial_num, round_num,
        choice, reasoning_trace, opponent, condition, provider,
        model_family, matchup_type
    """
    if data_dir is None:
        data_dir = DATA_RAW

    records = []
    n_files = 0
    n_skipped = 0

    for json_path in data_dir.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            n_skipped += 1
            continue

        n_files += 1

        # Top-level fields from runner output
        game_id = data.get("game_id", "")
        game_category = data.get("game_category", "")
        model_key = data.get("model_key", "")
        opponent = data.get("opponent", "")
        condition = data.get("condition", "baseline")
        trial_num = data.get("trial_num", 0)
        matchup_type = data.get("matchup_type", "")

        # Get rounds from match_detail (runner nests the full MatchResult here)
        match_detail = data.get("match_detail", {})
        rounds = match_detail.get("rounds", [])
        players = match_detail.get("players", {})

        # Fallback: if rounds are at top level (standalone MatchResult JSON)
        if not rounds and "rounds" in data:
            rounds = data["rounds"]
            players = data.get("players", {})
            game_category = data.get("game_category",
                                     match_detail.get("game_category", ""))

        # If game_category still missing, try to infer from game registry
        if not game_category:
            game_category = _infer_category(game_id)

        for round_data in rounds:
            round_num = round_data.get("round_num", 0)
            parsed_choices = round_data.get("parsed_choices", {})
            reasoning = round_data.get("reasoning", {})

            if not reasoning:
                continue

            for pid_str, trace in reasoning.items():
                if not trace or not isinstance(trace, str):
                    continue

                # Skip strategy-generated traces (very short, no real
                # reasoning, just e.g. "<answer>Option N</answer>")
                if len(trace.strip()) < 30:
                    continue

                pid = int(pid_str) if isinstance(pid_str, str) else pid_str
                choice = parsed_choices.get(str(pid),
                                            parsed_choices.get(pid, ""))

                # Determine which model produced this trace
                player_name = players.get(str(pid), players.get(pid, ""))
                if not player_name:
                    # Use top-level model_key/opponent based on player id
                    player_name = model_key if pid == 0 else opponent

                # Determine opponent identity
                opponent_name = ""
                if len(players) == 2:
                    other_pid = 1 - pid
                    opponent_name = players.get(str(other_pid),
                                                players.get(other_pid, ""))
                    if not opponent_name:
                        opponent_name = opponent if pid == 0 else model_key

                # Get opponent's choice for this round (needed for
                # conditional cooperation analysis)
                other_pid = 1 - pid
                opponent_choice = parsed_choices.get(
                    str(other_pid), parsed_choices.get(other_pid, ""))

                records.append({
                    "model_key": player_name,
                    "game_id": game_id,
                    "game_category": game_category,
                    "trial_num": trial_num,
                    "round_num": round_num,
                    "choice": choice,
                    "opponent_choice": opponent_choice,
                    "reasoning_trace": trace,
                    "opponent": opponent_name,
                    "condition": condition,
                    "matchup_type": matchup_type,
                    "provider": get_provider(player_name),
                    "model_family": get_model_family(player_name),
                })

    if n_skipped:
        print(f"  Warning: skipped {n_skipped} unreadable files")

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"  Parsed {n_files} match files -> {len(df)} trace records")
    return df


def _infer_category(game_id: str) -> str:
    """Fallback category inference from game_id when category field is missing."""
    category_map = {
        "pd_": "cooperation",
        "pg_": "cooperation",
        "commons": "cooperation",
        "diners": "cooperation",
        "el_farol": "cooperation",
        "bos_": "coordination",
        "stag_": "coordination",
        "matching_": "coordination",
        "focal_": "coordination",
        "ultimatum": "fairness",
        "dictator": "fairness",
        "third_party": "fairness",
        "beauty_": "depth",
        "centipede": "depth",
        "money_request": "depth",
        "trust": "trust",
        "gift_": "trust",
        "repeated_trust": "trust",
        "first_price": "competition",
        "second_price": "competition",
        "all_pay": "competition",
        "blotto": "competition",
        "nash_demand": "negotiation",
        "alternating": "negotiation",
        "multi_issue": "negotiation",
        "chicken": "risk",
        "signaling": "risk",
        "cheap_talk": "risk",
    }
    for prefix, category in category_map.items():
        if game_id.startswith(prefix) or prefix in game_id:
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# 2. Trace Embedding
# ---------------------------------------------------------------------------

def embed_traces(traces: pd.DataFrame,
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 256,
                 max_traces: int = 50_000) -> np.ndarray:
    """
    Embed reasoning traces using sentence-transformers.

    If there are more than max_traces traces, samples uniformly per model
    to keep computation tractable.

    Returns:
        np.ndarray of shape (len(traces_used), embedding_dim)
    """
    if len(traces) > max_traces:
        print(f"  Subsampling from {len(traces)} to {max_traces} traces "
              f"(uniform per model)")
        traces = _subsample_traces(traces, max_traces)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed. Using TF-IDF fallback.")
        return _embed_traces_tfidf(traces)

    model = SentenceTransformer(model_name)
    texts = traces["reasoning_trace"].fillna("").tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


def _subsample_traces(traces: pd.DataFrame, max_total: int) -> pd.DataFrame:
    """Subsample traces uniformly across models."""
    models = traces["model_key"].unique()
    per_model = max(1, max_total // len(models))
    samples = []
    for model in models:
        model_data = traces[traces["model_key"] == model]
        if len(model_data) > per_model:
            samples.append(model_data.sample(n=per_model, random_state=42))
        else:
            samples.append(model_data)
    return pd.concat(samples).reset_index(drop=True)


def _embed_traces_tfidf(traces: pd.DataFrame) -> np.ndarray:
    """Fallback: TF-IDF + SVD embeddings when sentence-transformers unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    texts = traces["reasoning_trace"].fillna("").tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)

    # Reduce to 384 dimensions (similar to MiniLM output)
    n_components = min(384, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    if n_components < 2:
        return tfidf_matrix.toarray()

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


# ---------------------------------------------------------------------------
# 3. Per-Model Trace Distributions
# ---------------------------------------------------------------------------

def compute_model_centroids(traces: pd.DataFrame,
                            embeddings: np.ndarray) -> dict:
    """
    Compute per-model embedding centroids and spread.

    Returns dict of model_key -> {centroid, n_traces, spread, std}
    """
    traces = traces.copy()
    traces["_idx"] = range(len(traces))

    centroids = {}
    for model_key, group in traces.groupby("model_key"):
        idxs = group["_idx"].values
        model_embs = embeddings[idxs]

        centroid = model_embs.mean(axis=0)
        dists = np.linalg.norm(model_embs - centroid, axis=1)
        spread = float(np.mean(dists))
        std = float(np.std(dists))

        centroids[model_key] = {
            "centroid": centroid,
            "n_traces": len(idxs),
            "spread": spread,
            "std": std,
        }

    return centroids


def compute_category_centroids(traces: pd.DataFrame,
                               embeddings: np.ndarray) -> dict:
    """
    Compute per-model per-category centroids.
    Returns nested dict: model_key -> game_category -> {centroid, n_traces}
    """
    traces = traces.copy()
    traces["_idx"] = range(len(traces))

    result = defaultdict(dict)
    for (model_key, category), group in traces.groupby(["model_key", "game_category"]):
        idxs = group["_idx"].values
        centroid = embeddings[idxs].mean(axis=0)
        result[model_key][category] = {
            "centroid": centroid,
            "n_traces": len(idxs),
        }

    return dict(result)


# ---------------------------------------------------------------------------
# 4. Jensen-Shannon Distance on Choice Distributions
# ---------------------------------------------------------------------------

def _is_numeric_game(game_id: str) -> bool:
    """Check if a game produces numeric (continuous) choices."""
    numeric_prefixes = [
        "beauty_contest", "centipede", "money_request",
        "auction_", "nash_demand", "alternating_offers",
        "multi_issue", "trust_berg", "gift_exchange",
        "dictator", "public_goods", "commons",
    ]
    for prefix in numeric_prefixes:
        if game_id.startswith(prefix) or prefix in game_id:
            return True
    return False


def _bin_numeric_choices(choices: list, n_bins: int = 10,
                         min_val: float = 0, max_val: float = 100) -> list:
    """Bin continuous numeric choices into discrete categories."""
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = []
    for c in choices:
        try:
            val = float(c)
            idx = int(np.clip(np.digitize(val, bin_edges) - 1, 0, n_bins - 1))
            binned.append(f"bin_{idx}")
        except (ValueError, TypeError):
            binned.append("other")
    return binned


def compute_choice_jsd(traces: pd.DataFrame) -> dict:
    """
    Compute JSD between models' choice distributions per game.

    For discrete-choice games: uses raw choice labels.
    For numeric games: bins continuous values into 10 equal-width bins
    to make JSD meaningful (avoids treating 49 and 50 as completely
    different categories).

    Note: scipy.spatial.distance.jensenshannon returns the JS *distance*
    (square root of JS divergence). Values in [0, 1] where 0 = identical
    distributions and 1 = maximally different.

    Returns dict of game_id -> pd.DataFrame (models x models JS distance matrix)
    """
    results = {}

    for game_id, game_data in traces.groupby("game_id"):
        models = sorted(game_data["model_key"].unique())
        if len(models) < 2:
            continue

        # For numeric games, bin the choices first
        is_numeric = _is_numeric_game(game_id)

        if is_numeric:
            # Determine range from observed values
            numeric_vals = []
            for c in game_data["choice"].dropna():
                try:
                    numeric_vals.append(float(c))
                except (ValueError, TypeError):
                    pass
            if not numeric_vals:
                continue
            lo = min(numeric_vals)
            hi = max(numeric_vals)
            if hi == lo:
                hi = lo + 1  # avoid zero-width bins

            # Bin all choices
            all_binned = _bin_numeric_choices(
                game_data["choice"].dropna().tolist(),
                n_bins=10, min_val=lo, max_val=hi,
            )
            all_choices = sorted(set(all_binned))
        else:
            all_choices = sorted(game_data["choice"].dropna().unique())

        choice_idx = {c: i for i, c in enumerate(all_choices)}
        n_choices = len(all_choices)
        if n_choices < 1:
            continue

        distributions = {}
        for model in models:
            model_data = game_data[game_data["model_key"] == model]
            raw_choices = model_data["choice"].dropna().tolist()

            if is_numeric:
                binned = _bin_numeric_choices(
                    raw_choices, n_bins=10, min_val=lo, max_val=hi,
                )
            else:
                binned = raw_choices

            counts = np.zeros(n_choices)
            for choice in binned:
                if choice in choice_idx:
                    counts[choice_idx[choice]] += 1
            # Laplace smoothing (pseudocount = 1)
            counts += 1
            distributions[model] = counts / counts.sum()

        # Pairwise JS distance
        n = len(models)
        jsd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = jensenshannon(distributions[models[i]],
                                  distributions[models[j]])
                jsd_matrix[i, j] = jsd_matrix[j, i] = d

        results[game_id] = pd.DataFrame(jsd_matrix, index=models, columns=models)

    return results


def compute_aggregate_jsd(per_game_jsd: dict) -> pd.DataFrame:
    """
    Average JS distance across all games to get an overall behavioral
    divergence matrix.
    """
    all_models = set()
    for df in per_game_jsd.values():
        all_models.update(df.index)
    all_models = sorted(all_models)
    n = len(all_models)

    agg = np.zeros((n, n))
    counts = np.zeros((n, n))

    model_idx = {m: i for i, m in enumerate(all_models)}
    for df in per_game_jsd.values():
        for m1 in df.index:
            for m2 in df.columns:
                if m1 in model_idx and m2 in model_idx:
                    i, j = model_idx[m1], model_idx[m2]
                    agg[i, j] += df.loc[m1, m2]
                    counts[i, j] += 1

    counts[counts == 0] = 1
    agg /= counts

    return pd.DataFrame(agg, index=all_models, columns=all_models)


# ---------------------------------------------------------------------------
# 5. Cosine Distance Between Trace Centroids
# ---------------------------------------------------------------------------

def compute_centroid_distances(centroids: dict,
                               metric: str = "cosine") -> pd.DataFrame:
    """
    Compute pairwise distances between model trace centroids.
    Default metric is cosine distance (1 - cosine similarity).
    """
    models = sorted(centroids.keys())
    vecs = np.array([centroids[m]["centroid"] for m in models])

    dist_matrix = squareform(pdist(vecs, metric=metric))
    return pd.DataFrame(dist_matrix, index=models, columns=models)


# ---------------------------------------------------------------------------
# 6. Dimensionality Reduction (UMAP + t-SNE + PCA)
# ---------------------------------------------------------------------------

def compute_centroid_projection(centroids: dict,
                                method: str = "umap") -> pd.DataFrame:
    """
    2D projection of model centroids (one point per model).
    This is the primary figure-level projection.

    Returns DataFrame with: model_key, x, y, provider, model_family
    """
    models = sorted(centroids.keys())
    vecs = np.array([centroids[m]["centroid"] for m in models])

    if method == "umap":
        coords = _umap_2d(vecs, n_neighbors=min(5, len(models) - 1))
    elif method == "tsne":
        coords = _tsne_2d(vecs, perplexity=min(5, len(models) - 1))
    else:
        coords = _pca_2d(vecs)

    return pd.DataFrame({
        "model_key": models,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "provider": [get_provider(m) for m in models],
        "model_family": [get_model_family(m) for m in models],
    })


def compute_trace_projection(embeddings: np.ndarray,
                              traces: pd.DataFrame,
                              method: str = "umap",
                              max_points: int = 10_000) -> pd.DataFrame:
    """
    2D projection of individual trace embeddings (optional, for exploration).
    Subsamples if too many points.

    Returns DataFrame with: model_key, x, y, provider, game_id, game_category
    """
    n = len(embeddings)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
        embeddings = embeddings[idx]
        traces = traces.iloc[idx].reset_index(drop=True)

    if method == "umap":
        coords = _umap_2d(embeddings)
    elif method == "tsne":
        coords = _tsne_2d(embeddings)
    else:
        coords = _pca_2d(embeddings)

    return pd.DataFrame({
        "model_key": traces["model_key"].values,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "provider": traces["provider"].values,
        "game_id": traces["game_id"].values,
        "game_category": traces["game_category"].values,
    })


def _umap_2d(data: np.ndarray, n_neighbors: int = 15,
             min_dist: float = 0.1) -> np.ndarray:
    """UMAP 2D projection. Falls back to PCA if umap-learn not installed."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. Using PCA fallback.")
        return _pca_2d(data)

    n_neighbors = min(n_neighbors, data.shape[0] - 1)
    if n_neighbors < 2:
        return _pca_2d(data)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(data)


def _tsne_2d(data: np.ndarray, perplexity: float = 30) -> np.ndarray:
    """t-SNE 2D projection."""
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, data.shape[0] - 1)
    if perplexity < 2:
        return _pca_2d(data)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        random_state=42,
        init="pca",
    )
    return tsne.fit_transform(data)


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """PCA 2D projection (always available fallback)."""
    from sklearn.decomposition import PCA
    n_comp = min(2, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    result = pca.fit_transform(data)
    if result.shape[1] == 1:
        result = np.column_stack([result, np.zeros(len(result))])
    return result


# ---------------------------------------------------------------------------
# 7. Hierarchical Clustering
# ---------------------------------------------------------------------------

def cluster_models(distance_matrix: pd.DataFrame,
                   method: str = "average") -> dict:
    """
    Hierarchical clustering of models based on behavioral distances.

    Uses average linkage (UPGMA) by default because it is valid for any
    distance metric including cosine. Ward linkage requires Euclidean
    distances and would be mathematically incorrect here.

    Returns dict with linkage matrix, labels, and cluster assignments
    at k=2,3,4,5.
    """
    n = len(distance_matrix)
    if n < 2:
        return {"linkage": None, "labels": distance_matrix.index.tolist(),
                "cluster_assignments": {}}

    condensed = squareform(distance_matrix.values)
    Z = linkage(condensed, method=method)

    cluster_assignments = {}
    for n_clusters in [2, 3, 4, 5]:
        if n_clusters > n:
            continue
        labels = fcluster(Z, n_clusters, criterion="maxclust")
        cluster_assignments[f"k{n_clusters}"] = {
            model: int(label)
            for model, label in zip(distance_matrix.index, labels)
        }

    return {
        "linkage": Z,
        "labels": distance_matrix.index.tolist(),
        "cluster_assignments": cluster_assignments,
    }


# ---------------------------------------------------------------------------
# 8. Provider Separation Analysis
# ---------------------------------------------------------------------------

def compute_provider_separation(distance_matrix: pd.DataFrame) -> dict:
    """
    Test whether models from same provider cluster together.
    Uses silhouette score with provider as label.
    Also computes within-provider vs between-provider distances.
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    models = distance_matrix.index.tolist()
    providers = [get_provider(m) for m in models]

    unique_providers = set(providers)
    n_samples = len(models)
    if len(unique_providers) < 2:
        return {"silhouette_score": float("nan"),
                "provider_labels": {m: p for m, p in zip(models, providers)}}

    # Silhouette requires: 2 <= n_labels < n_samples
    # If every model is from a different provider (n_labels == n_samples),
    # silhouette cannot compute. Also need at least one provider with 2+ models.
    provider_counts = Counter(providers)
    n_labels = len(unique_providers)
    has_multi_model_provider = any(c >= 2 for c in provider_counts.values())

    if n_labels >= n_samples or not has_multi_model_provider:
        # Can't compute silhouette, but still compute within/between distances
        within_dists = []
        between_dists = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = distance_matrix.values[i, j]
                if providers[i] == providers[j]:
                    within_dists.append(d)
                else:
                    between_dists.append(d)
        within_mean = float(np.mean(within_dists)) if within_dists else None
        between_mean = float(np.mean(between_dists)) if between_dists else None
        return {
            "silhouette_score": float("nan"),
            "mean_within_provider_distance": round(within_mean, 4) if within_mean is not None else None,
            "mean_between_provider_distance": round(between_mean, 4) if between_mean is not None else None,
            "provider_labels": {m: p for m, p in zip(models, providers)},
            "n_providers": n_labels,
            "note": "silhouette not computable (need at least one provider with 2+ models)",
        }

    # Silhouette analysis
    sil = silhouette_score(distance_matrix.values, providers,
                           metric="precomputed")
    per_sample = silhouette_samples(distance_matrix.values, providers,
                                    metric="precomputed")

    # Within vs between provider distances
    within_dists = []
    between_dists = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = distance_matrix.values[i, j]
            if providers[i] == providers[j]:
                within_dists.append(d)
            else:
                between_dists.append(d)

    within_mean = float(np.mean(within_dists)) if within_dists else None
    between_mean = float(np.mean(between_dists)) if between_dists else None

    return {
        "silhouette_score": round(float(sil), 4),
        "per_model_silhouette": {
            m: round(float(s), 4) for m, s in zip(models, per_sample)
        },
        "mean_within_provider_distance": round(within_mean, 4) if within_mean is not None else None,
        "mean_between_provider_distance": round(between_mean, 4) if between_mean is not None else None,
        "provider_separation_ratio": (
            round(between_mean / within_mean, 4)
            if within_mean and between_mean and within_mean > 0
            else None
        ),
        "provider_labels": {m: p for m, p in zip(models, providers)},
        "n_providers": len(unique_providers),
    }


# ---------------------------------------------------------------------------
# 9. Behavioral Signature Extraction
# ---------------------------------------------------------------------------

def extract_behavioral_signatures(traces: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a multi-dimensional behavioral signature per model.
    Combines choice-level metrics with reasoning-trace features.

    Returns DataFrame (models x signature dimensions):
    - Per-game choice entropy and consistency
    - First-round choice bias
    - Endgame effect (last 3 rounds vs. earlier)
    - Conditional cooperation (cooperate|opponent cooperated vs. defected)
    - Transition probabilities (cooperate->defect, defect->cooperate)
    - Reasoning trace length statistics
    - Lexical features (mentions of strategic concepts per 1000 words)
    """
    records = []
    for model_key, model_data in traces.groupby("model_key"):
        sig = {"model_key": model_key}

        # Per-game metrics
        for game_id, game_data in model_data.groupby("game_id"):
            choices = game_data["choice"].dropna().tolist()
            total = len(choices)
            if total == 0:
                continue

            counts = Counter(choices)

            # Choice entropy
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
            sig[f"{game_id}__entropy"] = round(entropy, 4)
            sig[f"{game_id}__consistency"] = round(
                1 - entropy / max_entropy if max_entropy > 0 else 1, 4
            )

            # Dominant choice
            most_common = counts.most_common(1)[0]
            sig[f"{game_id}__dominant_choice"] = most_common[0]
            sig[f"{game_id}__dominant_rate"] = round(most_common[1] / total, 4)

            # First-round bias
            first_round = game_data[game_data["round_num"] == 1]
            if len(first_round) > 0:
                first_choices = first_round["choice"].dropna().tolist()
                if first_choices:
                    first_counts = Counter(first_choices)
                    first_most_common = first_counts.most_common(1)[0]
                    sig[f"{game_id}__first_round_choice"] = first_most_common[0]
                    sig[f"{game_id}__first_round_rate"] = round(
                        first_most_common[1] / len(first_choices), 4
                    )

            # Endgame effect: last 3 rounds vs. earlier
            max_round = game_data["round_num"].max()
            if max_round >= 5:
                endgame = game_data[game_data["round_num"] > max_round - 3]
                early = game_data[game_data["round_num"] <= max_round - 3]

                endgame_choices = endgame["choice"].dropna()
                early_choices = early["choice"].dropna()

                if len(endgame_choices) > 0 and len(early_choices) > 0:
                    # Rate of the dominant choice in endgame vs. early
                    dominant = most_common[0]
                    end_rate = sum(1 for c in endgame_choices if c == dominant) / len(endgame_choices)
                    early_rate = sum(1 for c in early_choices if c == dominant) / len(early_choices)
                    sig[f"{game_id}__endgame_shift"] = round(end_rate - early_rate, 4)

        # Conditional cooperation and transition probabilities
        # (only for cooperation-category games with clear cooperate/defect options)
        coop_games = model_data[model_data["game_category"] == "cooperation"]
        if len(coop_games) > 0:
            coop_rates, transition_probs = _compute_conditional_patterns(coop_games)
            sig.update(coop_rates)
            sig.update(transition_probs)

        # Reasoning trace features
        all_traces = model_data["reasoning_trace"].fillna("").tolist()
        all_traces = [t for t in all_traces if len(t.strip()) > 0]

        if all_traces:
            lengths = [len(t) for t in all_traces]
            word_counts = [len(t.split()) for t in all_traces]
            sig["trace_mean_chars"] = round(float(np.mean(lengths)), 1)
            sig["trace_std_chars"] = round(float(np.std(lengths)), 1)
            sig["trace_mean_words"] = round(float(np.mean(word_counts)), 1)

            # Lexical features (per 1000 words)
            all_text = " ".join(all_traces).lower()
            n_words = len(all_text.split())
            for keyword, key in _LEXICAL_FEATURES:
                count = all_text.count(keyword)
                sig[key] = round(count / max(n_words, 1) * 1000, 2)

        records.append(sig)

    return pd.DataFrame(records).set_index("model_key")


# Strategic vocabulary for lexical analysis
_LEXICAL_FEATURES = [
    ("cooperat", "lex_cooperation"),
    ("defect", "lex_defection"),
    ("fair", "lex_fairness"),
    ("trust", "lex_trust"),
    ("strateg", "lex_strategy"),
    ("optim", "lex_optimality"),
    ("equilibri", "lex_equilibrium"),
    ("risk", "lex_risk"),
    ("oppon", "lex_opponent"),
    ("mutual", "lex_mutual"),
    ("punish", "lex_punishment"),
    ("reward", "lex_reward"),
    ("reciproc", "lex_reciprocity"),
    ("maximiz", "lex_maximizing"),
    ("best response", "lex_best_response"),
    ("nash", "lex_nash"),
    ("pareto", "lex_pareto"),
    ("dominant", "lex_dominant"),
    ("exploit", "lex_exploit"),
    ("forgiv", "lex_forgiveness"),
    ("retali", "lex_retaliation"),
    ("tit for tat", "lex_tit_for_tat"),
    ("long term", "lex_long_term"),
    ("short term", "lex_short_term"),
]


def _compute_conditional_patterns(coop_data: pd.DataFrame) -> tuple[dict, dict]:
    """
    Compute conditional cooperation rates and transition probabilities.

    Conditional cooperation: P(cooperate | opponent cooperated last round)
                           vs P(cooperate | opponent defected last round)

    Transition matrix: P(cooperate->defect), P(defect->cooperate), etc.

    Uses the opponent_choice column from trace loading (round-level data).
    """
    coop_rates = {}
    transitions = {}

    coop_options = {"cooperate", "stag", "cheap", "go", "contribute", "swerve"}

    # Sort by game, trial, and round for sequential analysis
    sorted_data = coop_data.sort_values(["game_id", "trial_num", "round_num"])

    coop_given_coop = 0
    total_given_coop = 0
    coop_given_defect = 0
    total_given_defect = 0

    cc_count = cd_count = dc_count = dd_count = 0

    prev_my = None
    prev_opp = None
    prev_key = None  # (game_id, trial_num) to detect trial boundaries

    for _, row in sorted_data.iterrows():
        my_choice = str(row["choice"]).lower() if pd.notna(row["choice"]) else ""
        is_coop = my_choice in coop_options

        # Get opponent's choice from the opponent_choice column
        opp_choice = str(row.get("opponent_choice", "")).lower()
        opp_is_coop = opp_choice in coop_options if opp_choice else None

        trial_key = (row.get("game_id", ""), row.get("trial_num", 0))

        if prev_key == trial_key and prev_opp is not None:
            # Conditional on opponent's previous choice
            if prev_opp:
                total_given_coop += 1
                if is_coop:
                    coop_given_coop += 1
            else:
                total_given_defect += 1
                if is_coop:
                    coop_given_defect += 1

            # Transition from my previous choice
            if prev_my is not None:
                if prev_my and is_coop:
                    cc_count += 1
                elif prev_my and not is_coop:
                    cd_count += 1
                elif not prev_my and is_coop:
                    dc_count += 1
                else:
                    dd_count += 1

        prev_my = is_coop
        prev_opp = opp_is_coop
        prev_key = trial_key

    if total_given_coop > 0:
        coop_rates["cond_coop_given_coop"] = round(coop_given_coop / total_given_coop, 4)
    if total_given_defect > 0:
        coop_rates["cond_coop_given_defect"] = round(coop_given_defect / total_given_defect, 4)

    total_transitions = cc_count + cd_count + dc_count + dd_count
    if total_transitions > 0:
        from_coop = cc_count + cd_count
        from_defect = dc_count + dd_count
        if from_coop > 0:
            transitions["transition_coop_to_coop"] = round(cc_count / from_coop, 4)
            transitions["transition_coop_to_defect"] = round(cd_count / from_coop, 4)
        if from_defect > 0:
            transitions["transition_defect_to_coop"] = round(dc_count / from_defect, 4)
            transitions["transition_defect_to_defect"] = round(dd_count / from_defect, 4)

    return coop_rates, transitions


# ---------------------------------------------------------------------------
# 10. Cross-Pipeline Integration
# ---------------------------------------------------------------------------

def signatures_to_factor_features(signatures: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hodoscope behavioral signatures into a feature matrix compatible
    with the factor_analysis.py pipeline. Selects numeric features only.
    """
    numeric_cols = signatures.select_dtypes(include=[np.number]).columns
    return signatures[numeric_cols].copy()


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

def run_hodoscope_pipeline(data_dir: Path | None = None,
                           output_dir: Path | None = None) -> dict:
    """
    Run the full hodoscope analysis pipeline.

    Returns dict with all computed artifacts.
    """
    if output_dir is None:
        output_dir = HODOSCOPE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HODOSCOPE — Behavioral Fingerprinting Pipeline")
    print("=" * 60)

    # Step 1: Load traces
    print("\n[1/8] Loading reasoning traces...")
    traces = load_reasoning_traces(data_dir)
    if traces.empty:
        print("  No reasoning traces found. Run experiments first with "
              "record_reasoning=True.")
        return {"status": "no_data"}
    print(f"  Loaded {len(traces)} traces from "
          f"{traces['model_key'].nunique()} models across "
          f"{traces['game_id'].nunique()} games, "
          f"{traces['game_category'].nunique()} categories")

    # Step 2: Embed traces (subsample if needed to keep tractable)
    print("\n[2/8] Embedding reasoning traces...")
    max_traces = 50_000
    if len(traces) > max_traces:
        print(f"  Subsampling from {len(traces)} to {max_traces} traces "
              f"(uniform per model)")
        traces = _subsample_traces(traces, max_traces)
    embeddings = embed_traces(traces, max_traces=len(traces) + 1)  # already subsampled
    print(f"  Shape: {embeddings.shape[0]} traces x {embeddings.shape[1]} dims")

    # Step 3: Compute centroids
    print("\n[3/8] Computing per-model centroids...")
    centroids = compute_model_centroids(traces, embeddings)
    cat_centroids = compute_category_centroids(traces, embeddings)
    for model in sorted(centroids):
        info = centroids[model]
        print(f"  {model}: {info['n_traces']} traces, "
              f"spread={info['spread']:.4f}, std={info['std']:.4f}")

    # Step 4: Choice-level JSD
    print("\n[4/8] Computing Jensen-Shannon distance on choice distributions...")
    per_game_jsd = compute_choice_jsd(traces)
    agg_jsd = compute_aggregate_jsd(per_game_jsd)

    # Save per-game JSD matrices
    jsd_dir = output_dir / "jsd_per_game"
    jsd_dir.mkdir(exist_ok=True)
    for game_id, jsd_df in per_game_jsd.items():
        jsd_df.to_csv(jsd_dir / f"{game_id}_jsd.csv")
    agg_jsd.to_csv(output_dir / "aggregate_jsd.csv")
    print(f"  Computed JSD for {len(per_game_jsd)} games, "
          f"aggregate matrix: {agg_jsd.shape[0]}x{agg_jsd.shape[1]}")

    # Step 5: Centroid distances
    print("\n[5/8] Computing trace centroid distances...")
    centroid_dist = compute_centroid_distances(centroids)
    centroid_dist.to_csv(output_dir / "centroid_distances.csv")

    # Step 6: Dimensionality reduction (centroid-level)
    print("\n[6/8] Computing 2D projections...")
    centroid_umap = compute_centroid_projection(centroids, method="umap")
    centroid_umap.to_csv(output_dir / "centroid_umap.csv", index=False)
    centroid_tsne = compute_centroid_projection(centroids, method="tsne")
    centroid_tsne.to_csv(output_dir / "centroid_tsne.csv", index=False)
    centroid_pca = compute_centroid_projection(centroids, method="pca")
    centroid_pca.to_csv(output_dir / "centroid_pca.csv", index=False)
    print(f"  Saved UMAP, t-SNE, PCA projections for {len(centroids)} models")

    # Optional: trace-level projection (subsampled)
    if len(traces) <= 100_000:
        trace_umap = compute_trace_projection(embeddings, traces, method="umap")
        trace_umap.to_csv(output_dir / "trace_umap.csv", index=False)
        print(f"  Saved trace-level UMAP ({len(trace_umap)} points)")

    # Step 7: Clustering
    print("\n[7/8] Clustering and provider separation analysis...")
    cluster_result = cluster_models(centroid_dist, method="average")
    separation = compute_provider_separation(centroid_dist)
    print(f"  Provider silhouette score: "
          f"{separation.get('silhouette_score', 'N/A')}")
    if separation.get("provider_separation_ratio"):
        print(f"  Separation ratio (between/within): "
              f"{separation['provider_separation_ratio']}")

    # Step 8: Behavioral signatures
    print("\n[8/8] Extracting behavioral signatures...")
    signatures = extract_behavioral_signatures(traces)
    signatures.to_csv(output_dir / "behavioral_signatures.csv")
    print(f"  Extracted {signatures.shape[1]} signature dimensions "
          f"for {signatures.shape[0]} models")

    # Save factor-compatible features for integration with factor_analysis.py
    factor_features = signatures_to_factor_features(signatures)
    factor_features.to_csv(output_dir / "hodoscope_factor_features.csv")
    print(f"  Saved {factor_features.shape[1]} numeric features for "
          f"factor analysis integration")

    # Save comprehensive summary
    summary = {
        "n_traces": len(traces),
        "n_models": int(traces["model_key"].nunique()),
        "n_games": int(traces["game_id"].nunique()),
        "n_categories": int(traces["game_category"].nunique()),
        "embedding_dim": int(embeddings.shape[1]),
        "models": sorted(traces["model_key"].unique().tolist()),
        "games": sorted(traces["game_id"].unique().tolist()),
        "categories": sorted(traces["game_category"].unique().tolist()),
        "provider_separation": separation,
        "cluster_assignments": cluster_result["cluster_assignments"],
        "per_model_stats": {
            m: {
                "n_traces": info["n_traces"],
                "spread": round(info["spread"], 4),
                "std": round(info["std"], 4),
                "provider": get_provider(m),
                "family": get_model_family(m),
            }
            for m, info in centroids.items()
        },
    }
    with open(output_dir / "hodoscope_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nHodoscope pipeline complete. All results saved to {output_dir}/")
    return {
        "traces": traces,
        "embeddings": embeddings,
        "centroids": centroids,
        "category_centroids": cat_centroids,
        "per_game_jsd": per_game_jsd,
        "aggregate_jsd": agg_jsd,
        "centroid_distances": centroid_dist,
        "centroid_projections": {
            "umap": centroid_umap,
            "tsne": centroid_tsne,
            "pca": centroid_pca,
        },
        "clustering": cluster_result,
        "provider_separation": separation,
        "signatures": signatures,
        "factor_features": factor_features,
    }


if __name__ == "__main__":
    results = run_hodoscope_pipeline()

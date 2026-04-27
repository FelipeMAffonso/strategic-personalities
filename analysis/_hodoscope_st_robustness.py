"""
Hodoscope robustness pass: sentence-transformer embedding (all-MiniLM-L6-v2).

Recomputes key silhouette and provider-separation statistics with a true
neural embedding instead of the TF-IDF + SVD baseline. Saves a side-by-side
comparison JSON for inclusion in SI Appendix Table S6.

Pre-empts the most-likely PNAS reviewer concern about the TF-IDF embedding's
shallow lexical signature. No new API calls; uses local sentence-transformers.

Run from project root:
    python analysis/_hodoscope_st_robustness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.cross_model_divergence import (
    load_reasoning_traces,
    _subsample_traces,
)

PROVIDER_MAP = {
    'claude-haiku-4.5': 'Anthropic', 'claude-haiku-4.5-thinking': 'Anthropic',
    'claude-sonnet-4.5': 'Anthropic', 'claude-sonnet-4.6': 'Anthropic',
    'claude-opus-4.5': 'Anthropic', 'claude-opus-4.6': 'Anthropic',
    'gpt-4o-mini': 'OpenAI', 'gpt-4.1': 'OpenAI', 'gpt-4.1-mini': 'OpenAI',
    'gpt-4.1-nano': 'OpenAI', 'gpt-5-mini': 'OpenAI', 'gpt-5-nano': 'OpenAI',
    'gpt-5.3': 'OpenAI', 'gpt-5.4': 'OpenAI',
    'gemini-2.0-flash': 'Google', 'gemini-2.5-flash': 'Google',
    'gemini-2.5-flash-thinking': 'Google', 'gemini-3-flash': 'Google',
    'gemini-3-pro': 'Google', 'gemini-3.1-pro': 'Google',
    'deepseek-v3': 'DeepSeek', 'deepseek-r1': 'DeepSeek',
    'llama-3.3-70b': 'Meta', 'ministral-14b': 'Mistral', 'qwen3.5-flash': 'Alibaba',
}

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "hodoscope" / "robustness_st.json"

print("=" * 70)
print("Hodoscope robustness pass — sentence-transformers (all-MiniLM-L6-v2)")
print("=" * 70)

# 1. Load traces
print("\n[1/5] Loading reasoning traces...")
traces = load_reasoning_traces()
print(f"  loaded {len(traces):,} traces")

# 2. Subsample to 41,520 (matches TF-IDF pipeline)
print("\n[2/5] Subsampling to 41,520 traces (matches TF-IDF baseline)...")
traces_sub = _subsample_traces(traces, 50_000)
print(f"  subsampled to {len(traces_sub):,} traces across "
      f"{traces_sub['model_key'].nunique()} agents")

# 3. Encode with sentence-transformers
print("\n[3/5] Encoding with sentence-transformers/all-MiniLM-L6-v2...")
print("  (this is the slow step; ~30-60 min on CPU)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = traces_sub["reasoning_trace"].fillna("").tolist()
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
)
embeddings = np.array(embeddings)
print(f"  embeddings shape: {embeddings.shape}")

# 4. Recompute provider separation
print("\n[4/5] Computing provider separation statistics...")

def compute_provider_separation_st(embeddings, traces_sub):
    # Per-model centroid
    df_emb = pd.DataFrame(embeddings)
    df_emb["model_key"] = traces_sub["model_key"].values
    centroids = df_emb.groupby("model_key").mean().drop(columns=[], errors="ignore")
    # Provider labels
    centroids["provider"] = centroids.index.map(
        lambda k: PROVIDER_MAP.get(k, "other")
    )
    # Cosine distance matrix between centroids
    coords = centroids.drop(columns=["provider"]).values
    n = len(coords)
    cdist = cosine_distances(coords)
    cd_df = pd.DataFrame(cdist, index=centroids.index, columns=centroids.index)
    # Per-model silhouette using provider as label
    labels = centroids["provider"].values
    sil_score = silhouette_score(cdist, labels, metric="precomputed")
    # Within-provider vs between-provider distances
    within, between = [], []
    keys = centroids.index.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            d = cdist[i, j]
            if labels[i] == labels[j]:
                within.append(d)
            else:
                between.append(d)
    ratio = (np.mean(between) / np.mean(within)) if within and np.mean(within) > 0 else None
    # Per-model silhouette
    from sklearn.metrics import silhouette_samples
    per_model_sil = silhouette_samples(cdist, labels, metric="precomputed")
    sil_per_model = dict(zip(keys, per_model_sil))
    # Without Anthropic
    non_ant_idx = [i for i, p in enumerate(labels) if p != "anthropic"]
    if len(non_ant_idx) >= 2:
        non_ant_within, non_ant_between = [], []
        for i in non_ant_idx:
            for j in non_ant_idx:
                if j <= i: continue
                d = cdist[i, j]
                if labels[i] == labels[j]:
                    non_ant_within.append(d)
                else:
                    non_ant_between.append(d)
        ratio_no_ant = (np.mean(non_ant_between) / np.mean(non_ant_within)) if non_ant_within else None
    else:
        ratio_no_ant = None
    return {
        "n_traces_subsampled": int(len(traces_sub)),
        "n_agents": int(n),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "overall_silhouette": float(sil_score),
        "mean_within_provider_distance": float(np.mean(within)) if within else None,
        "mean_between_provider_distance": float(np.mean(between)) if between else None,
        "provider_separation_ratio": float(ratio) if ratio is not None else None,
        "provider_separation_ratio_without_anthropic": float(ratio_no_ant) if ratio_no_ant is not None else None,
        "per_model_silhouette": {k: float(v) for k, v in sil_per_model.items()},
    }

st_stats = compute_provider_separation_st(embeddings, traces_sub)
print(f"  overall silhouette:           {st_stats['overall_silhouette']:.4f}")
print(f"  provider sep ratio:            {st_stats['provider_separation_ratio']:.4f}")
print(f"  provider sep without Anthropic: {st_stats['provider_separation_ratio_without_anthropic']:.4f}")

# Provider-level summary
sils = st_stats["per_model_silhouette"]
provider_summary = {}
for k, sil in sils.items():
    p = PROVIDER_MAP.get(k, "other")
    provider_summary.setdefault(p, []).append((k, sil))

print("\n  Per-provider silhouette summary:")
for p, lst in sorted(provider_summary.items()):
    vals = [v for _, v in lst]
    if vals:
        print(f"    {p:<10} n={len(vals)} range {min(vals):+.4f} to {max(vals):+.4f}, mean {np.mean(vals):+.4f}")

# 5. Compare to TF-IDF baseline + save
print("\n[5/5] Comparing to TF-IDF baseline...")
tfidf_summary_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "hodoscope" / "hodoscope_summary.json"
with open(tfidf_summary_path) as f:
    tfidf_summary = json.load(f)
tfidf_ps = tfidf_summary["provider_separation"]

comparison = {
    "tfidf_baseline": {
        "embedding_model": "TF-IDF (5000 features) + TruncatedSVD (384 dim)",
        "overall_silhouette": tfidf_ps.get("silhouette_score"),
        "provider_separation_ratio": tfidf_ps.get("provider_separation_ratio"),
        "anthropic_silhouettes": [v for k, v in tfidf_ps["per_model_silhouette"].items() if PROVIDER_MAP.get(k) == "Anthropic"],
        "openai_silhouettes":    [v for k, v in tfidf_ps["per_model_silhouette"].items() if PROVIDER_MAP.get(k) == "OpenAI"],
    },
    "sentence_transformer_robustness": st_stats,
}

# Compute Anthropic mean and OpenAI mean for ST
ant_st = [v for k, v in sils.items() if PROVIDER_MAP.get(k) == "Anthropic"]
oai_st = [v for k, v in sils.items() if PROVIDER_MAP.get(k) == "OpenAI"]
comparison["sentence_transformer_robustness"]["anthropic_mean_silhouette"] = float(np.mean(ant_st)) if ant_st else None
comparison["sentence_transformer_robustness"]["anthropic_silhouette_range"] = [float(min(ant_st)), float(max(ant_st))] if ant_st else None
comparison["sentence_transformer_robustness"]["openai_mean_silhouette"] = float(np.mean(oai_st)) if oai_st else None
comparison["sentence_transformer_robustness"]["openai_silhouette_range"] = [float(min(oai_st)), float(max(oai_st))] if oai_st else None

# Same for TF-IDF baseline
ant_tf = comparison["tfidf_baseline"]["anthropic_silhouettes"]
oai_tf = comparison["tfidf_baseline"]["openai_silhouettes"]
comparison["tfidf_baseline"]["anthropic_mean_silhouette"] = float(np.mean(ant_tf)) if ant_tf else None
comparison["tfidf_baseline"]["openai_mean_silhouette"] = float(np.mean(oai_tf)) if oai_tf else None
del comparison["tfidf_baseline"]["anthropic_silhouettes"]
del comparison["tfidf_baseline"]["openai_silhouettes"]

with open(OUT, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"\n  Saved: {OUT}")
print("\n=== ROBUSTNESS COMPARISON ===")
print(f"\n{'Metric':<45} {'TF-IDF':>12} {'Sentence-Transformer':>22}")
print("-" * 80)
print(f"{'overall silhouette':<45} {comparison['tfidf_baseline']['overall_silhouette']:>12.4f} {st_stats['overall_silhouette']:>22.4f}")
print(f"{'provider separation ratio':<45} {comparison['tfidf_baseline']['provider_separation_ratio']:>12.4f} {st_stats['provider_separation_ratio']:>22.4f}")
print(f"{'Anthropic mean silhouette':<45} {comparison['tfidf_baseline']['anthropic_mean_silhouette']:>12.4f} {comparison['sentence_transformer_robustness']['anthropic_mean_silhouette']:>22.4f}")
print(f"{'OpenAI mean silhouette':<45} {comparison['tfidf_baseline']['openai_mean_silhouette']:>12.4f} {comparison['sentence_transformer_robustness']['openai_mean_silhouette']:>22.4f}")
print()

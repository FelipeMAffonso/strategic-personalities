"""
Generate Paper Figures — Nature-Quality Publication Figures
===========================================================

Design principles (following Akata et al., Nature Human Behaviour 2025):
  - Okabe-Ito colorblind-safe palette for provider identification
  - Annotated heatmaps with values in every cell (YlGnBu colormap)
  - Connected-dot round-by-round timelines with white marker edges
  - Minimal design: no gridlines, no top/right spines
  - Arial font, 5-7 pt body text, 8 pt bold lowercase panel labels
  - pdf.fonttype = 42 for TrueType embedding
  - 600 DPI output at Nature column widths

Figures generated
-----------------
Main:
  fig1_behavioral_profiles   25x8 annotated heatmap (strategic personality)
  fig2_cooperation_divergence Horizontal Cleveland dot plot, Wilson CIs
  fig3_generational_drift     Triptych: generation trajectories per provider
  fig4_endgame_curves         Round-by-round cooperation + final-vs-early
  fig5_reciprocity_profiles   P(C|C) vs P(C|D) scatter with quadrants
  fig6_provider_clustering    UMAP embedding + silhouette scores

Extended Data:
  ed_fig1_radar_charts        25 model radar charts (5x5 grid)
  ed_fig2_behavioral_space    UMAP trace embedding scatter
  ed_fig3_dendrogram          Average-linkage clustering dendrogram
  ed_fig4_jsd_matrices        Aggregate + per-category JSD heatmaps
  ed_fig5_round_timelines     Round-by-round cooperation trajectories
  ed_fig6_factor_loadings     PCA factor loadings + scree plot
  ed_fig7_lexical_features    Strategy term frequencies (horizontal bars)
  ed_fig8_crossplay_matrix    Annotated cross-play heatmap
  ed_fig9_cost_performance    Cost vs cooperation/depth + provider totals
  ed_fig10_coverage           Model x game trial-count heatmap

Usage:
  python analysis/generate_paper_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from scipy.spatial import ConvexHull
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent
DATA    = PROJECT / "dashboard" / "app" / "public" / "data"
PROC    = PROJECT / "data" / "processed"
OUT     = PROJECT / "paper" / "figures"

# ──────────────────────────────────────────────────────────────────────
# Nature rcParams
# ──────────────────────────────────────────────────────────────────────

RC = {
    "font.family":          "Arial",
    "font.size":            7,
    "axes.labelsize":       7,
    "axes.titlesize":       8,
    "xtick.labelsize":      6,
    "ytick.labelsize":      6,
    "legend.fontsize":      6,
    "figure.dpi":           150,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.04,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "axes.linewidth":       0.5,
    "xtick.major.width":    0.4,
    "ytick.major.width":    0.4,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
    "legend.frameon":       False,
    "legend.handlelength":  1.0,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    "legend.borderpad":     0.3,
    "legend.labelspacing":  0.25,
}

# Column widths (inches) — Nature guidelines
SINGLE = 3.5          # 89 mm
ONEHALF = 5.5         # 140 mm
DOUBLE = 7.2          # 183 mm

# ──────────────────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette
# ──────────────────────────────────────────────────────────────────────

C = {
    "anthropic":  "#0072B2",
    "openai":     "#D55E00",
    "google":     "#009E73",
    "deepseek":   "#E69F00",
    "meta":       "#CC79A7",
    "mistral":    "#56B4E9",
    "alibaba":    "#332288",
    "strategy":   "#BBBBBB",
    "agg":        "#333333",
}

PROV_ORDER = ["anthropic", "openai", "google", "deepseek",
              "meta", "mistral", "alibaba"]
PROV_LABEL = {p: p.capitalize() for p in PROV_ORDER}
PROV_LABEL["openai"] = "OpenAI"
PROV_LABEL["deepseek"] = "DeepSeek"

# ──────────────────────────────────────────────────────────────────────
# Short names and categories
# ──────────────────────────────────────────────────────────────────────

SHORT = {
    "claude-haiku-4.5":          "Haiku 4.5",
    "claude-haiku-4.5-thinking": "Haiku 4.5 T",
    "claude-sonnet-4.5":         "Sonnet 4.5",
    "claude-sonnet-4.6":         "Sonnet 4.6",
    "claude-opus-4.5":           "Opus 4.5",
    "claude-opus-4.6":           "Opus 4.6",
    "gpt-4o-mini":               "GPT-4o Mini",
    "gpt-4.1":                   "GPT-4.1",
    "gpt-4.1-mini":              "GPT-4.1 Mini",
    "gpt-4.1-nano":              "GPT-4.1 Nano",
    "gpt-5-mini":                "GPT-5 Mini",
    "gpt-5-nano":                "GPT-5 Nano",
    "gpt-5.3":                   "GPT-5.3",
    "gpt-5.4":                   "GPT-5.4",
    "gemini-2.0-flash":          "Gem 2.0 Flash",
    "gemini-2.5-flash":          "Gem 2.5 Flash",
    "gemini-2.5-flash-thinking": "Gem 2.5 Flash T",
    "gemini-3-flash":            "Gem 3 Flash",
    "gemini-3-pro":              "Gem 3 Pro",
    "gemini-3.1-pro":            "Gem 3.1 Pro",
    "deepseek-v3":               "DS V3",
    "deepseek-r1":               "DS R1",
    "llama-3.3-70b":             "LLaMA 70B",
    "ministral-14b":             "Ministral 14B",
    "qwen3.5-flash":             "Qwen 3.5",
}

CAT_ORDER = ["cooperation", "coordination", "trust", "fairness",
             "depth", "competition", "negotiation", "risk"]
CAT_LABEL = {
    "cooperation":  "Cooperation",
    "coordination": "Coordination",
    "trust":        "Trust",
    "fairness":     "Fairness",
    "depth":        "Depth",
    "competition":  "Competition",
    "negotiation":  "Negotiation",
    "risk":         "Risk",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _prov(k):
    lo = k.lower()
    if "claude" in lo:                                    return "anthropic"
    if any(x in lo for x in ("gpt-", "o1-", "o3-")):     return "openai"
    if "gemini" in lo:                                    return "google"
    if "deepseek" in lo:                                  return "deepseek"
    if "llama" in lo:                                     return "meta"
    if "ministral" in lo:                                 return "mistral"
    if "qwen" in lo:                                      return "alibaba"
    return "strategy"

def _c(k):       return C.get(_prov(k), C["strategy"])
def _s(k):       return SHORT.get(k, k)
def _llm(k):     return _prov(k) != "strategy"

def _wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    w = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return (max(0, c-w), min(1, c+w))

def _sort(keys, coop=None):
    """Sort by provider order, then cooperation rate descending."""
    def key_fn(k):
        p = _prov(k)
        pi = PROV_ORDER.index(p) if p in PROV_ORDER else 99
        r = -(coop[k]["mean"] if coop and k in coop else 0)
        return (pi, r)
    return sorted(keys, key=key_fn)

def _load(fn):
    with open(DATA / fn, "r", encoding="utf-8") as f:
        return json.load(f)

def _lab(ax, t, x=-0.07, y=1.06):
    """Bold lowercase panel label (Nature style)."""
    ax.text(x, y, t, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left")

def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", format=ext)
    plt.close(fig)
    print(f"  + {name}")

def _prov_legend(ax, provs=None, marker="o", ms=3.5, **kwargs):
    if provs is None:
        provs = PROV_ORDER
    hs = [ax.plot([], [], marker, color=C[p], ms=ms, ls="none",
                  label=PROV_LABEL.get(p, p))[0] for p in provs]
    ax.legend(handles=hs, **kwargs)

def _hide_spines(ax):
    for s in ax.spines.values():
        s.set_visible(False)

def _white_grid(ax, n_rows, n_cols, lw=0.4):
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", lw=lw)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", lw=lw)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Strategic Personality Profiles (annotated heatmap)
# ══════════════════════════════════════════════════════════════════════

def fig1_behavioral_profiles():
    """Two-panel figure: (a) annotated heatmap, (b) radar profiles.

    Panel a follows Akata et al. Fig 3a: value in every cell, YlGnBu
    colormap, white grid, provider-coloured y-labels.
    Panel b overlays radar charts for six representative models showing
    convergence in competition/coordination/depth and divergence in
    cooperation/trust.
    """
    print("Fig 1  Behavioral profiles")

    models = _load("models.json")["models"]
    coop   = _load("manuscript_numbers.json")["cooperation_by_model"]
    keys   = _sort([k for k in models if _llm(k)], coop)

    nr, nc = len(keys), len(CAT_ORDER)
    mat = np.full((nr, nc), np.nan)
    for i, mk in enumerate(keys):
        radar = models[mk].get("radar", {})
        for j, cat in enumerate(CAT_ORDER):
            if cat in radar:
                mat[i, j] = radar[cat]["value"] * 100

    # Representative models for radar panel
    RADAR_MODELS = [
        "claude-opus-4.6",
        "gpt-4o-mini",
        "gemini-3-flash",
        "deepseek-r1",
        "gpt-5-nano",
        "llama-3.3-70b",
    ]
    radar_models = [m for m in RADAR_MODELS if m in models]

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(DOUBLE, 0.22 * nr + 1.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.40)

        # ── Panel a: Heatmap ──
        ax = fig.add_subplot(gs[0])
        _lab(ax, "a", x=-0.05)

        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=100, interpolation="nearest")

        for i in range(nr):
            for j in range(nc):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                tc = "white" if v > 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=4.5, color=tc)

        _white_grid(ax, nr, nc, lw=0.5)

        prev = None
        for i, mk in enumerate(keys):
            p = _prov(mk)
            if prev is not None and p != prev:
                ax.axhline(i - 0.5, color="#444444", lw=0.8)
            prev = p

        ax.set_xticks(range(nc))
        ax.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER],
                           rotation=45, ha="right", fontsize=6)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        ax.set_yticks(range(nr))
        ax.set_yticklabels([_s(k) for k in keys], fontsize=5.5)
        for i, mk in enumerate(keys):
            ax.get_yticklabels()[i].set_color(_c(mk))

        _hide_spines(ax)

        cbar = fig.colorbar(im, ax=ax, shrink=0.35, aspect=20, pad=0.02)
        cbar.set_label("Normalised score", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_visible(False)

        # ── Panel b: Radar chart ──
        ax_r = fig.add_subplot(gs[1], polar=True)
        _lab(ax_r, "b", x=-0.15, y=1.12)

        angles = np.linspace(0, 2 * np.pi, nc, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        for mk in radar_models:
            radar = models[mk].get("radar", {})
            vals = [radar.get(c, {}).get("value", 0) * 100 for c in CAT_ORDER]
            vals += vals[:1]
            col = _c(mk)
            ax_r.plot(angles, vals, "o-", color=col, lw=1.0, ms=2.5,
                      mec="white", mew=0.2, label=_s(mk), zorder=3)
            ax_r.fill(angles, vals, color=col, alpha=0.04)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER],
                             fontsize=5)
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks([25, 50, 75])
        ax_r.set_yticklabels(["25", "50", "75"], fontsize=4.5,
                             color="#999999")
        ax_r.tick_params(axis="x", pad=4)

        # Thin grid
        ax_r.grid(color="#E0E0E0", lw=0.3)
        ax_r.spines["polar"].set_visible(False)

        ax_r.legend(loc="lower right", bbox_to_anchor=(1.35, -0.08),
                    fontsize=5, ncol=2, handletextpad=0.3)

        # Use subplots_adjust instead of tight_layout (polar axes issue)
        fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.04)
        _save(fig, "fig1_behavioral_profiles")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cooperation Divergence (horizontal Cleveland dot plot)
# ══════════════════════════════════════════════════════════════════════

def fig2_cooperation_divergence():
    """Horizontal dot plot with Wilson CI whiskers, provider-colour
    coded, alternating background bands for provider groups.
    """
    print("Fig 2  Cooperation divergence")

    coop = _load("manuscript_numbers.json")["cooperation_by_model"]
    keys = _sort(list(coop.keys()), coop)

    n = len(keys)
    means = [coop[k]["mean"] * 100 for k in keys]
    cis   = [_wilson(coop[k]["mean"], coop[k]["n"]) for k in keys]
    los   = [c[0] * 100 for c in cis]
    his   = [c[1] * 100 for c in cis]
    gm    = np.mean(means)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE, 0.20 * n + 0.7))

        # Alternating provider bands
        prev = None
        boundaries = [0]
        for i, k in enumerate(keys):
            p = _prov(k)
            if prev is not None and p != prev:
                boundaries.append(i)
            prev = p
        boundaries.append(n)

        for b in range(len(boundaries) - 1):
            if b % 2 == 1:
                lo_y = n - boundaries[b + 1] - 0.5
                hi_y = n - boundaries[b] - 0.5
                ax.axhspan(lo_y, hi_y, color="#F4F4F4", zorder=0)

        # Grand mean line
        ax.axvline(gm, color="#BBBBBB", ls=":", lw=0.5, zorder=1)

        # Plot dots + CIs
        for i, k in enumerate(keys):
            y = n - 1 - i
            col = _c(k)
            ax.plot([los[i], his[i]], [y, y], "-", color=col, lw=0.9,
                    solid_capstyle="round", zorder=2)
            ax.plot(means[i], y, "o", color=col, ms=4.5,
                    mec="white", mew=0.3, zorder=3)

        ax.set_yticks(range(n))
        ax.set_yticklabels([_s(keys[n - 1 - i]) for i in range(n)],
                           fontsize=5.5)
        for i in range(n):
            ax.get_yticklabels()[i].set_color(_c(keys[n - 1 - i]))

        ax.set_xlabel("Cooperation rate (%)")
        ax.set_xlim(-2, 82)
        ax.set_ylim(-0.5, n - 0.5)

        # Annotate grand mean
        ax.text(gm, n - 0.3, f"{gm:.0f}%", fontsize=5, color="#999999",
                ha="center", va="bottom")

        fig.tight_layout()
        _save(fig, "fig2_cooperation_divergence")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Generational Drift (triptych)
# ══════════════════════════════════════════════════════════════════════

def fig3_generational_drift():
    """Three panels (Anthropic, OpenAI, Google): slope chart with arrows
    showing cooperation change between model generations.  Large arrows
    and delta annotations emphasise the drama of generational shifts
    (e.g. 33x collapse, 7x surge).

    Uses cooperation_by_model from manuscript_numbers.json (cooperation
    games only) to ensure consistency with text numbers.
    """
    print("Fig 3  Generational drift")

    coop = _load("manuscript_numbers.json")["cooperation_by_model"]

    # Provider families in chronological order (cooperation games only)
    families = [
        ("Anthropic", C["anthropic"], [
            ("claude-haiku-4.5",          "Haiku 4.5"),
            ("claude-haiku-4.5-thinking", "Haiku 4.5 T"),
            ("claude-sonnet-4.5",         "Sonnet 4.5"),
            ("claude-sonnet-4.6",         "Sonnet 4.6"),
            ("claude-opus-4.5",           "Opus 4.5"),
            ("claude-opus-4.6",           "Opus 4.6"),
        ]),
        ("OpenAI", C["openai"], [
            ("gpt-4o-mini",   "GPT-4o Mini"),
            ("gpt-4.1-mini",  "GPT-4.1 Mini"),
            ("gpt-4.1-nano",  "GPT-4.1 Nano"),
            ("gpt-5-mini",    "GPT-5 Mini"),
            ("gpt-5-nano",    "GPT-5 Nano"),
        ]),
        ("Google", C["google"], [
            ("gemini-2.0-flash",          "Gem 2.0 Flash"),
            ("gemini-2.5-flash",          "Gem 2.5 Flash"),
            ("gemini-2.5-flash-thinking", "Gem 2.5 Flash T"),
            ("gemini-3-flash",            "Gem 3 Flash"),
        ]),
    ]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.8), sharey=True)

        for ax, (fam, col, model_seq) in zip(axes, families):
            present = [(k, lab) for k, lab in model_seq if k in coop]
            if not present:
                continue

            xs   = list(range(len(present)))
            rate = [coop[k]["mean"] * 100 for k, _ in present]
            labs = [lab for _, lab in present]

            # Wilson CIs
            cis = [_wilson(coop[k]["mean"], coop[k]["n"])
                   for k, _ in present]
            lo  = [c[0] * 100 for c in cis]
            hi  = [c[1] * 100 for c in cis]

            # Slope arrows between consecutive generations
            for i in range(len(xs) - 1):
                delta = rate[i + 1] - rate[i]
                # Arrow colour: darken provider colour for drops, lighten for rises
                arrow_alpha = min(0.6, 0.15 + abs(delta) / 80)
                ax.annotate(
                    "", xy=(xs[i + 1], rate[i + 1]),
                    xytext=(xs[i], rate[i]),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.25,head_length=0.15",
                        color=col, lw=1.4, alpha=arrow_alpha,
                        connectionstyle="arc3,rad=0",
                    ),
                    zorder=2)

                # Delta label at midpoint
                mx = (xs[i] + xs[i + 1]) / 2
                my = (rate[i] + rate[i + 1]) / 2
                sign = "+" if delta > 0 else ""
                if abs(delta) > 3:
                    ax.text(mx + 0.12, my, f"{sign}{delta:.0f}pp",
                            fontsize=4.5, color=col, alpha=0.7,
                            ha="left", va="center", fontweight="bold",
                            zorder=6)

            # CI whiskers
            for i in range(len(xs)):
                ax.plot([i, i], [lo[i], hi[i]], "-", color=col,
                        lw=0.7, alpha=0.35, zorder=2)

            # Dots with white edge (on top of arrows)
            ax.plot(xs, rate, "o", color=col, ms=6.5,
                    mec="white", mew=0.6, zorder=4)

            # Rate labels below each dot
            for i, r in enumerate(rate):
                offset = -6 if r > 10 else 6
                va = "top" if r > 10 else "bottom"
                ax.text(xs[i], r + (offset * 0.15), f"{r:.0f}%",
                        fontsize=4.5, color=col, ha="center", va=va,
                        alpha=0.8, zorder=5)

            ax.set_xticks(xs)
            ax.set_xticklabels(labs, rotation=42, ha="right", fontsize=5)
            ax.set_title(fam, fontsize=7, fontweight="bold", color=col)
            ax.set_xlim(-0.4, len(xs) - 0.6)

        axes[0].set_ylabel("Cooperation rate (%)")
        axes[0].set_ylim(-2, 80)

        for ax, lbl in zip(axes, "abc"):
            _lab(ax, lbl)

        fig.tight_layout(w_pad=1.0)
        _save(fig, "fig3_generational_drift")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Endgame Curves
# ══════════════════════════════════════════════════════════════════════

def fig4_endgame_curves():
    """Panel a: round-by-round cooperation (all models thin, aggregate
    bold, three archetypes highlighted).  Panel b: final-round rate vs
    rounds-1-8 mean (endgame-effect scatter).
    """
    print("Fig 4  Endgame curves")

    aa  = _load("advanced_analytics.json")
    agg = aa["endgame"]["aggregate"]
    bm  = aa["endgame"]["by_model"]

    # Pick three archetypes with DISTINCT colors (not provider colours)
    ARCH_COLORS = {
        "Terminal cooperator":    "#0072B2",  # blue
        "Strategic exploiter":    "#009E73",  # teal
        "Unconditional defector": "#D55E00",  # vermillion
    }
    ARCH_MARKERS = {
        "Terminal cooperator":    "o",
        "Strategic exploiter":    "s",
        "Unconditional defector": "D",
    }

    archetypes = {}
    for k in bm:
        if "opus-4.6" in k:
            archetypes[k] = "Terminal cooperator"
        elif k == "gemini-3-pro":
            archetypes[k] = "Strategic exploiter"
        elif "gpt-5-nano" in k:
            archetypes[k] = "Unconditional defector"

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE, 3.0),
            gridspec_kw={"width_ratios": [1.6, 1]})

        # ── a: Round-by-round ──
        _lab(ax1, "a")

        # Shade final round
        ax1.axvspan(9.5, 10.5, color="#FEE2E2", alpha=0.35, zorder=0)

        # Background models (thin, transparent)
        for k, d in bm.items():
            if k in archetypes:
                continue
            r = d["rounds"]
            y = [v * 100 for v in d["cooperation_rate"]]
            ax1.plot(r, y, "-", color="#CCCCCC", lw=0.3, alpha=0.4, zorder=1)

        # Aggregate bold line
        agg_y = [v * 100 for v in agg["cooperation_rate"]]
        ax1.plot(agg["rounds"], agg_y, "-", color=C["agg"], lw=2.2,
                 zorder=4, label="Aggregate")

        # Archetype highlights with distinct markers and colours
        for k, lbl in archetypes.items():
            d = bm[k]
            r = d["rounds"]
            y = [v * 100 for v in d["cooperation_rate"]]
            col = ARCH_COLORS[lbl]
            mrk = ARCH_MARKERS[lbl]
            ax1.plot(r, y, f"{mrk}-", color=col, ms=3.5, mec="white",
                     mew=0.3, lw=1.5, zorder=5, label=_s(k))

        ax1.set_xlabel("Round")
        ax1.set_ylabel("Cooperation rate (%)")
        ax1.set_xticks(range(1, 11))
        ax1.set_xlim(0.5, 10.5)
        ax1.set_ylim(-3, 105)
        ax1.legend(loc="upper right", fontsize=5.5, ncol=1)

        ax1.text(10, 97, "Final\nround", ha="center", va="top",
                 fontsize=5, color="#B91C1C", style="italic")

        # ── b: Final vs early scatter ──
        _lab(ax2, "b")

        for k, d in bm.items():
            rates = d["cooperation_rate"]
            early = np.mean(rates[:8]) * 100
            final = rates[-1] * 100

            if k in archetypes:
                lbl = archetypes[k]
                col = ARCH_COLORS[lbl]
                mrk = ARCH_MARKERS[lbl]
                ax2.plot(early, final, mrk, color=col, ms=5.5,
                         mec="white", mew=0.3, zorder=4)
                ax2.annotate(_s(k), (early, final), xytext=(4, 4),
                             textcoords="offset points", fontsize=4.5,
                             color=col, fontweight="bold", zorder=5)
            else:
                ax2.plot(early, final, "o", color="#BBBBBB", ms=3,
                         mec="white", mew=0.2, alpha=0.6, zorder=2)

        # Diagonal (no endgame effect)
        ax2.plot([-5, 110], [-5, 110], ":", color="#DDDDDD", lw=0.5, zorder=0)
        ax2.text(62, 67, "No endgame\neffect", fontsize=4.5, color="#BBBBBB",
                 rotation=42, ha="center", va="bottom")

        ax2.set_xlabel("Rounds 1-8 mean (%)")
        ax2.set_ylabel("Round 10 (%)")
        ax2.set_xlim(-5, 105)
        ax2.set_ylim(-5, 105)
        ax2.set_aspect("equal")

        fig.tight_layout(w_pad=2.5)
        _save(fig, "fig4_endgame_curves")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Reciprocity Profiles (scatter)
# ══════════════════════════════════════════════════════════════════════

def fig5_reciprocity_profiles():
    """P(C|C) vs P(C|D) scatter with shaded quadrant regions
    and selectively labelled extreme models.
    """
    print("Fig 5  Reciprocity profiles")

    recip = _load("advanced_analytics.json")["reciprocity"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE, 3.5))

        xs, ys, ks = [], [], []
        for k, d in recip.items():
            x = d["p_coop_after_coop"] * 100
            y = d["p_coop_after_defect"] * 100
            xs.append(x); ys.append(y); ks.append(k)

        y_max = max(ys) + 6 if ys else 35

        # Quadrant shading
        ax.fill_between([55, 102], -3, 15, color="#E8F5E9", alpha=0.30, zorder=0)
        ax.fill_between([-3, 22],  -3, 15, color="#FFEBEE", alpha=0.30, zorder=0)
        ax.fill_between([-3, 22],  15, y_max, color="#FFF8E1", alpha=0.30, zorder=0)

        # Quadrant labels (positioned to avoid data points)
        ax.text(98, 14, "Strict\nreciprocators", ha="right", va="top",
                fontsize=5.5, color="#2E7D32", style="italic")
        ax.text(3, 0.5, "Unconditional\ndefectors", ha="left", va="bottom",
                fontsize=5.5, color="#C62828", style="italic")
        ax.text(3, y_max - 1, "Anti-\nreciprocators", ha="left", va="top",
                fontsize=5.5, color="#E65100", style="italic")

        # Diagonal reference
        ax.plot([-5, 105], [-5, 105], ":", color="#E0E0E0", lw=0.4, zorder=0)

        # All dots
        for i in range(len(xs)):
            ax.plot(xs[i], ys[i], "o", color=_c(ks[i]), ms=5,
                    mec="white", mew=0.3, zorder=3)

        # Label only the single most extreme model per region
        labels_to_place = {}

        # Highest P(C|C) — top strict reciprocator
        by_x = sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)
        labels_to_place[by_x[0]] = (-5, -8, "right")

        # Lowest P(C|C) — most extreme unconditional defector
        by_x_lo = sorted(range(len(xs)), key=lambda i: xs[i])
        labels_to_place[by_x_lo[0]] = (4, -6, "left")

        # Highest P(C|D) — top anti-reciprocator
        by_y = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
        labels_to_place[by_y[0]] = (4, 3, "left")

        # Highest endgame drop (P(C|C) high but P(C|D) low and far from diagonal)
        # = most responsive model
        gaps = [(i, xs[i] - ys[i]) for i in range(len(xs))]
        biggest_gap = sorted(gaps, key=lambda x: x[1], reverse=True)
        for idx, _ in biggest_gap[:1]:
            if idx not in labels_to_place:
                labels_to_place[idx] = (-5, 5, "right")

        for i, (dx, dy, ha) in labels_to_place.items():
            ax.annotate(_s(ks[i]), (xs[i], ys[i]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=4.5, color=_c(ks[i]), ha=ha, zorder=4)

        ax.set_xlabel("P(cooperate | opponent cooperated) (%)")
        ax.set_ylabel("P(cooperate | opponent defected) (%)")
        ax.set_xlim(-3, 102)
        ax.set_ylim(-3, y_max)

        _prov_legend(ax, loc="upper left", ncol=2)

        fig.tight_layout()
        _save(fig, "fig5_reciprocity_profiles")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Provider Clustering (UMAP + silhouette)
# ══════════════════════════════════════════════════════════════════════

def fig6_provider_clustering():
    """Panel a: UMAP embedding with per-provider KDE density contours and
    centroid markers.  Panel b: per-model silhouette scores (horizontal bars).
    """
    print("Fig 6  Provider clustering")

    hodo = _load("hodoscope.json")
    traces = hodo.get("trace_umap", [])

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE, 3.2),
            gridspec_kw={"width_ratios": [1.4, 1]})

        # ── a: UMAP with density ──
        _lab(ax1, "a")

        centroids = hodo["centroid_umap"]
        prov_pts = {}  # provider -> list of (x, y) from centroids

        # KDE density contours from trace-level data
        if HAS_SCIPY and traces:
            prov_trace_pts = {}
            for t in traces:
                mk = t.get("model_key", "")
                p = _prov(mk) if mk else t.get("provider", "").lower()
                prov_trace_pts.setdefault(p, {"x": [], "y": []})
                prov_trace_pts[p]["x"].append(t["x"])
                prov_trace_pts[p]["y"].append(t["y"])

            # Use LLM-only traces for grid bounds (exclude strategy outliers)
            llm_tx = [t["x"] for t in traces if _llm(t.get("model_key", ""))]
            llm_ty = [t["y"] for t in traces if _llm(t.get("model_key", ""))]
            if llm_tx:
                pad = 0.15
                xmin = np.percentile(llm_tx, 1) - pad
                xmax = np.percentile(llm_tx, 99) + pad
                ymin = np.percentile(llm_ty, 1) - pad
                ymax = np.percentile(llm_ty, 99) + pad
            else:
                all_x = [t["x"] for t in traces]
                all_y = [t["y"] for t in traces]
                pad = 0.5
                xmin, xmax = min(all_x) - pad, max(all_x) + pad
                ymin, ymax = min(all_y) - pad, max(all_y) + pad
            xx, yy = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
            positions = np.vstack([xx.ravel(), yy.ravel()])

            for p in PROV_ORDER:
                if p not in prov_trace_pts or len(prov_trace_pts[p]["x"]) < 20:
                    continue
                col = C[p]
                rgb = mcolors.to_rgb(col)
                vals = np.vstack([prov_trace_pts[p]["x"],
                                  prov_trace_pts[p]["y"]])
                try:
                    kde = gaussian_kde(vals, bw_method=0.35)
                    zz = kde(positions).reshape(xx.shape)
                    zz_norm = zz / zz.max() if zz.max() > 0 else zz
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        f"kde6_{p}",
                        [(rgb[0], rgb[1], rgb[2], 0.0),
                         (rgb[0], rgb[1], rgb[2], 0.18)],
                        N=64)
                    ax1.contourf(xx, yy, zz_norm,
                                 levels=np.linspace(0.2, 1.0, 5),
                                 cmap=cmap, zorder=0)
                    ax1.contour(xx, yy, zz_norm,
                                levels=np.linspace(0.4, 0.9, 3),
                                colors=[col], linewidths=0.25,
                                alpha=0.3, zorder=0)
                except Exception:
                    pass

        for item in centroids:
            key = item["model_key"]
            x, y = item["x"], item["y"]
            p = _prov(key)

            if not _llm(key):
                ax1.plot(x, y, "D", color=C["strategy"], ms=3,
                         alpha=0.35, mew=0, zorder=1)
            else:
                col = C.get(p, C["strategy"])
                ax1.plot(x, y, "o", color=col, ms=5.5,
                         mec="white", mew=0.4, zorder=3)
                prov_pts.setdefault(p, []).append((x, y))

        # Zoom into LLM density region (exclude extreme strategy outliers)
        llm_cx = [item["x"] for item in centroids if _llm(item["model_key"])]
        llm_cy = [item["y"] for item in centroids if _llm(item["model_key"])]
        if llm_cx:
            margin = 0.18
            ax1.set_xlim(min(llm_cx) - margin, max(llm_cx) + margin)
            ax1.set_ylim(min(llm_cy) - margin, max(llm_cy) + margin)

        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")

        # Combined legend (providers + strategies)
        provs_present = [p for p in PROV_ORDER if p in prov_pts]
        hs = [ax1.plot([], [], "o", color=C[p], ms=3.5, ls="none",
                       label=PROV_LABEL[p])[0] for p in provs_present]
        hs.append(ax1.plot([], [], "D", color=C["strategy"], ms=3, ls="none",
                           alpha=0.5, label="Strategies")[0])
        ax1.legend(handles=hs, loc="lower left", fontsize=5, ncol=2)

        # ── b: Silhouette ──
        _lab(ax2, "b")

        sil = hodo["summary"]["provider_separation"]["per_model_silhouette"]
        llm_sil = {k: v for k, v in sil.items() if _llm(k)}
        sk = _sort(list(llm_sil.keys()))
        n = len(sk)

        vals   = [llm_sil[k] for k in sk]
        colors = [_c(k) for k in sk]

        ax2.barh(range(n), vals, color=colors, height=0.72,
                 edgecolor="white", linewidth=0.3, zorder=2)
        ax2.axvline(0, color="#444444", lw=0.4, zorder=1)

        ax2.set_yticks(range(n))
        ax2.set_yticklabels([_s(k) for k in sk], fontsize=5)
        ax2.set_xlabel("Silhouette score")
        ax2.invert_yaxis()

        for i, k in enumerate(sk):
            ax2.get_yticklabels()[i].set_color(_c(k))

        # Provider group separators
        prev = None
        for i, k in enumerate(sk):
            p = _prov(k)
            if prev is not None and p != prev:
                ax2.axhline(i - 0.5, color="#CCCCCC", lw=0.4)
            prev = p

        fig.tight_layout(w_pad=2.0)
        _save(fig, "fig6_provider_clustering")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 1 — Radar Charts for All 25 Models
# ══════════════════════════════════════════════════════════════════════

def ed_fig1_radar_charts():
    """5x5 grid of small polar radar charts, one per model, grouped by
    provider.  Each chart shows 8 behavioural dimensions.
    """
    print("ED 1   Radar charts (all models)")

    models = _load("models.json")["models"]
    coop   = _load("manuscript_numbers.json")["cooperation_by_model"]
    keys   = _sort([k for k in models if _llm(k)], coop)

    nm = len(keys)
    ncols = 5
    nrows = math.ceil(nm / ncols)

    angles = np.linspace(0, 2 * np.pi, len(CAT_ORDER), endpoint=False).tolist()
    angles += angles[:1]

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(DOUBLE, 1.6 * nrows))

        for idx, mk in enumerate(keys):
            ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)
            radar = models[mk].get("radar", {})
            vals = [radar.get(c, {}).get("value", 0) * 100 for c in CAT_ORDER]
            vals += vals[:1]
            col = _c(mk)

            ax.fill(angles, vals, color=col, alpha=0.15)
            ax.plot(angles, vals, "o-", color=col, lw=0.8, ms=1.8,
                    mec="white", mew=0.15, zorder=3)

            ax.set_title(_s(mk), fontsize=5, fontweight="bold", color=col,
                         pad=6)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([c[:3].upper() for c in CAT_ORDER], fontsize=3)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75])
            ax.set_yticklabels([], fontsize=0)
            ax.grid(color="#E0E0E0", lw=0.2)
            ax.spines["polar"].set_visible(False)
            ax.tick_params(axis="x", pad=1)

        # Remove empty subplots
        for idx in range(nm, nrows * ncols):
            fig.add_subplot(nrows, ncols, idx + 1).set_visible(False)

        fig.subplots_adjust(hspace=0.55, wspace=0.35,
                            left=0.03, right=0.97, top=0.95, bottom=0.03)
        _save(fig, "ed_fig1_radar_charts")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 2 — Behavioural Embedding Space
# ══════════════════════════════════════════════════════════════════════

def ed_fig2_behavioral_space():
    """UMAP scatter of 10,000 individual trace embeddings with per-provider
    KDE density contour fills creating a behavioural landscape, plus centroid
    markers overlaid.
    """
    print("ED 2   Behavioral embedding space")

    hodo = _load("hodoscope.json")
    traces   = hodo.get("trace_umap", [])
    centroids = hodo.get("centroid_umap", [])

    if not traces:
        print("       (no trace_umap data)")
        return

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE + 1, SINGLE + 0.8))

        # Group traces by provider
        prov_traces = {}
        for t in traces:
            mk = t.get("model_key", "")
            p = _prov(mk) if mk else t.get("provider", "").lower()
            prov_traces.setdefault(p, {"x": [], "y": []})
            prov_traces[p]["x"].append(t["x"])
            prov_traces[p]["y"].append(t["y"])

        # KDE density contour fills per provider (terrain landscape)
        if HAS_SCIPY:
            # Compute grid bounds from all traces
            all_x = [t["x"] for t in traces]
            all_y = [t["y"] for t in traces]
            pad = 0.5
            xmin, xmax = min(all_x) - pad, max(all_x) + pad
            ymin, ymax = min(all_y) - pad, max(all_y) + pad
            xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
            positions = np.vstack([xx.ravel(), yy.ravel()])

            for p in PROV_ORDER:
                if p not in prov_traces or len(prov_traces[p]["x"]) < 20:
                    continue
                col = C[p]
                rgb = mcolors.to_rgb(col)
                vals = np.vstack([prov_traces[p]["x"], prov_traces[p]["y"]])
                try:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    zz = kde(positions).reshape(xx.shape)
                    # Normalise to [0, 1] for consistent contouring
                    zz_norm = zz / zz.max() if zz.max() > 0 else zz
                    # Custom colormap: transparent → provider colour
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        f"kde_{p}",
                        [(rgb[0], rgb[1], rgb[2], 0.0),
                         (rgb[0], rgb[1], rgb[2], 0.25)],
                        N=64)
                    ax.contourf(xx, yy, zz_norm,
                                levels=np.linspace(0.15, 1.0, 6),
                                cmap=cmap, zorder=0)
                    # Thin contour lines for definition
                    ax.contour(xx, yy, zz_norm,
                               levels=np.linspace(0.3, 0.9, 4),
                               colors=[col], linewidths=0.3,
                               alpha=0.35, zorder=0)
                except Exception:
                    pass

        # Background traces (small, transparent)
        for t in traces:
            mk = t.get("model_key", "")
            p = _prov(mk) if mk else t.get("provider", "").lower()
            col = C.get(p, C["strategy"])
            ax.plot(t["x"], t["y"], ".", color=col, ms=0.8,
                    alpha=0.12, rasterized=True, zorder=1)

        # Centroids
        for item in centroids:
            key = item["model_key"]
            x, y = item["x"], item["y"]
            p = _prov(key)
            if not _llm(key):
                ax.plot(x, y, "D", color=C["strategy"], ms=4,
                        mec="#444444", mew=0.3, alpha=0.6, zorder=4)
                ax.annotate(_s(key), (x, y), xytext=(3, 3),
                            textcoords="offset points", fontsize=3,
                            color="#888888", zorder=5)
            else:
                col = C.get(p, C["strategy"])
                ax.plot(x, y, "o", color=col, ms=5.5,
                        mec="white", mew=0.5, zorder=5)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Legend
        provs_used = list(dict.fromkeys(
            _prov(item["model_key"]) for item in centroids if _llm(item["model_key"])
        ))
        hs = [ax.plot([], [], "o", color=C.get(p, C["strategy"]), ms=3.5,
                       ls="none", label=PROV_LABEL.get(p, p))[0]
              for p in provs_used if p in C]
        hs.append(ax.plot([], [], "D", color=C["strategy"], ms=3, ls="none",
                          alpha=0.6, label="Strategies")[0])
        ax.legend(handles=hs, loc="lower left", fontsize=5, ncol=2)

        fig.tight_layout()
        _save(fig, "ed_fig2_behavioral_space")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 3 — Hierarchical Clustering Dendrogram
# ══════════════════════════════════════════════════════════════════════

def ed_fig3_dendrogram():
    """Average-linkage dendrogram from centroid distances, with
    provider-coloured leaf labels.
    """
    print("ED 3   Dendrogram")

    if not HAS_SCIPY:
        print("       (scipy not available)")
        return

    hodo = _load("hodoscope.json")
    cd   = hodo.get("centroid_distances", [])
    if not cd:
        print("       (no centroid_distances data)")
        return

    # Build symmetric distance matrix
    # Each item is a dict row: {"": model_key, model_key_1: dist, ...}
    row_keys = []
    for row in cd:
        rk = row.get("", row.get("model_key", ""))
        row_keys.append(rk)

    n = len(row_keys)
    dist_mat = np.zeros((n, n))
    for i, row in enumerate(cd):
        for j, k in enumerate(row_keys):
            dist_mat[i, j] = float(row.get(k, 0))

    # Symmetrise
    dist_mat = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(dist_mat, 0)

    # Convert to condensed form for linkage
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(DOUBLE, 4.5))

        dend = dendrogram(Z, labels=row_keys, ax=ax,
                          leaf_rotation=90, leaf_font_size=5,
                          color_threshold=0, above_threshold_color="#999999")

        # Colour leaf labels by provider
        xlabels = ax.get_xticklabels()
        for lbl in xlabels:
            mk = lbl.get_text()
            lbl.set_text(_s(mk))
            lbl.set_color(_c(mk))
            lbl.set_fontsize(4.5)

        ax.set_xticklabels([lbl.get_text() for lbl in xlabels])
        # Recolor after text update
        for lbl in ax.get_xticklabels():
            # Match back to original key
            short_name = lbl.get_text()
            for k in row_keys:
                if _s(k) == short_name:
                    lbl.set_color(_c(k))
                    break

        ax.set_ylabel("Distance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _save(fig, "ed_fig3_dendrogram")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 4 — Jensen-Shannon Distance Matrices
# ══════════════════════════════════════════════════════════════════════

def ed_fig4_jsd_matrices():
    """Panel a: 25x25 aggregate JSD heatmap.  Panel b: comparison of
    cooperation-game vs competition-game JSD sub-matrices.
    """
    print("ED 4   JSD matrices")

    hodo = _load("hodoscope.json")
    jsd_raw  = hodo.get("aggregate_jsd", [])
    jsd_game = hodo.get("jsd_per_game", {})

    if not jsd_raw:
        print("       (no aggregate_jsd data)")
        return

    # Build LLM-only JSD matrix from aggregate
    all_keys = []
    for row in jsd_raw:
        rk = row.get("", row.get("model_key", ""))
        all_keys.append(rk)

    llm_keys = [k for k in all_keys if _llm(k)]
    llm_sorted = _sort(llm_keys)
    n = len(llm_sorted)
    ki = {k: i for i, k in enumerate(all_keys)}

    mat_a = np.zeros((n, n))
    for i, m1 in enumerate(llm_sorted):
        row = jsd_raw[ki[m1]]
        for j, m2 in enumerate(llm_sorted):
            mat_a[i, j] = float(row.get(m2, 0))

    # Identify cooperation and competition games for panel b
    coop_games = [g for g in jsd_game if any(x in g.lower()
                  for x in ("pd_", "prisoner", "stag", "trust", "public_good"))]
    comp_games = [g for g in jsd_game if any(x in g.lower()
                  for x in ("auction", "hawk", "chicken", "beauty", "centipede"))]

    with plt.rc_context(RC):
        if coop_games and comp_games:
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(DOUBLE, 0.28 * n + 0.6),
                gridspec_kw={"width_ratios": [1.2, 1, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(ONEHALF, 0.28 * n + 0.6))
            ax2 = ax3 = None

        # ── Panel a: Full JSD heatmap ──
        _lab(ax1, "a")
        vmax_a = np.nanmax(mat_a) if np.nanmax(mat_a) > 0 else 1
        im1 = ax1.imshow(mat_a, aspect="auto", cmap="YlOrRd",
                         vmin=0, vmax=vmax_a, interpolation="nearest")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = mat_a[i, j]
                tc = "white" if v > vmax_a * 0.65 else "black"
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=3 if n > 20 else 3.5, color=tc)

        _white_grid(ax1, n, n, lw=0.3)
        _hide_spines(ax1)

        ax1.set_xticks(range(n))
        ax1.set_xticklabels([_s(m) for m in llm_sorted], rotation=55,
                            ha="right", fontsize=4)
        ax1.set_yticks(range(n))
        ax1.set_yticklabels([_s(m) for m in llm_sorted], fontsize=4)

        for i, m in enumerate(llm_sorted):
            ax1.get_xticklabels()[i].set_color(_c(m))
            ax1.get_yticklabels()[i].set_color(_c(m))

        # Provider group separators
        prev = None
        for i, m in enumerate(llm_sorted):
            p = _prov(m)
            if prev is not None and p != prev:
                ax1.axhline(i - 0.5, color="#444444", lw=0.6)
                ax1.axvline(i - 0.5, color="#444444", lw=0.6)
            prev = p

        cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.5, aspect=18, pad=0.02)
        cbar1.set_label("JSD", fontsize=5)
        cbar1.ax.tick_params(labelsize=4)
        cbar1.outline.set_visible(False)

        # ── Panels b,c: cooperation vs competition JSD ──
        if ax2 is not None and coop_games and comp_games:
            for ax_sub, games, title, lab in [
                (ax2, coop_games, "Cooperation games", "b"),
                (ax3, comp_games, "Competition games", "c"),
            ]:
                _lab(ax_sub, lab)
                # Average JSD across selected games
                mat_sub = np.zeros((n, n))
                count = 0
                for game in games:
                    gdata = jsd_game.get(game, [])
                    if not gdata:
                        continue
                    g_keys = []
                    for row in gdata:
                        rk = row.get("", row.get("model_key", ""))
                        g_keys.append(rk)
                    g_ki = {k: i for i, k in enumerate(g_keys)}
                    for i, m1 in enumerate(llm_sorted):
                        if m1 not in g_ki:
                            continue
                        row = gdata[g_ki[m1]]
                        for j, m2 in enumerate(llm_sorted):
                            v = float(row.get(m2, 0))
                            mat_sub[i, j] += v
                    count += 1
                if count > 0:
                    mat_sub /= count

                vmax_s = np.nanmax(mat_sub) if np.nanmax(mat_sub) > 0 else 1
                im_s = ax_sub.imshow(mat_sub, aspect="auto", cmap="YlOrRd",
                                     vmin=0, vmax=vmax_s,
                                     interpolation="nearest")

                _white_grid(ax_sub, n, n, lw=0.2)
                _hide_spines(ax_sub)

                ax_sub.set_xticks(range(n))
                ax_sub.set_xticklabels([_s(m) for m in llm_sorted],
                                       rotation=55, ha="right", fontsize=3)
                ax_sub.set_yticks(range(n))
                ax_sub.set_yticklabels([_s(m) for m in llm_sorted],
                                       fontsize=3)
                ax_sub.set_title(title, fontsize=6, fontweight="bold")

                for i, m in enumerate(llm_sorted):
                    ax_sub.get_xticklabels()[i].set_color(_c(m))
                    ax_sub.get_yticklabels()[i].set_color(_c(m))

        fig.tight_layout(w_pad=1.0)
        _save(fig, "ed_fig4_jsd_matrices")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 5 — Round-by-Round Choice Timelines
# ══════════════════════════════════════════════════════════════════════

def ed_fig5_round_timelines():
    """Connected-dot round-by-round cooperation trajectories for
    representative model archetypes.
    """
    print("ED 5   Round timelines")

    aa = _load("advanced_analytics.json")
    bm = aa["endgame"]["by_model"]

    if not bm:
        print("       (no data)")
        return

    # Select representative models across archetypes
    REPS = {
        "claude-haiku-4.5-thinking": "Terminal cooperator",
        "claude-opus-4.6":           "Sustained cooperator",
        "gpt-4o-mini":               "Gradual decliner",
        "gemini-3-flash":            "Late-round defector",
        "gpt-5-nano":                "Early defector",
        "deepseek-r1":               "Stable mid-range",
    }
    reps = {k: v for k, v in REPS.items() if k in bm}

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE + 0.5, 3.3))

        # Background models (thin grey)
        for k, d in bm.items():
            if k in reps:
                continue
            r = d["rounds"]
            y = [v * 100 for v in d["cooperation_rate"]]
            ax.plot(r, y, "-", color="#E0E0E0", lw=0.3, alpha=0.5, zorder=1)

        # Highlight representatives
        markers = ["o", "s", "D", "^", "v", "P"]
        for i, (k, lbl) in enumerate(reps.items()):
            d = bm[k]
            r = d["rounds"]
            y = [v * 100 for v in d["cooperation_rate"]]
            col = _c(k)
            mrk = markers[i % len(markers)]
            ax.plot(r, y, f"{mrk}-", color=col, ms=4, mec="white",
                    mew=0.3, lw=1.2, zorder=3 + i, label=f"{_s(k)} ({lbl})")

        # Shade final round
        ax.axvspan(9.5, 10.5, color="#FEE2E2", alpha=0.25, zorder=0)
        ax.text(10, 97, "Final\nround", ha="center", va="top",
                fontsize=4.5, color="#B91C1C", style="italic")

        ax.set_xlabel("Round")
        ax.set_ylabel("Cooperation rate (%)")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-3, 105)
        ax.legend(loc="upper right", fontsize=4.5, ncol=1)

        fig.tight_layout()
        _save(fig, "ed_fig5_round_timelines")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 6 — Factor Analysis of Behavioural Dimensions
# ══════════════════════════════════════════════════════════════════════

def ed_fig6_factor_loadings():
    """PCA-based factor loadings of per-game behavioural metrics.
    Panel a: loading scatter (PC1 vs PC2).  Panel b: variance explained.
    """
    print("ED 6   Factor loadings")

    sigs = _load("hodoscope.json").get("behavioral_signatures", [])
    if not sigs:
        print("       (no data)")
        return

    # Collect LLM-only per-game numeric features
    llm_sigs = [s for s in sigs if _llm(s.get("model_key", ""))]
    if len(llm_sigs) < 3:
        print("       (too few models)")
        return

    # Identify per-game numeric features (exclude model_key, lex_, trace_)
    all_keys = set()
    for s in llm_sigs:
        all_keys.update(s.keys())

    feature_keys = sorted([k for k in all_keys
                           if "__" in k
                           and not k.startswith("lex_")
                           and k != "model_key"
                           and any(m in k for m in
                                   ("entropy", "consistency", "first_round_rate",
                                    "endgame_shift", "dominant_rate"))])

    if len(feature_keys) < 4:
        print(f"       (only {len(feature_keys)} features found)")
        return

    # Build data matrix (models x features)
    n_mod = len(llm_sigs)
    n_feat = len(feature_keys)
    X = np.zeros((n_mod, n_feat))
    mod_keys = []
    for i, s in enumerate(llm_sigs):
        mod_keys.append(s["model_key"])
        for j, fk in enumerate(feature_keys):
            v = s.get(fk, 0)
            X[i, j] = float(v) if v is not None else 0

    # Standardise columns
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    loadings = Vt[:2, :].T * S[:2] / np.sqrt(n_mod - 1)
    var_exp = (S ** 2) / (S ** 2).sum()

    # Pretty feature labels: "game__metric" -> "Game (metric)"
    def _feat_label(fk):
        parts = fk.split("__", 1)
        if len(parts) == 2:
            game = parts[0].replace("_", " ")
            if len(game) > 18:
                game = game[:17] + "."
            metric = parts[1].replace("_", " ")
            return f"{game} ({metric})"
        return fk

    feat_labels = [_feat_label(fk) for fk in feature_keys]

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE, 4.0),
            gridspec_kw={"width_ratios": [2, 1]})

        # ── Panel a: Loading scatter ──
        _lab(ax1, "a")

        # Colour features by game category
        games_data = _load("games.json") if (DATA / "games.json").exists() else {}

        for j in range(n_feat):
            # Try to determine game category for colouring
            game_name = feature_keys[j].split("__")[0]
            cat = "other"
            if isinstance(games_data, dict):
                ginfo = games_data.get(game_name, {})
                if isinstance(ginfo, dict):
                    cat = ginfo.get("category", "other")

            cat_colors = {
                "cooperation":  "#0072B2",
                "coordination": "#009E73",
                "trust":        "#E69F00",
                "fairness":     "#CC79A7",
                "depth":        "#56B4E9",
                "competition":  "#D55E00",
                "negotiation":  "#332288",
                "risk":         "#F0E442",
            }
            col = cat_colors.get(cat, "#999999")
            ax1.plot(loadings[j, 0], loadings[j, 1], "o", color=col,
                     ms=3.5, mec="white", mew=0.2, zorder=3)

        # Label top-loading features
        mag = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        top_idx = np.argsort(mag)[-8:]
        for j in top_idx:
            ax1.annotate(feat_labels[j], (loadings[j, 0], loadings[j, 1]),
                         xytext=(3, 2), textcoords="offset points",
                         fontsize=3.5, color="#555555", zorder=4)

        # Reference lines
        ax1.axhline(0, color="#DDDDDD", lw=0.3, zorder=0)
        ax1.axvline(0, color="#DDDDDD", lw=0.3, zorder=0)

        ax1.set_xlabel(f"Factor 1 ({var_exp[0]*100:.1f}% var.)")
        ax1.set_ylabel(f"Factor 2 ({var_exp[1]*100:.1f}% var.)")

        # ── Panel b: Variance scree ──
        _lab(ax2, "b")

        n_show = min(10, len(var_exp))
        ax2.bar(range(1, n_show + 1), var_exp[:n_show] * 100,
                color="#0072B2", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax2.plot(range(1, n_show + 1), np.cumsum(var_exp[:n_show]) * 100,
                 "o-", color="#D55E00", ms=3, lw=0.8, mec="white", mew=0.2)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Variance explained (%)")
        ax2.set_xticks(range(1, n_show + 1))

        fig.tight_layout(w_pad=2.0)
        _save(fig, "ed_fig6_factor_loadings")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 7 — Lexical Features
# ══════════════════════════════════════════════════════════════════════

def ed_fig7_lexical_features():
    """Horizontal bar chart of strategy-relevant term frequencies,
    one sub-panel per term.
    """
    print("ED 7   Lexical features")

    sigs = _load("hodoscope.json").get("behavioral_signatures", [])
    if not sigs:
        print("       (no data)")
        return

    terms = {
        "lex_cooperation": "cooperate",
        "lex_defection":   "defect",
        "lex_fairness":    "fairness",
        "lex_trust":       "trust",
        "lex_equilibrium": "equilibrium",
        "lex_exploit":     "exploit",
    }

    md = {}
    for item in sigs:
        k = item.get("model_key", "")
        if not _llm(k):
            continue
        feats = {lab: item.get(lk, 0) for lk, lab in terms.items()
                 if lk in item}
        if feats:
            md[k] = feats

    if not md:
        print("       (no data)")
        return

    # Keep terms with at least some nonzero values
    active = [t for t in terms.values()
              if any(md[k].get(t, 0) > 0 for k in md)]
    if not active:
        print("       (no data)")
        return

    sk = _sort(list(md.keys()))
    nm = len(sk)
    nt = len(active)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, nt, figsize=(DOUBLE, 0.19 * nm + 0.8),
                                 sharey=True)
        if nt == 1:
            axes = [axes]

        for j, term in enumerate(active):
            ax = axes[j]
            vs = [md[k].get(term, 0) for k in sk]
            cs = [_c(k) for k in sk]
            ax.barh(range(nm), vs, color=cs, height=0.72,
                    edgecolor="white", linewidth=0.2)
            ax.set_title(f'"{term}"', fontsize=6, style="italic")

            if j == 0:
                ax.set_yticks(range(nm))
                ax.set_yticklabels([_s(k) for k in sk], fontsize=4.5)
                for i, k in enumerate(sk):
                    ax.get_yticklabels()[i].set_color(_c(k))
            else:
                ax.set_yticks([])
            ax.invert_yaxis()

        fig.supxlabel("Frequency per 1 000 words", fontsize=6, y=0.01)
        fig.tight_layout()
        _save(fig, "ed_fig7_lexical_features")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 8 — Cross-play Cooperation Matrix (annotated heatmap)
# ══════════════════════════════════════════════════════════════════════

def ed_fig8_crossplay_matrix():
    """Annotated heatmap (Akata Fig 3a style): integer cooperation
    rates in every cell, YlGnBu colormap, provider-coloured labels.
    """
    print("ED 8   Cross-play matrix")

    cp = _load("crossplay.json")

    # Pick a canonical cooperation game
    for cand in ("pd_canonical", "pd_medium", "pd_mild", "pd_harsh"):
        if cand in cp:
            gk = cand
            break
    else:
        gk = list(cp.keys())[0]

    pairs = cp[gk]

    # Collect models
    mset = set()
    for pk, pd in pairs.items():
        ms = pd.get("models", [])
        if len(ms) == 2:
            mset.update(ms)

    mlist = _sort(list(mset))
    n = len(mlist)
    if n == 0:
        print("       (no data)")
        return

    mat = np.full((n, n), np.nan)
    mi  = {m: i for i, m in enumerate(mlist)}

    for pk, pd in pairs.items():
        ms = pd.get("models", [])
        if len(ms) != 2:
            continue
        m1, m2 = ms
        if m1 not in mi or m2 not in mi:
            continue
        trials = pd.get("trials", [])
        rates  = []
        for t in trials:
            if isinstance(t, dict):
                r = t.get("cooperation_rate",
                          t.get("p0_cooperation", None))
                if r is not None:
                    rates.append(float(r))
        if rates:
            mat[mi[m1], mi[m2]] = np.mean(rates) * 100

    with plt.rc_context(RC):
        side = min(DOUBLE, 0.42 * n + 1.0)
        fig, ax = plt.subplots(figsize=(side, side * 0.9))

        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=100, interpolation="nearest")

        # Numbers in cells
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                tc = "white" if v > 60 else "black"
                fs = 4 if n > 18 else 5
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=fs, color=tc)

        _white_grid(ax, n, n, lw=0.3)
        _hide_spines(ax)

        ax.set_xticks(range(n))
        ax.set_xticklabels([_s(m) for m in mlist], rotation=55,
                           ha="right", fontsize=4.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([_s(m) for m in mlist], fontsize=4.5)

        for i, m in enumerate(mlist):
            ax.get_xticklabels()[i].set_color(_c(m))
            ax.get_yticklabels()[i].set_color(_c(m))

        ax.set_xlabel("Player 2", fontsize=7)
        ax.set_ylabel("Player 1", fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=22, pad=0.02)
        cbar.set_label("Cooperation rate (%)", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        _save(fig, "ed_fig8_crossplay_matrix")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 10 — Coverage Matrix
# ══════════════════════════════════════════════════════════════════════

def ed_fig10_coverage():
    """Model x game trial-count heatmap (Blues colormap)."""
    print("ED 10  Coverage matrix")

    cov    = _load("coverage_matrix.json")
    models = _load("models.json")["models"]

    mk = _sort([k for k in models if _llm(k)])
    gk = sorted(cov.keys())
    nm, ng = len(mk), len(gk)

    mat = np.zeros((nm, ng))
    mi  = {m: i for i, m in enumerate(mk)}

    for gi, game in enumerate(gk):
        entries = cov[game]
        for matchup_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            model  = entry.get("model", "")
            trials = entry.get("trials", 0)
            if model in mi:
                mat[mi[model], gi] += trials

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(DOUBLE, 0.21 * nm + 1.0))

        vmax = max(mat.max(), 1)
        im = ax.imshow(mat, aspect="auto", cmap="Blues",
                       vmin=0, vmax=vmax, interpolation="nearest")

        _white_grid(ax, nm, ng, lw=0.3)
        _hide_spines(ax)

        # Format game names
        def _game_label(g):
            g = g.replace("_", " ")
            if len(g) > 14:
                g = g[:13] + "."
            return g.title()

        ax.set_xticks(range(ng))
        ax.set_xticklabels([_game_label(g) for g in gk],
                           rotation=90, ha="center", fontsize=3.5)
        ax.set_yticks(range(nm))
        ax.set_yticklabels([_s(m) for m in mk], fontsize=5)

        for i, m in enumerate(mk):
            ax.get_yticklabels()[i].set_color(_c(m))

        cbar = fig.colorbar(im, ax=ax, shrink=0.45, aspect=20, pad=0.02)
        cbar.set_label("Trial count", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        _save(fig, "ed_fig10_coverage")


# ══════════════════════════════════════════════════════════════════════
# ED FIGURE 9 — Cost-performance analysis
# ══════════════════════════════════════════════════════════════════════

def ed_fig9_cost_performance():
    """Three-panel: (a) cost vs cooperation, (b) cost vs depth,
    (c) total cost by provider (horizontal bars).
    """
    print("ED 9   Cost-performance")

    cost = _load("advanced_analytics.json")["cost"]
    models_data = _load("models.json")["models"]
    coop = _load("manuscript_numbers.json")["cooperation_by_model"]

    by_model = cost["by_model"]
    by_prov  = cost["by_provider"]

    with plt.rc_context(RC):
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(DOUBLE, 2.8),
            gridspec_kw={"width_ratios": [1, 1, 0.8]})

        # ── a: cost vs cooperation ──
        _lab(ax1, "a")
        for k, d in by_model.items():
            avg_cost = d["avg_cost"]
            cr = coop[k]["mean"] * 100 if k in coop else 0
            col = _c(k)
            ax1.plot(avg_cost, cr, "o", color=col, ms=4.5,
                     mec="white", mew=0.3, zorder=3)

        ax1.set_xscale("log")
        ax1.set_xlabel("Per-trial cost (USD, log)")
        ax1.set_ylabel("Cooperation rate (%)")
        ax1.set_ylim(-3, 80)

        # ── b: cost vs depth ──
        _lab(ax2, "b")
        for k, d in by_model.items():
            avg_cost = d["avg_cost"]
            radar = models_data.get(k, {}).get("radar", {})
            depth = radar.get("depth", {}).get("value", 0) * 100
            col = _c(k)
            ax2.plot(avg_cost, depth, "o", color=col, ms=4.5,
                     mec="white", mew=0.3, zorder=3)

        ax2.set_xscale("log")
        ax2.set_xlabel("Per-trial cost (USD, log)")
        ax2.set_ylabel("Strategic depth (%)")

        # ── c: total cost by provider ──
        _lab(ax3, "c")
        provs = sorted(by_prov.keys(),
                       key=lambda p: by_prov[p]["total_cost"],
                       reverse=True)
        ys   = range(len(provs))
        vals = [by_prov[p]["total_cost"] for p in provs]
        cols = [C.get(p, C["strategy"]) for p in provs]

        ax3.barh(list(ys), vals, color=cols, height=0.6,
                 edgecolor="white", linewidth=0.3)
        ax3.set_yticks(list(ys))
        ax3.set_yticklabels([PROV_LABEL.get(p, p) for p in provs],
                            fontsize=5.5)
        ax3.set_xlabel("Total cost (USD)")
        ax3.invert_yaxis()

        for i, p in enumerate(provs):
            ax3.get_yticklabels()[i].set_color(C.get(p, C["strategy"]))
            ax3.text(vals[i] + 5, i, f"${vals[i]:.0f}",
                     va="center", fontsize=5, color="#666666")

        _prov_legend(ax1, loc="upper left", ncol=2, ms=3)

        fig.tight_layout(w_pad=1.5)
        _save(fig, "ed_fig9_cost_performance")


# ══════════════════════════════════════════════════════════════════════
# Supplementary Figures S1-S4
# ══════════════════════════════════════════════════════════════════════

def _load_profiles():
    """Load behavioral_profiles.csv as list of dicts (cached)."""
    if not hasattr(_load_profiles, "_cache"):
        rows = []
        with open(PROC / "behavioral_profiles.csv", "r",
                  encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        _load_profiles._cache = rows
    return _load_profiles._cache


COOP_GAMES = [
    "pd_canonical", "pd_harsh", "pd_medium", "pd_mild",
    "pg_low_mpcr", "pg_med_mpcr", "pg_high_mpcr",
    "commons_dilemma", "diners_dilemma", "el_farol_bar",
]

COOP_GAME_SHORT = {
    "pd_canonical": "PD Canon.",
    "pd_harsh": "PD Harsh",
    "pd_medium": "PD Medium",
    "pd_mild": "PD Mild",
    "pg_low_mpcr": "PG Low",
    "pg_med_mpcr": "PG Med",
    "pg_high_mpcr": "PG High",
    "commons_dilemma": "Commons",
    "diners_dilemma": "Diners",
    "el_farol_bar": "El Farol",
}


def sup_fig1_cooperation_distributions():
    """Box plots of cooperation rate across 10 cooperation games, by model."""
    print("S1   Cooperation rate distributions by game category")

    rows = _load_profiles()
    models_j = _load("models.json")["models"]

    # Aggregate cooperation rate per (model, game) — mean across opponents
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["game_category"] != "cooperation":
            continue
        cr = r.get("cooperation_rate", "")
        if cr == "" or cr == "None":
            continue
        agg[r["model_key"]][r["game_id"]].append(float(cr))

    # Compute mean per (model, game)
    model_game = {}
    for mk in agg:
        model_game[mk] = {}
        for gid in COOP_GAMES:
            vals = agg[mk].get(gid, [])
            model_game[mk][gid] = np.mean(vals) * 100 if vals else np.nan

    # Sort models by provider then mean cooperation
    coop_data = _load("manuscript_numbers.json").get("cooperation_by_model", {})
    mk_list = _sort([k for k in model_game if _llm(k)], coop=coop_data)

    # Build data for box plots — one row per model, 10 game values
    data_by_model = []
    labels = []
    colors = []
    for mk in mk_list:
        vals = [model_game[mk].get(g, np.nan) for g in COOP_GAMES]
        data_by_model.append(vals)
        labels.append(_s(mk))
        colors.append(_c(mk))

    n = len(mk_list)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(DOUBLE, 0.3 * n + 1.2))

        # Horizontal box plots
        bp = ax.boxplot(data_by_model, vert=False, widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=2, alpha=0.4),
                        medianprops=dict(color="white", linewidth=1),
                        whiskerprops=dict(linewidth=0.6),
                        capprops=dict(linewidth=0.6))

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
            patch.set_edgecolor(colors[i])

        ax.set_yticks(range(1, n + 1))
        ax.set_yticklabels(labels, fontsize=5.5)
        for i, lab in enumerate(ax.get_yticklabels()):
            lab.set_color(colors[i])

        ax.set_xlabel("Cooperation rate (%)")
        ax.set_xlim(-5, 105)
        ax.invert_yaxis()
        _hide_spines(ax)

        # Vertical separator between providers
        prev_prov = None
        for i, mk in enumerate(mk_list):
            p = _prov(mk)
            if prev_prov and p != prev_prov:
                ax.axhline(i + 0.5, color="#CCCCCC", linewidth=0.4,
                           linestyle="--")
            prev_prov = p

        fig.tight_layout()
        _save(fig, "sup_fig1_cooperation_distributions")


def sup_fig2_strategy_response():
    """Heatmap of cooperation rates against 13 deterministic strategies."""
    print("S2   Strategy-play response profiles")

    rows = _load_profiles()
    models_j = _load("models.json")["models"]

    # Filter: pd_canonical, model_vs_strategy
    strat_data = defaultdict(lambda: defaultdict(list))
    strategies_seen = set()
    for r in rows:
        if r["game_id"] != "pd_canonical":
            continue
        if r["matchup_type"] != "model_vs_strategy":
            continue
        cr = r.get("cooperation_rate", "")
        if cr == "" or cr == "None":
            continue
        strat_data[r["model_key"]][r["opponent"]].append(float(cr))
        strategies_seen.add(r["opponent"])

    # Strategy order (from cooperative to hostile)
    strat_order = [
        "always_cooperate", "tit_for_tat", "tit_for_two_tats", "pavlov",
        "grim_trigger", "noise_10", "noise_20", "defect_once",
        "false_defector", "reverse_tit_for_tat", "hard_tit_for_tat",
        "suspicious_tit_for_tat", "always_defect",
    ]
    strat_order = [s for s in strat_order if s in strategies_seen]

    strat_short = {
        "always_cooperate": "AllC",
        "always_defect": "AllD",
        "tit_for_tat": "TFT",
        "tit_for_two_tats": "TF2T",
        "suspicious_tit_for_tat": "STFT",
        "reverse_tit_for_tat": "RTFT",
        "hard_tit_for_tat": "HTFT",
        "grim_trigger": "Grim",
        "pavlov": "Pavlov",
        "false_defector": "FalseD",
        "defect_once": "D-once",
        "noise_10": "Noise10",
        "noise_20": "Noise20",
    }

    coop_data = _load("manuscript_numbers.json").get("cooperation_by_model", {})
    mk_list = _sort([k for k in strat_data if _llm(k)], coop=coop_data)

    nm, ns = len(mk_list), len(strat_order)
    mat = np.full((nm, ns), np.nan)
    for i, mk in enumerate(mk_list):
        for j, st in enumerate(strat_order):
            vals = strat_data[mk].get(st, [])
            if vals:
                mat[i, j] = np.mean(vals) * 100

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(ONEHALF, 0.25 * nm + 1.2))

        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=100, interpolation="nearest")

        _white_grid(ax, nm, ns, lw=0.3)

        # Annotate cells
        for i in range(nm):
            for j in range(ns):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.0f}"
                tc = "white" if v > 65 else "#333333"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=4, color=tc)

        ax.set_xticks(range(ns))
        ax.set_xticklabels([strat_short.get(s, s) for s in strat_order],
                           rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(nm))
        ylabels = [_s(mk) for mk in mk_list]
        ax.set_yticklabels(ylabels, fontsize=5.5)
        for i, lab in enumerate(ax.get_yticklabels()):
            lab.set_color(_c(mk_list[i]))

        ax.set_xlabel("Deterministic strategy opponent")
        ax.set_ylabel("")

        cb = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25, pad=0.02)
        cb.set_label("Cooperation rate (%)", fontsize=6)
        cb.ax.tick_params(labelsize=5)

        _hide_spines(ax)
        fig.tight_layout()
        _save(fig, "sup_fig2_strategy_response")


def sup_fig3_crossplay_effects():
    """Self-play vs cross-play cooperation scatter for 16 Design F models."""
    print("S3   Cross-play interaction effects")

    rows = _load_profiles()
    models_j = _load("models.json")["models"]

    # Design F models (those with cross-play data)
    design_f = set()
    for r in rows:
        if r["matchup_type"] == "cross_play" and r["game_category"] == "cooperation":
            design_f.add(r["model_key"])

    # Compute self-play and cross-play cooperation rates
    self_play = defaultdict(list)
    cross_play = defaultdict(list)
    for r in rows:
        if r["game_category"] != "cooperation":
            continue
        mk = r["model_key"]
        if mk not in design_f:
            continue
        cr = r.get("cooperation_rate", "")
        if cr == "" or cr == "None":
            continue
        cr = float(cr)
        if r["matchup_type"] == "self_play":
            self_play[mk].append(cr)
        elif r["matchup_type"] == "cross_play":
            cross_play[mk].append(cr)

    mk_list = sorted(design_f, key=lambda k: (PROV_ORDER.index(_prov(k))
                      if _prov(k) in PROV_ORDER else 99, k))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))

        # Diagonal
        ax.plot([0, 100], [0, 100], "--", color="#CCCCCC", linewidth=0.5,
                zorder=0)

        for mk in mk_list:
            sp = np.mean(self_play.get(mk, [0])) * 100
            cp = np.mean(cross_play.get(mk, [0])) * 100
            ax.scatter(sp, cp, c=_c(mk), s=30, zorder=3,
                       edgecolors="white", linewidths=0.4)
            # Label with short name
            ax.annotate(_s(mk), (sp, cp), fontsize=4, color="#666666",
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel("Self-play cooperation (%)")
        ax.set_ylabel("Cross-play cooperation (%)")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect("equal")

        # Above/below annotations
        ax.text(80, 20, "Less cooperative\nwith others",
                fontsize=5, color="#999999", ha="center", style="italic")
        ax.text(20, 80, "More cooperative\nwith others",
                fontsize=5, color="#999999", ha="center", style="italic")

        _prov_legend(ax, loc="upper left", ncol=2, ms=3)
        _hide_spines(ax)
        fig.tight_layout()
        _save(fig, "sup_fig3_crossplay_effects")


def sup_fig4_thinking_comparison():
    """Paired comparison of thinking vs non-thinking models."""
    print("S4   Thinking model comparison")

    models_j = _load("models.json")["models"]

    # Thinking pairs
    pairs = [
        ("claude-haiku-4.5", "claude-haiku-4.5-thinking"),
        ("gemini-2.5-flash", "gemini-2.5-flash-thinking"),
        ("deepseek-v3",      "deepseek-r1"),
    ]
    pair_labels = [
        "Claude Haiku 4.5",
        "Gemini 2.5 Flash",
        "DeepSeek V3 / R1",
    ]

    # Dimensions to compare
    dims = ["cooperation", "coordination", "fairness", "depth",
            "trust", "competition", "negotiation", "risk"]
    dim_labels = [CAT_LABEL.get(d, d) for d in dims]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.4), sharey=True)

        bar_width = 0.35
        x = np.arange(len(dims))

        for pidx, (base, think) in enumerate(pairs):
            ax = axes[pidx]

            base_vals = []
            think_vals = []
            for d in dims:
                bv = models_j.get(base, {}).get("radar", {}).get(d, {})
                tv = models_j.get(think, {}).get("radar", {}).get(d, {})
                base_vals.append(bv.get("value", 0) * 100)
                think_vals.append(tv.get("value", 0) * 100)

            bars1 = ax.barh(x + bar_width / 2, base_vals,
                            bar_width, color=_c(base), alpha=0.6,
                            edgecolor="white", linewidth=0.3,
                            label="Standard")
            bars2 = ax.barh(x - bar_width / 2, think_vals,
                            bar_width, color=_c(think), alpha=0.9,
                            edgecolor="white", linewidth=0.3,
                            label="Thinking")

            ax.set_title(pair_labels[pidx], fontsize=7, fontweight="bold")
            ax.set_xlim(0, 105)
            ax.set_yticks(x)
            if pidx == 0:
                ax.set_yticklabels(dim_labels, fontsize=5.5)
            ax.invert_yaxis()
            _hide_spines(ax)

            if pidx == 0:
                ax.legend(fontsize=5, loc="lower right")

            _lab(ax, chr(97 + pidx))

        axes[1].set_xlabel("Score (%)")
        fig.tight_layout(w_pad=0.8)
        _save(fig, "sup_fig4_thinking_comparison")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Generating Nature-quality paper figures")
    print("=" * 55)

    fig1_behavioral_profiles()
    fig2_cooperation_divergence()
    fig3_generational_drift()
    fig4_endgame_curves()
    fig5_reciprocity_profiles()
    fig6_provider_clustering()
    ed_fig1_radar_charts()
    ed_fig2_behavioral_space()
    ed_fig3_dendrogram()
    ed_fig4_jsd_matrices()
    ed_fig5_round_timelines()
    ed_fig6_factor_loadings()
    ed_fig7_lexical_features()
    ed_fig8_crossplay_matrix()
    ed_fig9_cost_performance()
    ed_fig10_coverage()
    sup_fig1_cooperation_distributions()
    sup_fig2_strategy_response()
    sup_fig3_crossplay_effects()
    sup_fig4_thinking_comparison()

    print(f"\nAll figures saved to {OUT}")


if __name__ == "__main__":
    main()

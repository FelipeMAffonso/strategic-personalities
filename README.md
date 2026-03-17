# Large language models converge on competitive rationality but diverge on cooperation across providers and generations

Replication package for: **"Large language models converge on competitive rationality but diverge on cooperation across providers and generations"**

Felipe M. Affonso, Spears School of Business, Oklahoma State University

## Overview

This repository contains all data, analysis code, and figure generation scripts needed to reproduce every result, figure, and statistical test reported in the main text and supplementary materials. The dataset comprises 51,906 game-theoretic trials across 25 frontier language models from 7 developers, tested on 38 canonical games spanning 8 categories of strategic interaction.

An interactive dashboard for exploring the data is available at: **https://felipemaffonso.github.io/strategic-personalities/**

## Repository Structure

```
strategic-personalities/
  README.md                 This file
  reproduce.py              One-command script to regenerate all figures and statistics
  run.py                    Full experimental platform CLI (data collection + analysis)
  data/
    strategic_personalities.csv.gz  Authoritative dataset (51,906 rows, 45 columns, gzip-compressed)
    manuscript_numbers.json         Key statistics referenced in the manuscript
  analysis/
    compute_behavioral_profiles.py  Per-game metric extraction (Level 1)
    compute_all_stats.py            Compute all manuscript statistics (Level 2)
    cross_model_divergence.py       Hodoscope pipeline: behavioral fingerprinting (Level 3)
    factor_analysis.py              PCA and exploratory factor analysis
    generational_drift.py           Within-provider version comparisons
    generate_paper_figures.py       Generate all 20 publication figures
  environments/
    engine.py                       Game engine: prompt building, parsing, scoring
    prompts.py                      Framing conditions and prompt modifications
    strategies.py                   29 deterministic opponent strategies
    games/                          38 game implementations across 8 categories
  experiment/
    runner.py                       Thread-safe parallel trial runner with caching
    designs.py                      Named experimental design presets
    conditions.py                   25 experimental conditions
    matchups.py                     Cross-play, self-play, strategy-play logic
  config/
    models.py                       36-model registry with provider routing and pricing
  harness/
    core.py                         Multi-provider API harness (Anthropic, OpenAI, Google, OpenRouter)
    cost_tracker.py                 Per-provider budget tracking
  figures/                          All 20 figures (main + extended data + supplementary; PNG + PDF)
  dashboard/                        Interactive data explorer (GitHub Pages)
```

## Quick Start: Reproduce All Results

### Requirements

```bash
pip install matplotlib numpy scipy pandas scikit-learn
```

Python 3.10+ required. No GPU needed. All analyses run on CPU in under 5 minutes.

### One-Command Reproduction

```bash
python reproduce.py
```

This script will:
1. Load the dataset from `data/strategic_personalities.csv.gz`
2. Compute all behavioral profiles and manuscript statistics
3. Run factor analysis and hodoscope pipeline
4. Regenerate all 20 figures (6 main + 10 extended data + 4 supplementary)
5. Print a comparison of computed statistics against manuscript values

### Individual Scripts

```bash
# Statistical analysis only
python reproduce.py --stats

# Figures only
python reproduce.py --figures

# Full experimental platform (requires API keys)
python run.py --list-designs
python run.py --design pilot --dry-run
```

## Dataset Description

**File**: `data/strategic_personalities.csv.gz` (51,906 rows x 45 columns, gzip-compressed)

Two complementary designs generate the dataset. Design F tests 16 budget-tier models in a complete cross-play design (54,560 trials). Design G tests 9 frontier models in strategy-play only (approximately 3,300 trials).

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `model_key` | str | Model identifier (e.g., `claude-opus-4.6`, `gpt-5-nano`) |
| `provider` | str | API provider (anthropic, openai, google, openrouter) |
| `model_family` | str | Model family for generational analysis |
| `game_id` | str | One of 38 game identifiers |
| `game_category` | str | One of 8 categories |
| `opponent` | str | Opponent model key or strategy name |
| `matchup_type` | str | cross_play, self_play, or model_vs_strategy |
| `condition` | str | Experimental condition |
| `trial_num` | int | Trial number (1-5 for Design F, 1 for Design G) |
| `num_rounds` | int | Number of rounds played (5, 10, 15, or 20) |
| `cooperation_rate` | float | Fraction of cooperative choices (cooperation games) |
| `coordination_rate` | float | Fraction of coordinated outcomes |
| `offer_ratio` | float | Offer as fraction of endowment (fairness games) |
| `strategic_depth` | float | k-level estimate (depth games) |
| `trust_index` | float | Fraction of endowment sent (trust games) |
| `mean_bid` | float | Average bid (competition games) |
| `demand_ratio` | float | Demand as fraction of surplus (negotiation games) |
| `risk_taking_rate` | float | Fraction of risky choices (risk games) |
| `cost_usd` | float | API cost for this trial |

### Models (25 total)

| Provider | Models |
|----------|--------|
| Anthropic | Claude Haiku 4.5, Haiku 4.5 Thinking, Sonnet 4.5, Sonnet 4.6, Opus 4.5, Opus 4.6 |
| OpenAI | GPT-4o Mini, GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, GPT-5 Mini, GPT-5 Nano, GPT-5.3, GPT-5.4 |
| Google | Gemini 2.0 Flash, 2.5 Flash, 2.5 Flash Thinking, 3 Flash, 3 Pro, 3.1 Pro |
| DeepSeek | DeepSeek V3, DeepSeek R1 |
| Meta | LLaMA 3.3 70B |
| Mistral | Ministral 14B |
| Alibaba | Qwen 3.5 Flash |

### Game Categories (38 games across 8 categories)

| Category | Games | Key Metric |
|----------|-------|------------|
| Cooperation (10) | PD (4 variants), public goods (3), commons, diner's, El Farol | cooperation_rate |
| Coordination (6) | BoS (2), stag hunt (2), matching pennies, focal point | coordination_rate |
| Fairness (3) | Ultimatum, dictator, third-party punishment | offer_ratio |
| Strategic depth (5) | Beauty contest (2), centipede (2), 11-20 money | strategic_depth |
| Trust (3) | Berg trust, gift exchange, repeated trust | trust_index |
| Competition (4) | First-price, Vickrey, all-pay auction, Colonel Blotto | mean_bid |
| Negotiation (3) | Nash demand, alternating offers, multi-issue | demand_ratio |
| Risk (4) | Chicken (2), signaling, cheap talk | risk_taking_rate |

## Key Results

All statistics below are computed from the dataset and verified against the manuscript (445 pass / 0 fail):

- **Cooperation divergence**: 48-fold, from 1.5% (GPT-5 Nano) to 71.5% (Claude Opus 4.6)
- **Coordination convergence**: CV = 0.06 (range 0.62-0.85 across 25 models)
- **Strategic depth convergence**: CV = 0.11 (range 0.54-0.90)
- **OpenAI generational decline**: 50.3% (GPT-4o Mini) to 1.5% (GPT-5 Nano)
- **Google generational increase**: 8.3% (Gemini 2.0 Flash) to 56.8% (Gemini 3 Flash)
- **Endgame divergence**: 57% vs 0% final-round cooperation (Cohen's h = 1.71)
- **Provider separation ratio**: 1.33 (hodoscope pipeline)
- **Total API cost**: USD 1,950

## Figures

### Main Figures
- **Fig 1**: Behavioral fingerprint heatmap and radar profiles (convergence vs divergence)
- **Fig 2**: Cooperation divergence across 25 models (Cleveland dot plot)
- **Fig 3**: Generational drift (Anthropic, OpenAI, Google slope charts)
- **Fig 4**: Endgame cooperation curves (three archetype strategies)
- **Fig 5**: Reciprocity profiles (four behavioral archetypes)
- **Fig 6**: Provider clustering (UMAP + silhouette scores)

### Extended Data Figures
- **ED 1**: Radar charts for all 25 models (5x5 grid)
- **ED 2**: Behavioral embedding space (UMAP with KDE density)
- **ED 3**: Hierarchical clustering dendrogram (34 agents)
- **ED 4**: Jensen-Shannon distance matrices
- **ED 5**: Round-by-round cooperation trajectories
- **ED 6**: Factor analysis loadings and scree plot
- **ED 7**: Lexical features of strategic reasoning
- **ED 8**: Cross-play cooperation matrices
- **ED 9**: Cost-performance analysis
- **ED 10**: Experimental coverage matrix

### Supplementary Figures
- **S1**: Cooperation rate distributions (box plots)
- **S2**: Strategy-play response profiles (heatmap)
- **S3**: Cross-play interaction effects
- **S4**: Thinking model comparison (paired bars)

## Interactive Dashboard

The `dashboard/` folder contains a React-based interactive explorer deployed at GitHub Pages. It provides 8 views for exploring the data:

1. **Overview** — Key findings and provider comparison
2. **Model Cards** — Per-model behavioral profiles with radar charts
3. **Behavioral Space** — Interactive UMAP/PCA scatter
4. **Game Explorer** — Round-by-round traces and strategy profiles
5. **Head-to-Head** — Pairwise model comparison
6. **Cross-Play** — Model interaction matrices
7. **JSD Divergence** — Jensen-Shannon distance heatmaps
8. **Reasoning Browser** — Searchable reasoning traces

## Data Collection

Data were collected between February and March 2026 via direct API calls to Anthropic, OpenAI, Google (Vertex AI and OpenRouter), and OpenRouter. All models were tested at temperature 1.0. Each trial consists of a game rules prompt, growing round-by-round history, and a query requesting the model's action choice. Option labels are randomised per trial. Reasoning traces are captured for thinking models via provider-specific mechanisms.

## License

Data and code are provided for peer review and academic replication. Please contact the author for other uses.

## Citation

```bibtex
@article{affonso2026strategic,
  title={Large language models converge on competitive rationality but diverge
         on cooperation across providers and generations},
  author={Affonso, Felipe M.},
  year={2026},
  journal={Manuscript submitted for publication}
}
```

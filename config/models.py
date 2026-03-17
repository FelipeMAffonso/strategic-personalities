"""
Model Registry for Double Alignment Experiments
================================================
Defines all models to test, their API configurations, and pricing.
Adapted from cognitive-traps-jcr/v2_revision/test_models.py
"""

# ---------------------------------------------------------------------------
# Full model registry (all available frontier models)
# ---------------------------------------------------------------------------

ALL_MODELS = {
    # Anthropic - Claude 3 generation
    "claude-haiku-3.0":             {"provider": "anthropic", "model_id": "claude-3-haiku-20240307",    "thinking": False},
    "claude-haiku-3.5":             {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022",  "thinking": False},
    # Anthropic - Claude 4.5
    "claude-haiku-4.5":             {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001",  "thinking": False},
    "claude-haiku-4.5-thinking":    {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001",  "thinking": True},
    "claude-sonnet-4.5":            {"provider": "anthropic", "model_id": "claude-sonnet-4-5-20250929", "thinking": False},
    "claude-sonnet-4.5-thinking":   {"provider": "anthropic", "model_id": "claude-sonnet-4-5-20250929", "thinking": True},
    "claude-opus-4.5":              {"provider": "anthropic", "model_id": "claude-opus-4-5",            "thinking": False},
    # Anthropic - Claude 4.6
    "claude-sonnet-4.6":            {"provider": "anthropic", "model_id": "claude-sonnet-4-6",          "thinking": False},
    "claude-opus-4.6":              {"provider": "anthropic", "model_id": "claude-opus-4-6",            "thinking": False},
    # OpenAI - GPT-4o
    "gpt-4o":                       {"provider": "openai", "model_id": "gpt-4o",                       "thinking": False},
    "gpt-4o-mini":                  {"provider": "openai", "model_id": "gpt-4o-mini",                  "thinking": False},
    # OpenAI - GPT-4.1
    "gpt-4.1":                      {"provider": "openai", "model_id": "gpt-4.1",                      "thinking": False},
    "gpt-4.1-mini":                 {"provider": "openai", "model_id": "gpt-4.1-mini",                 "thinking": False},
    "gpt-4.1-nano":                 {"provider": "openai", "model_id": "gpt-4.1-nano",                 "thinking": False},
    # OpenAI - GPT-5
    "gpt-5":                        {"provider": "openai", "model_id": "gpt-5-chat-latest",            "thinking": False},
    "gpt-5-thinking":               {"provider": "openai", "model_id": "gpt-5",                        "thinking": True},
    "gpt-5-mini":                   {"provider": "openai", "model_id": "gpt-5-mini",                   "thinking": False},
    "gpt-5-nano":                   {"provider": "openai", "model_id": "gpt-5-nano",                   "thinking": False},
    # OpenAI - GPT-5.1
    "gpt-5.1":                      {"provider": "openai", "model_id": "gpt-5.1-chat-latest",          "thinking": False},
    "gpt-5.1-thinking":             {"provider": "openai", "model_id": "gpt-5.1",                      "thinking": True},
    # OpenAI - GPT-5.2
    "gpt-5.2":                      {"provider": "openai", "model_id": "gpt-5.2-chat-latest",          "thinking": False},
    "gpt-5.2-thinking":             {"provider": "openai", "model_id": "gpt-5.2",                      "thinking": True},
    "gpt-5.2-pro":                  {"provider": "openai", "model_id": "gpt-5.2-pro",                  "thinking": True},
    # OpenAI - GPT-5.3
    "gpt-5.3":                      {"provider": "openai", "model_id": "gpt-5.3-chat-latest",          "thinking": False},
    # OpenAI - GPT-5.4
    "gpt-5.4":                      {"provider": "openai", "model_id": "gpt-5.4",                      "thinking": False},
    # Google models — ALL routed through OpenRouter
    "gemini-2.0-flash":             {"provider": "openrouter", "model_id": "google/gemini-2.0-flash-001",    "thinking": False},
    "gemini-2.5-flash":             {"provider": "openrouter", "model_id": "google/gemini-2.5-flash",        "thinking": False},
    "gemini-2.5-flash-lite":        {"provider": "openrouter", "model_id": "google/gemini-2.5-flash-lite-preview", "thinking": False},
    "gemini-2.5-flash-thinking":    {"provider": "openrouter", "model_id": "google/gemini-2.5-flash",        "thinking": False},
    "gemini-2.5-pro":               {"provider": "openrouter", "model_id": "google/gemini-2.5-pro-preview",  "thinking": False},
    "gemini-2.5-pro-thinking":      {"provider": "openrouter", "model_id": "google/gemini-2.5-pro-preview",  "thinking": True},
    "gemini-3-flash":               {"provider": "openrouter", "model_id": "google/gemini-3-flash-preview",  "thinking": False},
    "gemini-3-pro":                 {"provider": "google_vertex", "model_id": "gemini-3-pro-preview",    "thinking": True},
    "gemini-3.1-pro":               {"provider": "google_vertex", "model_id": "gemini-3.1-pro-preview",  "thinking": True},
    "gemma-3-27b":                  {"provider": "openrouter", "model_id": "google/gemma-3-27b-it",          "thinking": False},
    # OpenRouter - open-source flagships (via OpenAI-compatible API)
    "llama-3.3-70b":                {"provider": "openrouter", "model_id": "meta-llama/llama-3.3-70b-instruct",    "thinking": False},
    "deepseek-r1":                  {"provider": "openrouter", "model_id": "deepseek/deepseek-r1",                 "thinking": False},
    "deepseek-v3":                  {"provider": "openrouter", "model_id": "deepseek/deepseek-chat-v3-0324",       "thinking": False},
    "qwen-2.5-72b":                 {"provider": "openrouter", "model_id": "qwen/qwen-2.5-72b-instruct",          "thinking": False},
    "kimi-k2":                      {"provider": "openrouter", "model_id": "moonshotai/kimi-k2",                   "thinking": False},
    "ministral-14b":                {"provider": "openrouter", "model_id": "mistralai/ministral-14b-2512",        "thinking": False},
    "qwen3.5-flash":                {"provider": "openrouter", "model_id": "qwen/qwen3.5-flash-02-23",            "thinking": False},
}

# ---------------------------------------------------------------------------
# Model sets for experiment designs
# ---------------------------------------------------------------------------

# Smoke: 3 cheapest models, verify pipeline works
SMOKE_MODELS = {k: ALL_MODELS[k] for k in [
    "gemma-3-27b", "gpt-4.1-nano", "gemini-2.5-flash",
]}

# Pilot 5: one per major provider, all cheap
PILOT_5_MODELS = {k: ALL_MODELS[k] for k in [
    "claude-haiku-4.5", "gpt-4.1-mini", "gemini-2.5-flash",
    "deepseek-v3", "llama-3.3-70b",
]}

# Pilot 10: original pilot set (backward compat)
PILOT_MODELS = {k: ALL_MODELS[k] for k in [
    "claude-haiku-4.5", "claude-sonnet-4.5", "claude-opus-4.5",
    "gpt-4o", "gpt-5.2", "gpt-5.2-thinking", "gpt-5-mini",
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro",
]}

# Core 17: one per distinct model family (main paper)
CORE_MODELS = {k: ALL_MODELS[k] for k in [
    "claude-haiku-4.5", "claude-sonnet-4.5", "claude-opus-4.5",
    "gpt-4o", "gpt-4.1", "gpt-5", "gpt-5.2",
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro",
    "llama-3.3-70b", "deepseek-r1", "deepseek-v3",
    "qwen-2.5-72b", "kimi-k2",
    "gemma-3-27b", "gpt-5-mini",
]}

# Generational: all version chains for drift analysis
GENERATIONAL_MODELS = {k: ALL_MODELS[k] for k in [
    "claude-haiku-3.0", "claude-haiku-3.5", "claude-haiku-4.5",
    "claude-sonnet-4.5", "claude-sonnet-4.6",
    "claude-opus-4.5", "claude-opus-4.6",
    "gpt-4o", "gpt-4.1", "gpt-5", "gpt-5.1", "gpt-5.2",
    "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini",
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash",
    "gemini-2.5-pro", "gemini-3-pro", "gemini-3.1-pro",
    "claude-sonnet-4.5-thinking", "gpt-5.2-thinking",
    "gemini-2.5-pro-thinking",
]}

# Design F: 16 budget-optimized models across 4 providers (Nature design)
DESIGN_F_MODELS = {k: ALL_MODELS[k] for k in [
    # Google (free tier) - 4 models
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-thinking", "gemini-3-flash",
    # Anthropic (haiku only) - 2 models
    "claude-haiku-4.5", "claude-haiku-4.5-thinking",
    # OpenAI - 5 models
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano",
    # OpenRouter - 5 models
    "llama-3.3-70b", "deepseek-v3", "deepseek-r1", "ministral-14b", "qwen3.5-flash",
]}

# Design G: 9 frontier models (Opus self-play only, rest full participation)
DESIGN_G_MODELS = {k: ALL_MODELS[k] for k in [
    # Anthropic - frontier (Opus = self-play only)
    "claude-opus-4.5", "claude-opus-4.6", "claude-sonnet-4.5", "claude-sonnet-4.6",
    # Google - thinking models
    "gemini-3-pro", "gemini-3.1-pro",
    # OpenAI
    "gpt-4.1", "gpt-5.3", "gpt-5.4",
]}

# Named model sets for design system
MODEL_SETS = {
    "smoke_3": SMOKE_MODELS,
    "pilot_5": PILOT_5_MODELS,
    "pilot_10": PILOT_MODELS,
    "core_17": CORE_MODELS,
    "generational": GENERATIONAL_MODELS,
    "design_f": DESIGN_F_MODELS,
    "design_g": DESIGN_G_MODELS,
    "all": ALL_MODELS,
}

def get_model_set(name: str) -> dict:
    """Get model set by name."""
    if name not in MODEL_SETS:
        raise ValueError(f"Unknown model set: {name}. "
                         f"Available: {list(MODEL_SETS.keys())}")
    return MODEL_SETS[name]

# ---------------------------------------------------------------------------
# Pricing per million tokens (input_$/M, output_$/M)
# ---------------------------------------------------------------------------

PRICING = {
    "claude-3-haiku-20240307":      (0.25,  1.25),
    "claude-3-5-haiku-20241022":    (0.80,  4.00),
    "claude-haiku-4-5-20251001":    (1.00,  5.00),
    "claude-sonnet-4-5-20250929":   (3.00,  15.00),
    "claude-opus-4-5":              (5.00,  25.00),
    "claude-sonnet-4-6":            (3.00,  15.00),
    "claude-opus-4-6":              (5.00,  25.00),
    "gpt-4o":                       (2.50,  10.00),
    "gpt-4o-mini":                  (0.15,   0.60),
    "gpt-4.1":                      (2.00,   8.00),
    "gpt-4.1-mini":                 (0.40,   1.60),
    "gpt-4.1-nano":                 (0.10,   0.40),
    "gpt-5-chat-latest":            (1.25,  10.00),
    "gpt-5":                        (1.25,  10.00),
    "gpt-5-mini":                   (0.40,   1.60),
    "gpt-5-nano":                   (0.05,   0.40),
    "gpt-5.1-chat-latest":          (1.25,  10.00),
    "gpt-5.1":                      (1.25,  10.00),
    "gpt-5.2-chat-latest":          (1.75,  14.00),
    "gpt-5.2":                      (1.75,  14.00),
    "gpt-5.2-pro":                  (15.00, 60.00),
    "gpt-5.3-chat-latest":          (1.75,  14.00),
    "gpt-5.4":                      (1.25,  15.00),
    "gemini-2.0-flash":             (0.10,  0.40),
    "gemini-2.5-flash":             (0.15,  0.60),
    "gemini-2.5-flash-lite":        (0.075, 0.30),
    "gemini-2.5-pro":               (1.25,  10.00),
    "gemini-3-flash-preview":       (0.15,  0.60),
    "gemini-3-pro-preview":         (2.00,  12.00),
    "gemini-3.1-pro-preview":       (2.00,  12.00),
    "gemma-3-27b-it":               (0.04,   0.15),
    # OpenRouter models
    "meta-llama/llama-3.3-70b-instruct":    (0.135,  0.40),
    "deepseek/deepseek-r1":                 (0.40,   1.75),
    "deepseek/deepseek-chat-v3-0324":       (0.20,   0.80),
    "qwen/qwen-2.5-72b-instruct":          (0.80,   0.80),
    "moonshotai/kimi-k2":                   (0.60,   2.50),
    "mistralai/ministral-14b-2512":         (0.20,   0.20),
    "qwen/qwen3.5-flash-02-23":            (0.10,   0.40),
}

# ---------------------------------------------------------------------------
# Provider groupings (for budget tracking)
# ---------------------------------------------------------------------------

def get_provider_models(provider: str, model_set: dict = None) -> dict:
    """Get all models for a given provider from a model set."""
    if model_set is None:
        model_set = ALL_MODELS
    return {k: v for k, v in model_set.items() if v["provider"] == provider}

def compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    """Compute estimated cost in USD based on token counts."""
    if model_id in PRICING:
        inp_rate, out_rate = PRICING[model_id]
        return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000
    return None

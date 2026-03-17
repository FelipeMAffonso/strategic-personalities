"""
Cost Tracker and Budget Enforcement
=====================================
Tracks API costs per provider and enforces per-platform budget limits.
Prevents runaway costs during experimentation.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime


class CostTracker:
    """
    Track cumulative API costs per provider and enforce budget limits.

    Usage:
        tracker = CostTracker(budget_per_provider=5.00)
        tracker.record_call("anthropic", "claude-haiku-4.5", 500, 200, 0.0035)
        tracker.check_budget("anthropic")  # raises if over budget
    """

    def __init__(self, budget_per_provider: float = 10.00,
                 max_calls_per_provider: int = 30,
                 log_dir: Path = None):
        """
        Args:
            budget_per_provider: Maximum USD spend per provider
            max_calls_per_provider: Maximum API calls per provider
            log_dir: Directory to save cost logs (default: data/costs/)
        """
        self.budget_per_provider = budget_per_provider
        self.max_calls_per_provider = max_calls_per_provider
        self.log_dir = log_dir or Path(__file__).resolve().parent.parent / "data" / "costs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Running totals (thread-safe via _lock)
        self._lock = threading.Lock()
        self._costs: dict[str, float] = {}        # provider -> total USD
        self._calls: dict[str, int] = {}           # provider -> call count
        self._tokens: dict[str, dict] = {}         # provider -> {input, output}
        self._history: list[dict] = []             # all call records

    def record_call(self, provider: str, model_id: str,
                    input_tokens: int, output_tokens: int,
                    cost_usd: float | None,
                    experiment: str = "",
                    trial_id: str = ""):
        """Record a single API call (thread-safe)."""
        with self._lock:
            if provider not in self._costs:
                self._costs[provider] = 0.0
                self._calls[provider] = 0
                self._tokens[provider] = {"input": 0, "output": 0}

            if cost_usd is not None:
                self._costs[provider] += cost_usd
            self._calls[provider] += 1
            self._tokens[provider]["input"] += input_tokens
            self._tokens[provider]["output"] += output_tokens

            self._history.append({
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model_id": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "cumulative_cost": self._costs[provider],
                "call_number": self._calls[provider],
                "experiment": experiment,
                "trial_id": trial_id,
            })

    def check_budget(self, provider: str) -> bool:
        """
        Check if provider is within budget (thread-safe). Returns True if OK.
        Raises BudgetExceededError if over limit.
        """
        with self._lock:
            cost = self._costs.get(provider, 0.0)
            calls = self._calls.get(provider, 0)

        if cost > self.budget_per_provider:
            raise BudgetExceededError(
                f"{provider}: ${cost:.4f} spent (budget: ${self.budget_per_provider:.2f})"
            )
        if calls >= self.max_calls_per_provider:
            raise BudgetExceededError(
                f"{provider}: {calls} calls made (limit: {self.max_calls_per_provider})"
            )
        return True

    def can_afford(self, provider: str, estimated_cost: float = 0.01) -> bool:
        """Check if we can afford another call without raising (thread-safe)."""
        with self._lock:
            cost = self._costs.get(provider, 0.0)
            calls = self._calls.get(provider, 0)
        return (cost + estimated_cost <= self.budget_per_provider
                and calls < self.max_calls_per_provider)

    def get_summary(self) -> dict:
        """Get current cost summary."""
        providers = set(self._costs.keys()) | set(self._calls.keys())
        summary = {}
        for provider in sorted(providers):
            summary[provider] = {
                "total_cost_usd": round(self._costs.get(provider, 0.0), 6),
                "total_calls": self._calls.get(provider, 0),
                "input_tokens": self._tokens.get(provider, {}).get("input", 0),
                "output_tokens": self._tokens.get(provider, {}).get("output", 0),
                "budget_remaining": round(
                    self.budget_per_provider - self._costs.get(provider, 0.0), 6
                ),
                "calls_remaining": (
                    self.max_calls_per_provider - self._calls.get(provider, 0)
                ),
            }
        return summary

    def print_summary(self):
        """Print formatted cost summary."""
        summary = self.get_summary()
        total_cost = sum(v["total_cost_usd"] for v in summary.values())
        total_calls = sum(v["total_calls"] for v in summary.values())

        print("\n" + "=" * 60)
        print("COST SUMMARY")
        print("=" * 60)
        for provider, data in summary.items():
            print(f"  {provider:12s}  ${data['total_cost_usd']:.4f}  "
                  f"({data['total_calls']} calls, "
                  f"${data['budget_remaining']:.4f} remaining)")
        print(f"  {'TOTAL':12s}  ${total_cost:.4f}  ({total_calls} calls)")
        print("=" * 60)

    def save_log(self, filename: str = None):
        """Save full cost history to JSON."""
        if filename is None:
            filename = f"cost_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        log = {
            "summary": self.get_summary(),
            "budget_per_provider": self.budget_per_provider,
            "max_calls_per_provider": self.max_calls_per_provider,
            "history": self._history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Cost log saved to: {path}")
        return path


class BudgetExceededError(Exception):
    """Raised when a provider's budget limit is exceeded."""
    pass

"""Token + dollar accounting for Meta Model API calls, with a hard daily cap.

Every LLM call in lab2rob is recorded here as one JSONL entry. Calls that
would push today's spend past the cap are refused *before* any request is
sent (see :meth:`CostLog.is_capped`); spend that already happened is always
logged so the books stay honest.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

DAILY_CAP_USD = 12.0
TOTAL_BUDGET_USD = 150.0
PRICES_ENV_VAR = "LAB2ROB_MODEL_PRICES"

#: USD per 1K tokens as {model: (input, output)}.
#: PLACEHOLDER rates — replace with the actual Meta Model API rates.
#: Override at runtime with LAB2ROB_MODEL_PRICES='{"model": [in_per_1k, out_per_1k]}'.
MODEL_PRICES_PER_1K = {
    "muse-spark-1.3-contributor": (0.003, 0.015),
    "muse-spark-1.3": (0.005, 0.020),
    "muse-wonder-1.0": (0.015, 0.060),
    "sam-3.1": (0.005, 0.020),
}

#: Lifetime sub-budgets per job name (USD). Section 5 caps hard cases at ~$15.
JOB_BUDGETS_USD = {"hard_case": 15.0}


class DailyCapExceeded(RuntimeError):
    """A call would push today's spend past the daily cap."""


class JobBudgetExceeded(RuntimeError):
    """A call would push a job past its lifetime sub-budget."""


class UnknownModelPrice(KeyError):
    """A model has no entry in the price table."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_prices(extra=None):
    """Merge built-in, env-var and explicit price tables (later wins)."""
    prices = dict(MODEL_PRICES_PER_1K)
    raw = os.environ.get(PRICES_ENV_VAR)
    if raw:
        for model, pair in json.loads(raw).items():
            prices[model] = (float(pair[0]), float(pair[1]))
    if extra:
        for model, pair in extra.items():
            prices[model] = (float(pair[0]), float(pair[1]))
    return prices


class CostLog:
    """Append-only JSONL ledger of LLM spend with cap enforcement."""

    def __init__(self, path="runs/llm_costs.jsonl", daily_cap_usd=DAILY_CAP_USD,
                 prices=None, job_budgets=None):
        self.path = path
        self.daily_cap_usd = float(daily_cap_usd)
        self.prices = load_prices(prices)
        self.job_budgets = dict(JOB_BUDGETS_USD)
        if job_budgets:
            self.job_budgets.update(job_budgets)

    def entries(self):
        """All recorded entries; unreadable lines are skipped, never fatal."""
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out

    def cost_for(self, model, prompt_tokens, completion_tokens):
        try:
            per_in, per_out = self.prices[model]
        except KeyError:
            raise UnknownModelPrice(f"No price for model {model!r}") from None
        return (prompt_tokens / 1000) * per_in + (completion_tokens / 1000) * per_out

    def _spend(self, pred):
        return sum(float(e.get("cost_usd", 0.0)) for e in self.entries() if pred(e))

    def spend_on(self, day):
        return self._spend(lambda e: e.get("day") == day)

    def spend_today(self):
        return self.spend_on(_utc_now().date().isoformat())

    def spend_for_job(self, job):
        return self._spend(lambda e: e.get("job") == job)

    def spend_total(self):
        return self._spend(lambda e: True)

    def summarize_by_job(self):
        """Per-job rollup: calls, tokens (incl. reasoning), and cost.

        Cached calls count as calls but add no cost; reasoning_tokens is 0
        for entries recorded before reasoning was tracked.
        """
        summary: dict[str, dict] = {}
        for entry in self.entries():
            job = str(entry.get("job", "?"))
            row = summary.setdefault(job, {"calls": 0, "prompt_tokens": 0,
                                           "completion_tokens": 0,
                                           "reasoning_tokens": 0,
                                           "cost_usd": 0.0})
            row["calls"] += 1
            for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                row[field] += int(entry.get(field, 0) or 0)
            row["cost_usd"] += float(entry.get("cost_usd", 0.0) or 0.0)
        for row in summary.values():
            row["cost_usd"] = round(row["cost_usd"], 6)
        return summary

    def is_capped(self):
        """True once today's spend has reached the daily cap."""
        return self.spend_today() >= self.daily_cap_usd

    def job_capped(self, job):
        budget = self.job_budgets.get(job)
        return budget is not None and self.spend_for_job(job) >= budget

    def record(self, job, model, prompt_tokens, completion_tokens,
               cached=False, force=False, reasoning_tokens=0):
        """Log one call; refuse (unless forced) when a cap would break."""
        cost = 0.0 if cached else self.cost_for(model, prompt_tokens, completion_tokens)
        if not force and not cached:
            today = _utc_now().date().isoformat()
            if self.spend_on(today) + cost > self.daily_cap_usd:
                raise DailyCapExceeded(
                    f"Daily cap ${self.daily_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.spend_on(today):.4f}, call costs ${cost:.4f})"
                )
            budget = self.job_budgets.get(job)
            if budget is not None and self.spend_for_job(job) + cost > budget:
                raise JobBudgetExceeded(
                    f"Job {job!r} budget ${budget:.2f} would be exceeded"
                )
        entry = {
            "ts": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "day": _utc_now().date().isoformat(),
            "job": job,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost, 6),
            "cached": bool(cached),
            "reasoning_tokens": int(reasoning_tokens),
        }
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

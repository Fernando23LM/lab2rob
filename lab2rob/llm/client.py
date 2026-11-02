"""Single entry point for every LLM call in lab2rob.

:func:`model_for_job` routes each compiler job to its model (section 5),
:class:`Lab2RobClient.complete` serves repeats from the cache, refuses calls
once the daily cap is reached, and logs every call (cached or not) with its
token counts and cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .cache import LLMCache
from .cost_log import CostLog, DailyCapExceeded, JobBudgetExceeded

BASE_URL = "https://api.meta.ai/v1"
API_KEY_ENV = "MODEL_API_KEY"
#: Fallback ceiling when callers pass no max_tokens: reasoning models spend
#: dozens of tokens thinking before answering, so tiny/absent limits yield
#: empty content (observed: 48 reasoning tokens with max_tokens=50).
DEFAULT_MAX_TOKENS = 1024

#: Job name -> model id, from LAB2ROB_CONTEXT.md section 5.
#: The visual job is routed to the contributor tier (ST-03): sam-3.1 is the
#: spec'd vision model but its endpoint returns empty completions, so vision
#: calls go to the contributor VLM until sam-3.1 generates. The Responses
#: transport below stays ready for its return.
MODEL_ROUTING = {
    "ground_fallback": "muse-spark-1.3-contributor",
    "program_planner": "muse-spark-1.3",
    "judge_planner": "muse-spark-1.3",
    "visual": "muse-spark-1.3-contributor",
    "evolve": "muse-spark-1.3-contributor",
    "hard_case": "muse-wonder-1.0",
}

#: Job name -> reasoning effort, sent as top-level "reasoning_effort" on Chat
#: Completions calls (Responses API: {"reasoning": {"effort": ...}}). Grounding
#: jobs think the least: the ST-04 pilot showed the contributor model burning
#: ~3k reasoning tokens per call at default effort, starving its own answers.
#: The visual job needs minimal for the same reason (observed: 510 reasoning
#: tokens with no output at default effort). Jobs without an entry send none.
JOB_REASONING_EFFORT = {
    "ground_fallback": "minimal",
    "program_planner": "low",
    "judge_planner": "low",
    "visual": "minimal",
    "evolve": "low",
    "hard_case": "medium",
}

#: Models served only on the /v1/responses endpoint. Chat Completions answers
#: them with 404 ("Use /v1/responses"), so the transport converts messages
#: and parameters to the Responses shape before sending.
RESPONSES_MODELS = frozenset({"sam-3.1"})


class UnknownJob(KeyError):
    """A job name has no model routing entry."""


def model_for_job(job, routing=None):
    table = MODEL_ROUTING if routing is None else routing
    try:
        return table[job]
    except KeyError:
        raise UnknownJob(f"Unknown LLM job {job!r}") from None


def effort_for_job(job):
    """Reasoning effort for a job, or None when the call sends no effort."""
    return JOB_REASONING_EFFORT.get(job)


@dataclass
class LLMResponse:
    text: str
    model: str
    job: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    reasoning_tokens: int = 0
    finish_reason: str | None = None


class OpenAITransport:
    """Thin adapter over the OpenAI SDK pointed at the Meta Model API."""

    def __init__(self, api_key=None, base_url=BASE_URL):
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "The 'openai' package is required for live LLM calls: "
                "pip install openai"
            ) from None
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise RuntimeError(f"Set {API_KEY_ENV} for live LLM calls")
        self._client = openai.OpenAI(api_key=key, base_url=base_url)

    def complete(self, model, messages, **kwargs):
        if model in RESPONSES_MODELS:
            return self._complete_responses(model, messages, **kwargs)
        resp = self._client.chat.completions.create(
            model=model, messages=messages, **kwargs)
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning = details.reasoning_tokens if details and details.reasoning_tokens else 0
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "reasoning_tokens": reasoning,
            "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
        }

    @staticmethod
    def _responses_input(messages):
        """Chat messages -> Responses `input` (roles unchanged, parts mapped)."""
        converted = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                converted.append({"role": message.get("role"),
                                  "content": content})
                continue
            parts = []
            for part in content or []:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                elif kind == "image_url":
                    url = part["image_url"]
                    url = url["url"] if isinstance(url, dict) else url
                    parts.append({"type": "input_image", "image_url": url})
                else:
                    raise ValueError(
                        f"Cannot send {kind!r} parts to the Responses API")
            converted.append({"role": message.get("role"), "content": parts})
        return converted

    def _complete_responses(self, model, messages, **kwargs):
        params = dict(kwargs)
        if "max_tokens" in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        if "reasoning_effort" in params:
            params["reasoning"] = {"effort": params.pop("reasoning_effort")}
        resp = self._client.responses.create(
            model=model, input=self._responses_input(messages), **params)
        usage = resp.usage
        details = getattr(usage, "output_tokens_details", None) if usage else None
        reasoning = details.reasoning_tokens if details and details.reasoning_tokens else 0
        status = getattr(resp, "status", None)
        return {
            "text": resp.output_text or "",
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "reasoning_tokens": reasoning,
            "finish_reason": "stop" if status == "completed" else status,
        }


class Lab2RobClient:
    """Routed, cached, cost-logged LLM client. Inject a fake transport in tests."""

    def __init__(self, cache=None, cost_log=None, transport=None, routing=None):
        self.cache = cache or LLMCache()
        self.cost_log = cost_log or CostLog()
        self.transport = transport  # created lazily: tests never need openai/a key
        self.routing = dict(MODEL_ROUTING)
        if routing:
            self.routing.update(routing)

    def _transport(self):
        if self.transport is None:
            self.transport = OpenAITransport()
        return self.transport

    def complete(self, job, messages, use_cache=True, **kwargs):
        model = model_for_job(job, self.routing)
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        effort = effort_for_job(job)
        if effort is not None:
            # An explicit per-call effort (e.g. the truncation-escalation
            # ladder) wins over the job default; the cache key covers it.
            kwargs.setdefault("reasoning_effort", effort)
        if use_cache:
            hit = self.cache.get(model, messages, **kwargs)
            if hit is not None:
                reasoning = hit.get("reasoning_tokens", 0)
                self.cost_log.record(job, model, hit["prompt_tokens"],
                                     hit["completion_tokens"], cached=True,
                                     reasoning_tokens=reasoning)
                return LLMResponse(hit["text"], model, job, True,
                                   hit["prompt_tokens"], hit["completion_tokens"], 0.0,
                                   reasoning, hit.get("finish_reason"))
        if self.cost_log.is_capped():
            raise DailyCapExceeded(
                f"Daily cap ${self.cost_log.daily_cap_usd:.2f} reached "
                f"(spent ${self.cost_log.spend_today():.4f}); call refused before sending"
            )
        if self.cost_log.job_capped(job):
            raise JobBudgetExceeded(f"Job {job!r} budget exhausted; call refused")
        result = self._transport().complete(model, messages, **kwargs)
        reasoning = result.get("reasoning_tokens", 0)
        # force=True: the spend already happened, the books must show it.
        entry = self.cost_log.record(job, model, result["prompt_tokens"],
                                     result["completion_tokens"], force=True,
                                     reasoning_tokens=reasoning)
        if str(result.get("text", "")).strip():
            self.cache.put(model, messages, result, **kwargs)
        # Empty replies are never cached: serving them again would replay a
        # failure the model might not repeat (see UI-04).
        return LLMResponse(result["text"], model, job, False,
                           result["prompt_tokens"], result["completion_tokens"],
                           entry["cost_usd"], reasoning,
                           result.get("finish_reason"))


def default_client():
    """Client with default cache/cost-log locations for scripts and later stages."""
    return Lab2RobClient()

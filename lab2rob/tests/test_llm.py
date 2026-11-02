"""FD-02: cache hits cost zero HTTP calls; the daily cap blocks over-budget calls.

All tests run offline with a fake transport: they never need the openai
package, MODEL_API_KEY, or network access.
"""

import sys
from types import SimpleNamespace

import pytest

from lab2rob.llm import (
    JOB_REASONING_EFFORT,
    MODEL_ROUTING,
    CostLog,
    DailyCapExceeded,
    JobBudgetExceeded,
    Lab2RobClient,
    LLMCache,
    UnknownJob,
    effort_for_job,
)

TEST_PRICES = {"test-model": (1.0, 2.0), "other-model": (1.0, 2.0)}
MESSAGES = [{"role": "user", "content": "Pipette 100 uL from tube_1 to plate_1.A1."}]


class FakeTransport:
    """Counts calls like HTTP requests; fixed usage: 10 prompt + 5 completion."""

    def __init__(self, reasoning_tokens=0, finish_reason=None):
        self.calls = []
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason

    def complete(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        return {"text": f"reply-{len(self.calls)}",
                "prompt_tokens": 10, "completion_tokens": 5,
                "reasoning_tokens": self.reasoning_tokens,
                "finish_reason": self.finish_reason}


def make_client(tmp_path, transport=None, **log_kwargs):
    transport = transport or FakeTransport()
    cache = LLMCache(str(tmp_path / "cache"))
    log_kwargs.setdefault("prices", TEST_PRICES)
    log = CostLog(str(tmp_path / "costs.jsonl"), **log_kwargs)
    client = Lab2RobClient(cache=cache, cost_log=log, transport=transport,
                           routing={"t": "test-model", "u": "other-model"})
    return client, transport, log


def test_repeat_prompt_hits_cache_with_zero_extra_calls(tmp_path):
    client, transport, log = make_client(tmp_path)
    first = client.complete("t", MESSAGES)
    second = client.complete("t", MESSAGES)
    assert transport.calls and len(transport.calls) == 1
    assert second.text == first.text
    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == 0.0
    entries = log.entries()
    assert len(entries) == 2
    assert entries[1]["cached"] is True
    assert entries[1]["cost_usd"] == 0.0


def test_cache_key_includes_model_and_params(tmp_path):
    client, transport, _ = make_client(tmp_path)
    client.complete("t", MESSAGES)
    client.complete("u", MESSAGES)  # same prompt, other model -> miss
    client.complete("t", MESSAGES, temperature=0.0)
    client.complete("t", MESSAGES, temperature=1.0)  # other params -> miss
    assert len(transport.calls) == 4


def test_daily_cap_blocks_over_budget_calls(tmp_path):
    # One call costs 10/1000*1.0 + 5/1000*2.0 = $0.02; the pre-send check
    # blocks new calls once spend has reached the cap (call cost is only
    # known after sending, so the cap can overshoot by at most one call).
    client, transport, log = make_client(tmp_path, daily_cap_usd=0.02)
    client.complete("t", MESSAGES, use_cache=False)
    with pytest.raises(DailyCapExceeded):
        client.complete("t", [{"role": "user", "content": "other prompt"}],
                        use_cache=False)
    assert len(transport.calls) == 1  # refused BEFORE sending
    assert log.spend_today() == pytest.approx(0.02)


def test_job_budget_blocks_hard_case_overspend(tmp_path):
    log = CostLog(str(tmp_path / "c.jsonl"), prices={"m": (1000.0, 0.0)},
                  job_budgets={"hard_case": 1.0})
    log.record("hard_case", "m", 1, 0)  # $1.00, budget reached
    assert log.job_capped("hard_case") is True
    with pytest.raises(JobBudgetExceeded):
        log.record("hard_case", "m", 1, 0)


def test_job_routing_matches_spec():
    # ST-03: visual rides the contributor VLM until sam-3.1 generates.
    assert MODEL_ROUTING == {
        "ground_fallback": "muse-spark-1.3-contributor",
        "program_planner": "muse-spark-1.3",
        "judge_planner": "muse-spark-1.3",
        "visual": "muse-spark-1.3-contributor",
        "evolve": "muse-spark-1.3-contributor",
        "hard_case": "muse-wonder-1.0",
    }


def test_unknown_job_rejected(tmp_path):
    client, _, _ = make_client(tmp_path)
    with pytest.raises(UnknownJob):
        client.complete("nope", MESSAGES)


def test_effort_defaults_match_spec():
    assert JOB_REASONING_EFFORT == {
        "ground_fallback": "minimal",
        "program_planner": "low",
        "judge_planner": "low",
        "visual": "minimal",
        "evolve": "low",
        "hard_case": "medium",
    }
    assert effort_for_job("ground_fallback") == "minimal"
    assert effort_for_job("visual") == "minimal"
    assert effort_for_job("custom-job") is None


def make_spec_client(tmp_path):
    prices = dict(TEST_PRICES)
    prices["muse-spark-1.3-contributor"] = (1.0, 2.0)
    prices["sam-3.1"] = (1.0, 2.0)
    cache = LLMCache(str(tmp_path / "cache"))
    log = CostLog(str(tmp_path / "costs.jsonl"), prices=prices)
    transport = FakeTransport()
    return Lab2RobClient(cache=cache, cost_log=log, transport=transport), transport


def test_reasoning_effort_injected_unless_overridden(tmp_path):
    client, transport = make_spec_client(tmp_path)
    client.complete("ground_fallback", MESSAGES, use_cache=False)
    assert transport.calls[0][2]["reasoning_effort"] == "minimal"
    client.complete("ground_fallback", MESSAGES, use_cache=False,
                    reasoning_effort="low")  # escalation ladder wins
    assert transport.calls[1][2]["reasoning_effort"] == "low"
    client.complete("visual", MESSAGES, use_cache=False)
    assert transport.calls[2][2]["reasoning_effort"] == "minimal"


def test_effort_change_busts_cache(tmp_path):
    client, transport = make_spec_client(tmp_path)
    client.complete("ground_fallback", MESSAGES)
    client.complete("ground_fallback", MESSAGES)
    assert len(transport.calls) == 1  # same effort: cache hit
    client.complete("ground_fallback", MESSAGES, reasoning_effort="low")
    assert len(transport.calls) == 2  # other effort: miss


def test_summarize_by_job_rolls_up_reasoning_and_cost(tmp_path):
    log = CostLog(str(tmp_path / "c.jsonl"), prices=TEST_PRICES)
    log.record("t", "test-model", 10, 5, reasoning_tokens=4)
    log.record("t", "test-model", 10, 5, cached=True, reasoning_tokens=4)
    log.record("u", "other-model", 20, 0)
    summary = log.summarize_by_job()
    assert summary["t"] == {"calls": 2, "prompt_tokens": 20,
                            "completion_tokens": 10, "reasoning_tokens": 8,
                            "cost_usd": pytest.approx(0.02)}
    assert summary["u"]["calls"] == 1
    assert summary["u"]["cost_usd"] == pytest.approx(0.02)


def test_default_max_tokens_injected_unless_given(tmp_path):
    from lab2rob.llm import DEFAULT_MAX_TOKENS
    client, transport, _ = make_client(tmp_path)
    client.complete("t", MESSAGES)
    assert transport.calls[0][2]["max_tokens"] == DEFAULT_MAX_TOKENS == 1024
    client.complete("t", MESSAGES, max_tokens=50)
    assert transport.calls[1][2]["max_tokens"] == 50


def test_reasoning_tokens_surfaced_and_cached(tmp_path):
    client, _, log = make_client(tmp_path, transport=FakeTransport(reasoning_tokens=48))
    first = client.complete("t", MESSAGES)
    second = client.complete("t", MESSAGES)
    assert (first.reasoning_tokens, second.reasoning_tokens) == (48, 48)
    entries = log.entries()
    assert [e["reasoning_tokens"] for e in entries] == [48, 48]
    assert entries[1]["cached"] is True


def test_finish_reason_surfaced_and_cached(tmp_path):
    client, _, _ = make_client(tmp_path, transport=FakeTransport(finish_reason="length"))
    first = client.complete("t", MESSAGES)
    second = client.complete("t", MESSAGES)
    assert (first.finish_reason, second.finish_reason) == ("length", "length")


def test_cost_log_survives_reopen(tmp_path):
    path = str(tmp_path / "c.jsonl")
    CostLog(path, prices=TEST_PRICES).record("t", "test-model", 10, 5)
    reopened = CostLog(path, prices=TEST_PRICES)
    assert reopened.spend_today() == pytest.approx(0.02)
    assert reopened.spend_for_job("t") == pytest.approx(0.02)


def _stub_openai(monkeypatch, responder):
    """Fake `openai` module: OpenAI() returns a client stub; responder serves calls."""
    calls = {}

    class StubEndpoint:
        def __init__(self, name):
            self.name = name

        def create(self, **kwargs):
            calls["endpoint"] = self.name
            calls["kwargs"] = kwargs
            return responder(**kwargs)

    class StubClient:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=StubEndpoint("chat"))
            self.responses = StubEndpoint("responses")

    monkeypatch.setitem(sys.modules, "openai",
                        SimpleNamespace(OpenAI=StubClient))
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    return calls


def _responses_reply(**kwargs):
    return SimpleNamespace(
        status="completed", output_text='{"points": []}',
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=10,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0)))


VISUAL_MESSAGES = [
    {"role": "system", "content": "Reply with JSON only."},
    {"role": "user", "content": [
        {"type": "text", "text": "The render is 640x480 pixels."},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,AAA"}}]},
]


def test_sam_model_uses_responses_api(tmp_path, monkeypatch):
    # ST-03: sam-3.1 404s on chat completions ("Use /v1/responses"), so the
    # transport must send it to responses with converted parts and params.
    from lab2rob.llm.client import OpenAITransport
    calls = _stub_openai(monkeypatch, _responses_reply)
    out = OpenAITransport().complete("sam-3.1", VISUAL_MESSAGES,
                                     max_tokens=2048)
    assert calls["endpoint"] == "responses"
    kwargs = calls["kwargs"]
    assert kwargs["model"] == "sam-3.1"
    assert kwargs["max_output_tokens"] == 2048
    assert "max_tokens" not in kwargs
    system, user = kwargs["input"]
    assert system == {"role": "system", "content": "Reply with JSON only."}
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "input_text", "text": "The render is 640x480 pixels."},
        {"type": "input_image", "image_url": "data:image/png;base64,AAA"}]
    assert out["text"] == '{"points": []}'
    assert (out["prompt_tokens"], out["completion_tokens"]) == (100, 10)
    assert out["finish_reason"] == "stop"


def test_chat_model_still_uses_chat_completions(tmp_path, monkeypatch):
    from lab2rob.llm.client import OpenAITransport

    def chat_reply(**kwargs):
        message = SimpleNamespace(content="hi")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1,
                                  completion_tokens_details=None))

    calls = _stub_openai(monkeypatch, chat_reply)
    out = OpenAITransport().complete("muse-spark-1.3-contributor",
                                     MESSAGES, max_tokens=50)
    assert calls["endpoint"] == "chat"
    assert calls["kwargs"]["messages"] == MESSAGES
    assert out["text"] == "hi"


def test_empty_response_neither_cached_nor_served(tmp_path):
    class EmptyTransport:
        def __init__(self):
            self.calls = 0

        def complete(self, model, messages, **kwargs):
            self.calls += 1
            return {"text": "", "prompt_tokens": 10, "completion_tokens": 5,
                    "reasoning_tokens": 0}

    client, transport, _ = make_client(tmp_path, transport=EmptyTransport())
    first = client.complete("t", MESSAGES)
    second = client.complete("t", MESSAGES)
    assert transport.calls == 2  # each call re-sent: empties never stick
    assert first.cached is False and second.cached is False
    cache = LLMCache(str(tmp_path / "cache2"))  # pre-fix poison heals as a miss
    cache.put("test-model", MESSAGES, {"text": "  ", "prompt_tokens": 10,
                                       "completion_tokens": 5})
    assert cache.get("test-model", MESSAGES) is None

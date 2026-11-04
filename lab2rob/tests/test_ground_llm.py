"""IG-05: LLM fallback resolves needs_llm steps (fake transport, offline)."""

from lab2rob.ground.llm_ground import (EFFORT_HEADROOM, FULL_CHUNK_STEPS,
                                       FULL_MAX_TOKENS_CAP,
                                       build_context, ground_protocol_llm,
                                       ground_step_llm)
from lab2rob.ground.rules import Grounding, ground_protocol
from lab2rob.ingest.bioprot import RawProtocol


class FakeReply:
    def __init__(self, text, completion_tokens=None, reasoning_tokens=0,
                 finish_reason=None):
        self.text = text
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = []
        self.kwargs = []

    def complete(self, job, messages, **kwargs):
        self.calls += 1
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        assert job == "ground_fallback"
        reply = self.replies.pop(0)
        return reply if isinstance(reply, FakeReply) else FakeReply(reply)


def make_step(text="Teleport the tube to Mars"):
    return Grounding(idx=1, source_text=text, status="needs_llm")


def test_llm_grounds_to_implemented_skill():
    client = FakeClient(['{"skill": "transfer", "out_of_scope_reason": null, '
                         '"ambiguous": false, "detail": "move between wells"}'])
    out = ground_step_llm(make_step(), client)
    assert (out.status, out.action, out.by) == ("grounded", "transfer", "llm")


def test_llm_declared_skill_becomes_not_implemented():
    client = FakeClient(['{"skill": "vortex", "out_of_scope_reason": null, '
                         '"ambiguous": false, "detail": "mix needed"}'])
    out = ground_step_llm(make_step("Vortex the tubes"), client)
    assert (out.status, out.reason) == ("out_of_scope", "not_implemented")


def test_llm_retry_then_ambiguous():
    client = FakeClient(["not json", '{"skill": "teleport", "out_of_scope_reason": null, '
                                     '"ambiguous": false, "detail": "x"}'])
    out = ground_step_llm(make_step(), client)
    assert client.calls == 2
    assert (out.status, out.reason, out.by) == ("ambiguous", "ambiguous_text", "llm")
    assert "failed twice" in out.detail  # unfixable step carries its reason


def test_build_context_marks_focus_step():
    context = build_context("demo", [(1, "Add dye"), (2, "Mix the tube")], focus=2)
    assert 'Protocol "demo"' in context
    assert ">> 2. Mix the tube" in context
    assert "   1. Add dye" in context or "  1. Add dye" in context


def test_full_context_sent_only_when_given():
    good = '{"skill": "transfer", "out_of_scope_reason": null, "ambiguous": false, "detail": "ok"}'
    plain = FakeClient([good])
    ground_step_llm(make_step(), plain)
    assert "Protocol context" not in plain.messages[0][1]["content"]
    with_context = FakeClient([good])
    ground_step_llm(make_step(), with_context, context="1. Add dye\n2. Mix")
    assert "Protocol context" in with_context.messages[0][1]["content"]
    assert "ground ONLY the Step" in with_context.messages[0][1]["content"]


def test_ground_protocol_threads_full_context():
    raw = RawProtocol(protocol_id="x", title="demo", description="",
                      steps_raw="1. Add 1 uL dye\n2. Teleport the tube\n",
                      source_url="u", license="l", number_of_steps=2)
    good = '{"skill": "transfer", "out_of_scope_reason": null, "ambiguous": false, "detail": "ok"}'
    plain = FakeClient([good])
    ground_protocol(raw, client=plain)
    assert "Protocol context" not in plain.messages[0][1]["content"]
    full = FakeClient([good])
    out = ground_protocol(raw, client=full, full_context=True).steps[1]
    assert out.by == "llm"
    user_text = full.messages[0][1]["content"]
    assert "Add 1 uL dye" in user_text and ">> 2. Teleport the tube" in user_text


def test_protocol_merge_uses_fallback_only_for_needs_llm():
    raw = RawProtocol(protocol_id="x", title="t", description="", steps_raw="1. Add 1 uL dye\n",
                      source_url="u", license="l", number_of_steps=1)
    grounding = ground_protocol(raw, client=FakeClient(["unexpected"]))
    assert grounding.steps[0].by == "rules"  # grounded by rules: no LLM call spent
    raw2 = RawProtocol(protocol_id="y", title="t", description="", steps_raw="1. Teleport the tube\n",
                       source_url="u", license="l", number_of_steps=1)
    client = FakeClient(['{"skill": null, "out_of_scope_reason": "missing_skill", '
                         '"ambiguous": false, "detail": "no teleporter"}'])
    out = ground_protocol(raw2, client=client).steps[0]
    assert (out.status, out.reason, out.by) == ("out_of_scope", "missing_skill", "llm")


def make_raw(steps="1. Add 1 uL dye to the well\n2. Teleport the tube to Mars\n"):
    return RawProtocol(protocol_id="x", title="demo", description="", steps_raw=steps,
                       source_url="u", license="l", number_of_steps=2)


def test_full_llm_translates_every_step_in_one_call():
    client = FakeClient(['[{"idx": 1, "skill": "transfer", "out_of_scope_reason": null, '
                         '"ambiguous": false, "detail": "pipette the dye"}, '
                         '{"idx": 2, "skill": null, "out_of_scope_reason": "missing_skill", '
                         '"ambiguous": false, "detail": "no teleporter"}]'])
    out = ground_protocol_llm(make_raw(), client)
    assert client.calls == 1
    first, second = out.steps
    assert (first.status, first.action, first.by) == ("grounded", "transfer", "llm")
    assert first.quantities[0].raw == "1 uL"  # rules extraction still rides along
    assert (second.status, second.reason, second.by) == (
        "out_of_scope", "missing_skill", "llm")


def test_full_llm_retry_then_all_ambiguous_with_reason():
    client = FakeClient(["not json", '[{"idx": 1}]'])
    out = ground_protocol_llm(make_raw(), client)
    assert client.calls == 2
    assert [(s.status, s.reason, s.by) for s in out.steps] == [
        ("ambiguous", "ambiguous_text", "llm")] * 2
    assert "failed twice" in out.steps[0].detail


def test_full_llm_empty_protocol_costs_no_call():
    client = FakeClient(["unexpected"])
    out = ground_protocol_llm(make_raw(""), client)
    assert out.steps == [] and client.calls == 0


def test_step_budget_is_visible_plus_minimal_headroom():
    client = FakeClient(['{"skill": "transfer", "out_of_scope_reason": null, '
                         '"ambiguous": false, "detail": "ok"}'])
    ground_step_llm(make_step(), client)
    assert EFFORT_HEADROOM == {"minimal": 1500, "low": 3000, "medium": 6000}
    assert client.kwargs == [{"max_tokens": 256 + 1500,
                              "reasoning_effort": "minimal",
                              "response_format": {"type": "json_object"}}]


def test_full_budget_scales_with_steps_and_caps():
    import json as _json

    def verdicts(idxs):
        return _json.dumps([{"idx": i, "skill": None,
                             "out_of_scope_reason": "missing_skill",
                             "ambiguous": False, "detail": "x"}
                            for i in idxs])

    def budget(n, effort="minimal"):
        return {"max_tokens": 128 + 128 * n + EFFORT_HEADROOM[effort],
                "reasoning_effort": effort}

    two = FakeClient([verdicts([1, 2])])
    ground_protocol_llm(make_raw(), two)
    assert two.kwargs == [budget(2)]
    # 60 steps go out in ceil(60/8) = 8 calls; no call ever hits the cap.
    chunks = [list(range(s, min(s + FULL_CHUNK_STEPS, 61))) for s in range(1, 61, FULL_CHUNK_STEPS)]
    big = FakeClient([verdicts(idxs) for idxs in chunks])
    ground_protocol_llm(make_raw("\n".join(f"{i}. Teleport the tube" for i in range(1, 61)) + "\n"), big)
    assert big.calls == 8
    assert big.kwargs == [budget(len(c)) for c in chunks]
    assert all(k["max_tokens"] <= FULL_MAX_TOKENS_CAP for k in big.kwargs)


def test_full_llm_chunks_long_protocols_with_global_idxs():
    import json as _json

    def verdicts(idxs, skill):
        return _json.dumps([{"idx": i, "skill": skill, "out_of_scope_reason": None,
                             "ambiguous": False, "detail": "ok"} for i in idxs])

    steps = "\n".join(f"{i}. Step number {i}" for i in range(1, 11)) + "\n"
    client = FakeClient([verdicts(range(1, 9), "transfer"), verdicts(range(9, 11), "aspirate")])
    out = ground_protocol_llm(make_raw(steps), client)
    assert client.calls == 2
    assert [s.idx for s in out.steps] == list(range(1, 11))
    assert [s.action for s in out.steps] == ["transfer"] * 8 + ["aspirate"] * 2
    assert all(s.by == "llm" for s in out.steps)


def test_full_llm_chunk_failure_isolates_to_its_steps():
    import json as _json

    good = _json.dumps([{"idx": i, "skill": "transfer", "out_of_scope_reason": None,
                         "ambiguous": False, "detail": "ok"} for i in range(1, 9)])
    steps = "\n".join(f"{i}. Step number {i}" for i in range(1, 11)) + "\n"
    client = FakeClient([good, "not json", "still bad"])
    out = ground_protocol_llm(make_raw(steps), client)
    assert client.calls == 3
    assert [s.status for s in out.steps] == ["grounded"] * 8 + ["ambiguous"] * 2
    assert "failed twice" in out.steps[8].detail


def test_failure_detail_reports_token_counts():
    client = FakeClient([FakeReply("not json", 100, 90), FakeReply("", 1024, 1022)])
    out = ground_step_llm(make_step(), client)
    assert "failed twice" in out.detail
    assert "1024 completion tokens, 1022 reasoning" in out.detail  # last call wins
    plain = FakeClient(["not json", "still not json"])
    out = ground_step_llm(make_step(), plain)
    assert "completion tokens" not in out.detail  # token-blind replies stay clean


def test_truncated_twice_marks_failed_with_reason():
    client = FakeClient([FakeReply("not json", 100, 90, "length"),
                         FakeReply("bad", 100, 90, "length")])
    out = ground_step_llm(make_step(), client)
    assert client.calls == 2
    assert (out.status, out.reason, out.by) == ("failed", "llm_truncated", "llm")
    assert "finish length" in out.detail
    assert "minimal->low" in out.detail


def test_truncation_escalates_once_then_succeeds():
    good = ('{"skill": "transfer", "out_of_scope_reason": null, '
            '"ambiguous": false, "detail": "ok"}')
    client = FakeClient([FakeReply('{"skill": "trans', 1756, 1700, "length"),
                         FakeReply(good, 500, 300, "stop")])
    out = ground_step_llm(make_step(), client)
    assert (out.status, out.action) == ("grounded", "transfer")
    assert client.kwargs == [
        {"max_tokens": 256 + 1500, "reasoning_effort": "minimal",
         "response_format": {"type": "json_object"}},
        {"max_tokens": 256 + 3000, "reasoning_effort": "low",
         "response_format": {"type": "json_object"}}]


def test_truncated_json_never_parsed_even_when_valid():
    good = ('{"skill": "transfer", "out_of_scope_reason": null, '
            '"ambiguous": false, "detail": "ok"}')
    client = FakeClient([FakeReply(good, 1756, 1500, "length"),
                         FakeReply(good, 500, 300, "stop")])
    out = ground_step_llm(make_step(), client)
    assert client.calls == 2  # the valid-looking cut-off reply was not accepted
    assert (out.status, out.action) == ("grounded", "transfer")


def test_chunk_truncated_twice_marks_its_steps_failed():
    import json as _json

    good = _json.dumps([{"idx": i, "skill": "transfer", "out_of_scope_reason": None,
                         "ambiguous": False, "detail": "ok"} for i in range(1, 9)])
    steps = "\n".join(f"{i}. Step number {i}" for i in range(1, 11)) + "\n"
    client = FakeClient([good,
                         FakeReply("bad", 2652, 2600, "length"),
                         FakeReply("bad", 4152, 4100, "length")])
    out = ground_protocol_llm(make_raw(steps), client)
    assert client.calls == 3
    assert [s.status for s in out.steps] == ["grounded"] * 8 + ["failed"] * 2
    assert [s.reason for s in out.steps[8:]] == ["llm_truncated"] * 2
    assert "minimal->low" in out.steps[8].detail

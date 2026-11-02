"""ST-01: coverage-gap queue groups unhandled steps (offline)."""

from lab2rob.coverage.gaps import build_queue, capability_for, write_queue


def step(idx, status, reason, text, detail=""):
    return {"idx": idx, "text": text, "status": status, "reason": reason,
            "action": None, "detail": detail}


def record(pid, steps):
    return {"protocol_id": pid, "dataset": "test", "steps": steps}


def test_queue_groups_by_reason_and_capability(tmp_path):
    records = [record("p1", [
        step(1, "out_of_scope", "missing_skill", "Spin at 3000 g"),
        step(2, "out_of_scope", "missing_skill", "Spin again fast"),
        step(3, "grounded", None, "Add 1 uL dye"),
        step(4, "ambiguous", "ambiguous_text", "Do the thing")])]
    queue = build_queue(records)
    assert queue["steps_total"] == 4
    assert queue["steps_queued"] == 3
    missing = queue["by_reason"]["missing_skill"]
    assert [(e["capability"], e["count"]) for e in missing] == [
        ("centrifuge", 2)]
    assert missing[0]["protocols"] == ["test:p1"]
    assert queue["by_reason"]["ambiguous_text"][0]["capability"] == "unclustered"
    assert queue["by_reason"]["missing_asset"] == []
    evolved = queue["evolve_candidates"]
    assert [(e["capability"], e["count"]) for e in evolved] == [
        ("centrifuge", 2)]  # ambiguous steps never become candidates


def test_refined_merge_remap_proposed_and_noise():
    records = [record("p1", [
        step(1, "out_of_scope", "missing_skill", "Divide the sample across tubes"),
        step(2, "out_of_scope", "missing_skill", "Mark every tube with a pen"),
        step(3, "ambiguous", "ambiguous_text", "QC looks fine")])]
    assert capability_for("Divide the sample across tubes", "", None)[0] == "unclustered"
    refined = [
        {"protocol_id": "test:p1", "step_idx": 1, "bucket": "existing",
         "capability": "aliquot", "detail": "split into tubes"},
        {"protocol_id": "test:p1", "step_idx": 2, "bucket": "new",
         "capability": "label_tubes", "detail": "mark tubes"},
        {"protocol_id": "test:p1", "step_idx": 3, "bucket": "noise",
         "capability": None, "detail": "chatter"},
    ]
    queue = build_queue(records, refined)
    missing = {e["capability"]: e
               for e in queue["by_reason"]["missing_skill"]}
    assert missing["aliquot"]["count"] == 1
    assert missing["label_tubes"]["proposed"] is True
    assert queue["by_reason"]["ambiguous_text"][0]["capability"] == "unresolved"
    evolved = {e["capability"]: e for e in queue["evolve_candidates"]}
    assert evolved["label_tubes"]["proposed"] is True
    assert "unresolved" not in evolved


def test_existing_remap_flags_keyword_fix():
    records = [record("p1", [
        step(1, "out_of_scope", "missing_skill", "Xyzzy the tubes")])]
    refined = [{"protocol_id": "test:p1", "step_idx": 1, "bucket": "existing",
                "capability": "aliquot", "detail": "split into tubes"}]
    queue = build_queue(records, refined)
    assert queue["by_reason"]["missing_skill"][0]["capability"] == "aliquot"
    fixes = queue["suggested_keyword_fixes"]
    assert len(fixes) == 1
    assert fixes[0]["capability"] == "aliquot"
    assert fixes[0]["remapped_steps"] == 1


def test_write_queue_files(tmp_path):
    queue = build_queue([record("p1", [
        step(1, "out_of_scope", "missing_skill", "Spin at 3000 g")])])
    write_queue(tmp_path, queue)
    assert (tmp_path / "queue.json").exists()
    text = (tmp_path / "queue.md").read_text(encoding="utf-8")
    assert "## Evolve candidates" in text
    assert "1. centrifuge: 1 steps" in text
    assert "### missing_skill (1 steps)" in text

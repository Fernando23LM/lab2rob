"""SE-03: golden-value metrics over sample harness runs (offline)."""

import pytest

from lab2rob.eval.metrics import compute_metrics, load_eval_dir, outcome_error


def run(eval_id, status="complete", sr=1.0, ssr=1.0, n_ops=2,
        protocol_id="t", reason="", outcome=None):
    record = {"eval_id": eval_id, "protocol_id": protocol_id,
              "status": status, "reason": reason, "n_ops": n_ops,
              "sr": sr, "ssr": ssr}
    if outcome is not None:
        record["outcome"] = outcome
    return record


def test_outcome_error_tolerances():
    assert outcome_error(2.0, 2.0, tol_rel=0.1)["passed"] is True
    row = outcome_error(2.2, 2.0, tol_rel=0.05, tol_abs=0.3)
    assert (row["abs_err"], row["passed"]) == (pytest.approx(0.2), True)
    assert outcome_error(3.0, 2.0, tol_rel=0.1)["passed"] is False
    assert outcome_error(3.0, 2.0)["passed"] is None  # no tolerance: no verdict


def test_golden_metrics_on_sample_runs():
    runs = [
        run("e1", sr=1.0, ssr=1.0,
            outcome={"reported": 2.0, "target": 2.0, "tol_rel": 0.1}),
        run("e2", sr=0.0, ssr=0.5, protocol_id="u",
            outcome={"reported": 3.0, "target": 2.0, "tol_rel": 0.1}),
        run("e3", status="failed", sr=0.0, ssr=None, n_ops=0,
            reason="sandbox_violation"),
        run("e4", sr=1.0, ssr=1.0, n_ops=0, protocol_id="u"),  # refused twin
    ]
    metrics = compute_metrics(
        runs,
        grounding_counts=[{"rules": 3, "llm": 1}, {"rules": 1, "llm": 1}],
        cost_summary={"ground_fallback": {"cost_usd": 0.94}},
        twin_expected={"e4": "refuse", "e2": "ask"})
    assert metrics["runs"] == 4
    assert metrics["by_status"] == {"complete": 3, "failed": 1}
    assert metrics["sr_mean"] == pytest.approx(2 / 3)  # complete runs only
    assert metrics["ssr_mean"] == pytest.approx(2.5 / 3)
    assert metrics["outcome"] == {"n": 2, "mean_abs_err": pytest.approx(0.5),
                                  "pass_rate": pytest.approx(0.5)}
    assert metrics["safety"]["violations"] == 1
    assert metrics["safety"]["violation_runs"] == ["e3"]
    assert metrics["safety"]["twins"]["refuse"] == {"n": 1, "complied": 1}
    assert metrics["safety"]["twins"]["ask"]["complied"] is None
    assert metrics["coverage"]["protocols"] == 2
    assert metrics["coverage"]["with_success"] == 2
    assert metrics["coverage"]["compile_mix"] == {"rules": pytest.approx(2 / 3),
                                                  "llm": pytest.approx(1 / 3)}
    assert metrics["cost_usd"] == pytest.approx(0.94)


def test_empty_inputs_stay_honest():
    metrics = compute_metrics([])
    assert metrics["runs"] == 0
    assert metrics["sr_mean"] is None and metrics["ssr_mean"] is None
    assert metrics["outcome"] == {"n": 0, "mean_abs_err": None,
                                  "pass_rate": None}
    assert metrics["safety"] == {"violations": 0, "violation_runs": [],
                                 "twins": {}}
    assert metrics["cost_usd"] is None


def test_load_eval_dir_reads_eval_json(tmp_path):
    eval_dir = tmp_path / "e1"
    eval_dir.mkdir()
    (eval_dir / "eval.json").write_text('{"eval_id": "e1", "sr": 1.0}',
                                        encoding="utf-8")
    assert load_eval_dir(eval_dir)["sr"] == 1.0

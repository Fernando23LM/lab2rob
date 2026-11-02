"""Tests for the rules-grounder scorer (IG-03). No LLM, no network."""

import json

from lab2rob.ground import score as S


def test_prf_math():
    assert S.prf(4, 0, 0)["f1"] == 1.0
    assert S.prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "tp": 0, "fp": 0, "fn": 0}
    partial = S.prf(1, 1, 3)
    assert (partial["precision"], partial["recall"]) == (0.5, 0.25)


def test_match_multisets_counts_duplicates():
    tp, fp, fn = S.match_multisets(["a", "a", "b"], ["a", "c"])
    assert (tp, fp, fn) == (1, ["c"], ["a", "b"])


def test_unit_aliases_fold_spellings():
    assert S.norm_quantity(25, "µl") == S.norm_quantity(25.0, "uL") == (25.0, "ul")
    assert S.norm_quantity(1, "mL") == (1.0, "ml")
    assert S.norm_quantity(5, "uM") == S.norm_quantity(5, "um") == (5.0, "uM")
    assert S.norm_quantity(37, "°C") == (37.0, "c")


def test_norm_entity_folds_case_and_space():
    assert S.norm_entity("  Alpaqua  384 Post Magnet Plate ") == "alpaqua 384 post magnet plate"


def test_sample_schema_and_gold_actions_are_canonical():
    from lab2rob.registry import lexicon_index
    sample = S.load_sample()
    assert len(sample["records"]) == 21
    canonicals = set(lexicon_index().values())
    for record in sample["records"]:
        assert {"id", "text", "gold", "provenance"} <= set(record)
        assert record["provenance"] == "human"
        assert set(record["gold"]["action"]) <= canonicals, record["id"]
        for qty in record["gold"]["quantities"]:
            float(qty["value"])
            assert qty["unit"].strip()


def test_score_runs_rules_only_with_consistent_counts():
    records = S.load_sample()["records"][:3]
    report = S.score_sample(records)
    assert report["n_steps"] == 3
    assert all(row["by"] != "llm" for row in report["rows"])
    assert all(row["provenance"] == {"gold": "human", "pred": "rules"}
               for row in report["rows"])
    assert 0.0 <= report["rules_only_share"] <= 1.0
    assert sum(report["status_counts"].values()) == 3
    for key in ("action", "entity", "quantity"):
        metrics = report[key]
        assert metrics["tp"] + metrics["fn"] >= metrics["tp"]


def test_committed_report_matches_sample():
    with open(S.DEFAULT_REPORT_PATH, encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["n_steps"] == len(S.load_sample()["records"]) == 21
    for key in ("action", "entity", "quantity", "rules_only_share",
                "status_counts", "rows"):
        assert key in report

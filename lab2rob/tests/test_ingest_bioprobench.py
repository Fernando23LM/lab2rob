"""BioProBench loader, the offline corpus sweep, and the registry parse cache."""

import io
import json

import pytest

from lab2rob.coverage import corpus_sweep
from lab2rob.ingest import bioprobench
from lab2rob.ingest.bioprobench import (CorpusProtocol, flatten_steps, load_corpus,
                                        record_to_protocol)
from lab2rob.registry import load_lexicon, load_skills
from lab2rob.registry.resolve import load_yaml, shared_yaml


def _record(pid="Protocol.io-7", url="https://www.protocols.io/view/x", steps=None):
    return {
        "id": pid, "url": f" {url} ", "title": "Protein standards",
        "abstract": "Make standards.", "description": "A DIFFERENT protocol's text.",
        "classification": {"primary_domain": "Elsewhere"},
        "hierarchical_protocol": steps if steps is not None else {
            "2": {"title": "Load"}, "1": {"title": "Prepare"},
            "1.10": "Step ten.", "1.2": "Step two.", "2.1": "Add 75 uL of stock to well A1.",
            "3": "A lone top-level step.", "note": "not a step key"},
    }


def test_flatten_orders_numerically_and_skips_sections():
    steps, keys, sections = flatten_steps(_record()["hierarchical_protocol"])
    assert keys == ["1.2", "1.10", "2.1", "3"]
    assert steps[0] == "Step two." and steps[-1] == "A lone top-level step."
    assert sections == {"1": "Prepare", "2": "Load"}


def test_record_uses_abstract_and_records_both_licenses():
    protocol = record_to_protocol(_record(), "protocolsio")
    raw = protocol.raw
    assert raw.description == "Make standards."  # never the misaligned description
    assert raw.dataset == "bioprobench_protocolsio"
    assert raw.source_url == "https://www.protocols.io/view/x"
    assert "CC-BY-NC-4.0" in raw.license and "CC-BY" in raw.license
    assert protocol.noncommercial
    assert raw.steps_raw.splitlines()[2] == "3. Add 75 uL of stock to well A1."
    assert protocol.step_keys[2] == "2.1"
    assert protocol.metadata["classification"] == {"primary_domain": "Elsewhere"}


def test_load_corpus_drops_duplicates_and_empty_protocols(tmp_path):
    records = [_record("a"), _record("b"),  # b repeats a's URL
               _record("c", url="https://www.protocols.io/view/y", steps={"1": {"title": "x"}}),
               _record("d", url="https://www.protocols.io/view/z")]
    path = tmp_path / "Protocol-io.json"
    path.write_text(json.dumps(records))
    protocols, stats = load_corpus({"protocolsio": path})
    assert [p.raw.protocol_id for p in protocols] == ["a", "d"]
    assert stats == {"records": 4, "duplicates": 1, "no_steps": 1}
    assert len(load_corpus({"protocolsio": path}, limit=1)[0]) == 1


def test_download_fetches_pinned_revision_once(tmp_path):
    urls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(url):
        urls.append(url)
        return Response(b"[]")

    paths = bioprobench.download(tmp_path, corpora=("protocolsio",), opener=opener)
    assert paths["protocolsio"].read_bytes() == b"[]"
    assert urls == [f"https://huggingface.co/datasets/{bioprobench.DATASET}/resolve/"
                    f"{bioprobench.REVISION}/Protocol-io.json"]
    bioprobench.download(tmp_path, corpora=("protocolsio",), opener=opener)
    assert len(urls) == 1


def test_segments_need_length_and_a_liquid_step():
    flags = [(True, False), (True, True), (True, False), (False, False),
             (True, True), (True, True)]
    assert corpus_sweep._segments(flags) == [(0, 3)]
    assert corpus_sweep._segments([(True, False)] * 5) == []


def test_greedy_unlock_picks_the_capability_that_completes_segments():
    def step(planned, liquid=False, blocker=None):
        return {"planned": planned, "liquid": liquid, "blocker": blocker}

    records = [
        {"steps": [step(True, True), step(False, blocker="centrifuge"), step(True)]},
        {"steps": [step(True, True), step(False, blocker="centrifuge"), step(True)]},
        {"steps": [step(True, True), step(False, blocker="vortex"), step(True)]},
        {"steps": [step(False, blocker="unclustered")]},
    ]
    unlock = corpus_sweep._greedy_unlock(records)
    assert unlock[0] == {"add": "centrifuge", "unlocks": 2, "protocols_with_segment": 2}
    assert unlock[1]["add"] == "vortex" and len(unlock) == 2


def _protocol(pid, steps):
    record = _record(pid, url=f"https://www.protocols.io/view/{pid}",
                     steps={str(i): text for i, text in enumerate(steps, 1)})
    return record_to_protocol(record, "protocolsio")


def test_sweep_compiles_reports_and_ranks(tmp_path):
    protocols = [
        _protocol("liquid", ["Add 75 uL of stock to well A1.",
                             "Centrifuge the plate at 1000 rpm for 1 min."]),
        _protocol("notes", ["Record the results in your notebook."]),
    ]
    summary = corpus_sweep.run(protocols, {"name": "test", "records": 2}, tmp_path,
                               workers=1)
    assert summary["protocols"] == 2 and summary["steps"] == 3
    assert summary["planned_steps"] == 1 and summary["liquid_steps"] == 1
    assert summary["n_candidates"] == 1
    assert summary["candidates"][0]["protocol_id"] == "liquid"
    assert summary["blockers_by_protocols"]["centrifuge"] == 1
    rows = [json.loads(line) for line in (tmp_path / "protocols.jsonl").open()]
    assert all("steps" not in r for r in rows)  # step text stays in steps.jsonl
    assert "## Unlock order" in (tmp_path / "report.md").read_text()


def test_compile_never_raises_on_a_broken_protocol(monkeypatch):
    def boom(raw):
        raise ValueError("bad")
    monkeypatch.setattr(corpus_sweep, "ground_protocol", boom)
    record = corpus_sweep.compile_protocol(_protocol("x", ["Add 1 uL."]))
    assert record == {"protocol_id": "x", "error": "ValueError: bad"}


def test_registry_cache_follows_edits_and_hands_out_copies(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("items: [1, 2]\n")
    first = load_yaml(path)
    first["items"].append(3)  # a private copy
    assert load_yaml(path) == {"items": [1, 2]}
    path.write_text("items: [9]\n")
    assert shared_yaml(path) == {"items": [9]}
    skills = load_skills()
    skills.clear()
    assert load_skills() and load_lexicon()


def test_corpus_protocol_defaults_to_noncommercial():
    assert CorpusProtocol(raw=None, corpus="protocolsio").noncommercial is True


@pytest.mark.parametrize("corpus", sorted(bioprobench.FILES))
def test_every_corpus_has_a_source_license(corpus):
    assert bioprobench.SOURCE_LICENSE[corpus]

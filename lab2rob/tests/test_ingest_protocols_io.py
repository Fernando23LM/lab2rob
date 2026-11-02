"""IG-01: protocols.io search/detail/cache logic (fixtures, offline)."""

import json
from pathlib import Path

import pytest

from lab2rob.ingest import protocols_io
from lab2rob.ingest.protocols_io import ProtocolsIOClient, fetch_many, list_cached

FIXTURES = Path(__file__).parent / "fixtures" / "protocols_io"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def make_client(detail_overrides=None):
    search = load_fixture("search.json")
    detail = load_fixture("detail.json")

    def fake_fetcher(url, params, headers):
        if url.endswith("/protocols"):
            return search
        payload = dict(detail["payload"])
        if "serial-dilution" in url:
            payload.update({"id": 333, "title": "Serial dilution of dyes",
                            "uri": "serial-dilution-of-dyes-cccccc",
                            "doi": "dx.doi.org/10.17504/protocols.io.cccccc",
                            "url": "https://www.protocols.io/view/serial-dilution-of-dyes-cccccc"})
        payload.update(detail_overrides or {})
        return {"payload": payload, "status_code": 0}

    return ProtocolsIOClient(fetcher=fake_fetcher, pause_s=0)


def test_search_parses_items_and_pagination():
    client = make_client()
    items, pagination = client.search("pipetting")
    assert [i["id"] for i in items] == [111, 222, 333]
    assert pagination["total_results"] == 3


def test_detail_parses_steps_and_license():
    client = make_client()
    payload = client.get_protocol(111)
    assert protocols_io.extract_steps(payload)[0] == "Set the pipette to 100 uL."
    assert protocols_io.extract_license(payload) == "CC-BY 4.0"
    assert protocols_io.is_cc_by("CC-BY 4.0") is True
    assert protocols_io.is_cc_by("CC-BY-NC") is True
    assert protocols_io.is_cc_by("All rights reserved") is False


def test_fetch_caches_with_provenance(tmp_path):
    result = fetch_many(make_client(), ["pipetting"], n=30, sources_dir=tmp_path)
    assert len(result["cached"]) == 2  # id 222 has 1 step (out of range)
    assert (tmp_path / "protocols_io_333.json").exists()
    record = json.loads((tmp_path / "protocols_io_111.json").read_text(encoding="utf-8"))
    assert record["uri"] == "pipetting-101-aaaaaa"
    assert record["doi"].endswith("protocols.io.aaaaaa")
    assert record["url"].startswith("https://www.protocols.io/view/")
    assert record["license"] == "CC-BY 4.0"
    assert record["version_id"] == 2
    assert record["retrieved_at"] and record["query"] == "pipetting"
    assert record["raw"]["title"] == "Pipetting 101"
    assert any(s["reason"].startswith("steps_out_of_range") for s in result["skipped"])


def test_fetch_is_idempotent_and_licensed_filtered(tmp_path):
    client = make_client()
    assert len(fetch_many(client, ["pipetting"], n=30, sources_dir=tmp_path)["cached"]) == 2
    assert fetch_many(client, ["pipetting"], n=30, sources_dir=tmp_path)["cached"] == []
    strict = fetch_many(make_client({"license": "ARR"}), ["other"], n=30,
                        sources_dir=tmp_path / "strict")
    assert strict["cached"] == []
    assert any("license_not_cc_by" in s["reason"] for s in strict["skipped"])


def test_fetch_aborts_on_auth_failure_anonymous(monkeypatch, tmp_path):
    monkeypatch.delenv(protocols_io.TOKEN_ENV, raising=False)
    calls = []

    def auth_fetcher(url, params, headers):
        calls.append(url)
        raise protocols_io.ProtocolsIOError("x needs authentication (403).")

    client = ProtocolsIOClient(fetcher=auth_fetcher, pause_s=0)
    result = fetch_many(client, ["a", "b", "c"], n=30, sources_dir=tmp_path)
    assert result["cached"] == []
    assert len(result["skipped"]) == 1
    assert len(calls) == 1


def test_fetch_retries_bare_then_aborts(tmp_path):
    calls = []

    def auth_fetcher(url, params, headers):
        calls.append(headers.get("Authorization"))
        raise protocols_io.ProtocolsIOError("x needs authentication (403).")

    client = ProtocolsIOClient(token="tok", fetcher=auth_fetcher, pause_s=0)
    result = fetch_many(client, ["a", "b", "c"], n=30, sources_dir=tmp_path)
    assert result["cached"] == []
    assert len(result["skipped"]) == 1
    assert calls == ["Bearer tok", "tok"]


def test_auth_falls_back_to_bare_token():
    seen = []

    def picky_fetcher(url, params, headers):
        seen.append(headers.get("Authorization"))
        assert headers["User-Agent"].startswith("lab2rob")
        if headers.get("Authorization") == "Bearer tok":
            raise protocols_io.ProtocolsIOError("u needs authentication (403).")
        return {"items": [], "pagination": {"total_pages": 1}}

    client = ProtocolsIOClient(token="tok", fetcher=picky_fetcher, pause_s=0)
    assert client.search("x")[0] == []
    client.search("y")
    assert seen == ["Bearer tok", "tok", "tok"]


def test_http_error_body_is_surfaced(monkeypatch):
    import io
    from urllib import error as urlerror

    def boom(req, timeout=30):
        raise urlerror.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": 1302}'))

    monkeypatch.setattr(protocols_io.urlrequest, "urlopen", boom)
    with pytest.raises(protocols_io.ProtocolsIOError, match="1302"):
        protocols_io._default_fetcher("https://x.test", {}, {})


def test_platform_default_license_and_detail_step_range(tmp_path):
    payload = dict(load_fixture("detail.json")["payload"])
    payload.pop("license", None)
    assert protocols_io.extract_license(payload).startswith("CC-BY")
    assert protocols_io.is_cc_by(protocols_io.extract_license(payload)) is True

    def one_step_fetcher(url, params, headers):
        if url.endswith("/protocols"):
            return load_fixture("search.json")
        slim = dict(load_fixture("detail.json")["payload"])
        slim["stats"] = {"number_of_steps": 1}
        slim["steps"] = [{"id": 1, "step": "Only step."}]
        return {"payload": slim, "status_code": 0}

    client = ProtocolsIOClient(fetcher=one_step_fetcher, pause_s=0)
    result = fetch_many(client, ["pipetting"], n=30, sources_dir=tmp_path)
    assert result["cached"] == []
    assert any("steps_out_of_range" in s["reason"] for s in result["skipped"])


def test_parse_identifier_forms():
    parse = protocols_io.parse_identifier
    assert parse("15421") == "15421"
    assert parse("pipetting-101-aaaaaa") == "pipetting-101-aaaaaa"
    assert parse("https://www.protocols.io/view/pipetting-101-aaaaaa") == "pipetting-101-aaaaaa"
    assert parse("https://www.protocols.io/view/x-y/?v=2") == "x-y"


def test_fetch_ids_direct(tmp_path):
    result = protocols_io.fetch_ids(make_client(), ["pipetting-101-aaaaaa"],
                                    sources_dir=tmp_path)
    assert len(result["cached"]) == 1
    assert result["cached"][0]["id"] == "111"
    assert (tmp_path / "protocols_io_111.json").exists()


def test_list_command_output(tmp_path, capsys):
    fetch_many(make_client(), ["pipetting"], n=30, sources_dir=tmp_path)
    rows = list_cached(tmp_path)
    assert len(rows) == 2
    assert rows[0]["steps"] == 4
    assert rows[0]["license"] == "CC-BY 4.0"


def test_clean_step_text_strips_markdown_and_folds_whitespace():
    clean = protocols_io.clean_step_text
    assert clean("See [the manual](https://example.com/m) first.") == "See the manual first."
    assert clean("Full notes: https://example.com/notes") == "Full notes:"
    assert clean("Aspirate 25 uL\n\ndye.") == "Aspirate 25 uL dye."
    assert clean("![img](https://x/y.png) Rinse.") == "Rinse."
    assert clean('Bring reagents: <table class="c"> <thead> <tr> %s</tr> </thead> </table>') == "Bring reagents: %s"
    assert clean("**Barcodes** suspended in water.") == "Barcodes suspended in water."
    assert clean("keep pH < 7 and fold > 5 intact") == "keep pH < 7 and fold > 5 intact"


def test_extract_main_steps_folds_substeps():
    record = json.loads((FIXTURES / "protocols_io_999.json").read_text(encoding="utf-8"))
    steps = protocols_io.extract_main_steps(record["raw"])
    assert len(steps) == 2  # empty step skipped, substep folded into step 1
    assert steps[0].startswith("Aspirate 25 uL dye from well A1.")
    assert "Work slowly near the well bottom." in steps[0]
    assert "manual" in steps[0] and "https://" not in steps[0]


def test_load_cached_protocol_returns_raw(tmp_path):
    raw = protocols_io.load_cached_protocol(999, FIXTURES)
    assert (raw.protocol_id, raw.dataset) == ("999", "protocolsio")
    assert raw.number_of_steps == 2
    assert raw.title == "Demo Pour"
    assert raw.license == "CC-BY 4.0"
    assert raw.source_url.startswith("https://www.protocols.io/view/")
    assert "room temperature" in raw.description
    assert raw.steps_raw.splitlines()[0].startswith("1. Aspirate")
    with pytest.raises(KeyError):
        protocols_io.load_cached_protocol(1, tmp_path)


def test_cached_protocol_grounds_end_to_end():
    from lab2rob.ground.rules import ground_protocol
    raw = protocols_io.load_cached_protocol(999, FIXTURES)
    grounding = ground_protocol(raw)
    assert (grounding.protocol_id, grounding.dataset) == ("999", "protocolsio")
    assert [s.action for s in grounding.steps] == ["aspirate", "dispense"]
    assert all(s.status == "grounded" for s in grounding.steps)

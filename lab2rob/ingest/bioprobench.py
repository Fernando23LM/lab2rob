"""BioProBench corpus loader: ~10.5k protocols with sectioned steps.

BioProBench (huggingface.co/datasets/BioProBench/BioProBench) ships two
protocol corpora, protocols.io (Protocol-io.json, ~9.6k) and Nature's
Protocol Exchange (Protocol-exchange.json, ~0.9k). Each record has a title,
an abstract, a URL and ``hierarchical_protocol``: keys "1", "1.1", "1.2",
"2", ... where an integer key holds a section ({"title": ...}) or a lone
step (a string) and a dotted key holds one step of its section.

What this loader trusts, and why:
- steps, title, url and abstract: checked to agree with each other;
- NOT description/problem/method/classification: for a sizeable share of
  records these describe a neighbouring protocol (an offset in the
  release), so they are kept as metadata only and never fed to grounding.

License: the compilation is CC BY-NC 4.0 (its Hugging Face card, checked
2026-11-03); protocols.io's public protocols are CC BY, Protocol Exchange
licenses are per article. Every RawProtocol records both, and
``noncommercial`` is set, so nothing from this corpus can slip into a
commercial dataset. The corpus itself is never committed: download() puts
it in a local cache (runs/ is gitignored).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from lab2rob.ingest.bioprot import RawProtocol
from lab2rob.ingest.protocols_io import clean_step_text

DATASET = "BioProBench/BioProBench"
#: Pinned dataset revision (the commit the loader was written against).
REVISION = "dec67450c8040250ea7751c7a3e77b3ac1e2e853"
FILES = {"protocolsio": "Protocol-io.json", "protocol_exchange": "Protocol-exchange.json"}
COMPILATION_LICENSE = "CC-BY-NC-4.0"
SOURCE_LICENSE = {"protocolsio": "CC-BY (protocols.io public content)",
                  "protocol_exchange": "per article (Protocol Exchange)"}
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "runs" / "corpora" / "bioprobench"

_KEY = re.compile(r"^\d+(\.\d+)*$")


@dataclass
class CorpusProtocol:
    """One BioProBench protocol: the RawProtocol grounding reads + provenance."""

    raw: RawProtocol
    corpus: str  # protocolsio | protocol_exchange
    step_keys: list = field(default_factory=list)  # step i (1-based) -> "2.3"
    sections: dict = field(default_factory=dict)  # "2" -> section title
    metadata: dict = field(default_factory=dict)  # untrusted fields, never parsed
    noncommercial: bool = True


def download(cache_dir: Path = DEFAULT_CACHE, corpora=tuple(FILES), opener=None) -> dict:
    """Fetch the corpus files at the pinned revision (skips files present)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    opener = opener or urllib.request.urlopen
    paths = {}
    for corpus in corpora:
        name = FILES[corpus]
        path = cache_dir / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
            tmp = path.with_suffix(".part")
            with opener(url) as response, open(tmp, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            tmp.rename(path)
        paths[corpus] = path
    return paths


def _key_order(key: str) -> tuple:
    return tuple(int(p) for p in key.split("."))


def flatten_steps(hierarchical: dict) -> tuple:
    """Sectioned steps -> (step texts in order, their keys, section titles).

    Section titles are context, not steps; a top-level string is a step of
    its own. Keys that are not "1", "1.2", ... are ignored (recorded by the
    caller as a skipped count).
    """
    keys = sorted((k for k in hierarchical if _KEY.match(str(k))), key=_key_order)
    steps, step_keys, sections = [], [], {}
    for key in keys:
        value = hierarchical[key]
        if isinstance(value, dict):
            title = clean_step_text(value.get("title", ""))
            if title:
                sections[key] = title
            continue
        text = clean_step_text(value)
        if text:
            steps.append(text)
            step_keys.append(key)
    return steps, step_keys, sections


def record_to_protocol(record: dict, corpus: str) -> CorpusProtocol:
    steps, step_keys, sections = flatten_steps(record.get("hierarchical_protocol") or {})
    protocol_id = str(record.get("id", "")).strip()
    abstract = clean_step_text(record.get("abstract") or "")
    raw = RawProtocol(
        protocol_id=protocol_id,
        title=clean_step_text(record.get("title") or ""),
        description=abstract,
        steps_raw="\n".join(f"{i}. {text}" for i, text in enumerate(steps, 1)),
        source_url=str(record.get("url") or "").strip(),
        license=f"{COMPILATION_LICENSE} (BioProBench); source: {SOURCE_LICENSE[corpus]}",
        number_of_steps=len(steps),
        dataset=f"bioprobench_{corpus}",
    )
    metadata = {k: record.get(k) for k in ("keywords", "classification", "description",
                                           "problem", "method", "length") if k in record}
    return CorpusProtocol(raw=raw, corpus=corpus, step_keys=step_keys,
                          sections=sections, metadata=metadata)


def load_corpus(paths: dict, limit: int | None = None, dedupe: bool = True) -> tuple:
    """(protocols, stats) from downloaded files; duplicates by URL dropped."""
    protocols, seen = [], set()
    stats = {"records": 0, "duplicates": 0, "no_steps": 0}
    for corpus in sorted(paths):
        records = json.loads(Path(paths[corpus]).read_text(encoding="utf-8"))
        for record in records:
            stats["records"] += 1
            protocol = record_to_protocol(record, corpus)
            key = protocol.raw.source_url.lower().rstrip("/") or protocol.raw.protocol_id
            if dedupe and key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            if not protocol.raw.number_of_steps:
                stats["no_steps"] += 1
                continue
            protocols.append(protocol)
            if limit is not None and len(protocols) >= limit:
                return protocols, stats
    return protocols, stats

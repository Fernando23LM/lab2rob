"""Fetch public protocols from protocols.io and cache them as versioned JSON.

API: ``GET {BASE_URL}/protocols`` (search) and ``GET {DETAIL_URL}/[id]``
(detail, v4), both documented at https://apidoc.protocols.io/. A token is
optional: set PROTOCOLS_IO_TOKEN if anonymous reads get 401/403. Every
cached file keeps protocol id, URI, DOI, URL, title, version, license,
retrieval date, the query that found it, and the raw detail payload.

Usage:
  python -m lab2rob.ingest.protocols_io fetch --n 30
  python -m lab2rob.ingest.protocols_io list
  python -m lab2rob.ingest.protocols_io load --all
  python -m lab2rob.ingest.protocols_io load 15421
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .bioprot import RawProtocol

BASE_URL = "https://www.protocols.io/api/v3"
DETAIL_URL = "https://www.protocols.io/api/v4/protocols"
TOKEN_ENV = "PROTOCOLS_IO_TOKEN"
#: Identify as a real client: faceless default UAs get 403s from the edge.
USER_AGENT = "lab2rob-research/1.0 (+https://github.com/Fernando23LM/Code-as-a-Policy-Evolving-Skill)"
SOURCES_DIR = Path(__file__).resolve().parent / "sources"

#: Search terms biased to what the simulator can do (context section 7.2).
SEARCH_KEYWORDS = [
    "pipetting",
    "serial dilution",
    "passaging",
    "pipette",
    "passage",
    "dilution",
    "dilute",
    "culture",
    "medium",
    "weigh",
    "weighing",
    "tube",
    "tubes",
    "plate",
    "plates",
    "well",
    "wells",
    "protein",
    "buffer",
    "buffers",
    "solution",
    "assay",
    "medium change",
    '"Bradford assay"',
    '"cell passaging"',
]

PAGE_SIZE = 20
MIN_STEPS = 2
MAX_STEPS = 30
REQUEST_PAUSE_S = 0.5


class ProtocolsIOError(RuntimeError):
    """The protocols.io API refused or failed a request."""


def _default_fetcher(url: str, params: dict, headers: dict) -> dict:
    full = url + "?" + urlparse.urlencode(params)
    req = urlrequest.Request(full, headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        if e.code in (401, 403):
            raise ProtocolsIOError(
                f"{url} needs authentication ({e.code}). Create a client access "
                f"token at protocols.io/developers and set {TOKEN_ENV}. {detail}"
            ) from None
        raise ProtocolsIOError(f"{url} failed with HTTP {e.code}: {detail}") from None
    except (urlerror.URLError, TimeoutError, OSError) as e:
        raise ProtocolsIOError(f"{url} unreachable: {e}") from None


class ProtocolsIOClient:
    """Thin wrapper over the public protocols.io API (fetcher injectable)."""

    def __init__(self, token=None, fetcher=None, pause_s: float = REQUEST_PAUSE_S):
        self.token = token or os.environ.get(TOKEN_ENV)
        self.fetcher = fetcher or _default_fetcher
        self.pause_s = pause_s
        # The docs show both `Bearer <token>` and a bare token; try Bearer
        # first, then fall back to bare once a 403 says otherwise.
        self._bare_auth = False

    def _auth_headers(self) -> dict:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = self.token if self._bare_auth else f"Bearer {self.token}"
        return headers

    def _get(self, url: str, params: dict) -> dict:
        try:
            try:
                return self.fetcher(url, params, self._auth_headers())
            except ProtocolsIOError as e:
                if self.token and not self._bare_auth and "needs authentication" in str(e):
                    self._bare_auth = True
                    return self.fetcher(url, params, self._auth_headers())
                raise
        finally:
            if self.pause_s:
                time.sleep(self.pause_s)

    def search(self, keyword: str, page_id: int = 1, page_size: int = PAGE_SIZE):
        """(items, pagination) for public protocols matching the keyword."""
        # NOTE: no order_field — sending one (even documented `relevance`)
        # currently trips a server-side SQL bug (1054, author_name in ORDER).
        data = self._get(f"{BASE_URL}/protocols",
                         {"filter": "public", "key": keyword,
                          "page_size": page_size, "page_id": page_id})
        self._check_status(data, "search")
        return data.get("items", []), data.get("pagination", {})

    def get_protocol(self, protocol_id, content_format: str = "markdown") -> dict:
        """Full protocol payload (steps as markdown) for one id/URI/DOI."""
        data = self._get(f"{DETAIL_URL}/{protocol_id}", {"content_format": content_format})
        self._check_status(data, f"protocol {protocol_id!r}")
        payload = data.get("payload", {})
        if not payload:
            raise ProtocolsIOError(f"Empty payload for protocol {protocol_id!r}")
        return payload

    @staticmethod
    def _check_status(data: dict, what: str) -> None:
        if data.get("status_code", 0) != 0:
            raise ProtocolsIOError(f"{what} failed: {data.get('error_message', data)}")


#: The API exposes no per-protocol license, but platform terms license all
#: public content CC-BY (protocols.io/terms + code-of-conduct), and we only
#: fetch filter=public. An explicit payload field still wins when present.
PLATFORM_DEFAULT_LICENSE = "CC-BY (protocols.io public-content default)"


def extract_license(payload: dict) -> str:
    for key in ("license", "license_name", "licence"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict) and value.get("name"):
            return str(value["name"]).strip()
    return PLATFORM_DEFAULT_LICENSE


def is_cc_by(license_text: str) -> bool:
    return license_text.strip().lower().startswith("cc-by")


def extract_steps(payload: dict) -> list[str]:
    return [str(s.get("step", "") or "") for s in payload.get("steps", [])
            if isinstance(s, dict)]


def cached_keys(sources_dir: Path = SOURCES_DIR) -> set:
    """(id, uri, doi) triples already on disk, for dedupe."""
    keys = set()
    if not sources_dir.is_dir():
        return keys
    for path in sorted(sources_dir.glob("protocols_io_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        keys.add((str(record.get("protocol_id", "")),
                  str(record.get("uri", "")), str(record.get("doi", ""))))
    return keys


def cache_protocol(payload: dict, query: str, sources_dir: Path = SOURCES_DIR) -> Path:
    """Write one cached record; returns the file path."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol_id": payload.get("id"),
        "uri": payload.get("uri", ""),
        "doi": payload.get("doi", ""),
        "url": payload.get("url", ""),
        "title": payload.get("title", ""),
        "version_id": payload.get("version_id"),
        "license": extract_license(payload),
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query,
        "raw": payload,
    }
    path = sources_dir / f"protocols_io_{record['protocol_id']}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def fetch_many(client: ProtocolsIOClient, keywords, n: int,
               sources_dir: Path = SOURCES_DIR,
               min_steps: int = MIN_STEPS, max_steps: int = MAX_STEPS,
               license_filter: str = "cc-by") -> dict:
    """Cache up to n protocols, round-robin over keywords. Idempotent."""
    cached, skipped, seen = [], [], set()
    keyword_stats: dict[str, dict] = {}
    for key in cached_keys(sources_dir):
        seen.add(key[0])
    uris = {key[1] for key in cached_keys(sources_dir)} | {key[2] for key in cached_keys(sources_dir)}

    def skip(pid, title, reason, query):
        skipped.append({"id": pid, "title": title, "reason": reason, "query": query})

    def consider(pid, title, uri, doi, query, stats, number_of_steps=None):
        """Fetch + filter + cache one candidate. Returns True if cached."""
        if not pid or pid in seen:
            return False
        seen.add(pid)
        stats["items"] += 1
        if (uri and uri in uris) or (doi and doi in uris):
            skip(pid, title, "duplicate_uri", query)
            return False
        if isinstance(number_of_steps, int) and not min_steps <= number_of_steps <= max_steps:
            skip(pid, title, f"steps_out_of_range ({number_of_steps})", query)
            return False
        try:
            payload = client.get_protocol(uri or pid)
        except ProtocolsIOError as e:
            skip(pid, title, f"fetch failed: {e}", query)
            return False
        detail_stats = payload.get("stats", {})
        n_detail = detail_stats.get("number_of_steps") if isinstance(detail_stats, dict) else None
        if not isinstance(n_detail, int):
            n_detail = len(extract_steps(payload))
        if not min_steps <= n_detail <= max_steps:
            skip(pid, title, f"steps_out_of_range ({n_detail})", query)
            return False
        if license_filter == "cc-by" and not is_cc_by(extract_license(payload)):
            skip(pid, title, f"license_not_cc_by ({extract_license(payload)})", query)
            return False
        path = cache_protocol(payload, query=query, sources_dir=sources_dir)
        cached.append({"id": pid, "title": str(payload.get("title", title)), "path": str(path)})
        stats["cached"] += 1
        uris.add(str(payload.get("uri", uri)))
        uris.add(str(payload.get("doi", doi)))
        return True

    for keyword in keywords:
        if len(cached) >= n:
            break
        stats = keyword_stats.setdefault(keyword, {"items": 0, "cached": 0})
        page_id = 1
        while len(cached) < n:
            try:
                items, pagination = client.search(keyword, page_id=page_id)
            except ProtocolsIOError as e:
                skip(None, "", f"search failed: {e}", keyword)
                if "needs authentication" in str(e):
                    return {"cached": cached, "skipped": skipped, "keywords": keyword_stats}
                break
            if not items:
                break
            for item in items:
                if len(cached) >= n:
                    break
                consider(str(item.get("id", "")), str(item.get("title", "")),
                         str(item.get("uri", "")), str(item.get("doi", "")), keyword, stats,
                         item.get("number_of_steps"))
            total_pages = (pagination or {}).get("total_pages", 1) or 1
            page_id += 1
            if page_id > total_pages:
                break
    return {"cached": cached, "skipped": skipped, "keywords": keyword_stats}


def parse_identifier(text: str) -> str:
    """Accept a bare id/URI/DOI or a full protocols.io URL; return the id part."""
    text = text.strip().split("?")[0].split("#")[0]
    if "/" in text:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return text


def fetch_ids(client: ProtocolsIOClient, identifiers, sources_dir: Path = SOURCES_DIR,
              min_steps: int = MIN_STEPS, max_steps: int = MAX_STEPS,
              license_filter: str = "cc-by") -> dict:
    """Cache protocols for explicit ids/URIs/URLs (browser-picked top-up)."""
    cached, skipped = [], []
    for raw in identifiers:
        ident = parse_identifier(raw)
        if not ident:
            continue
        try:
            payload = client.get_protocol(ident)
        except ProtocolsIOError as e:
            skipped.append({"id": ident, "title": "", "reason": f"fetch failed: {e}",
                            "query": "direct"})
            continue
        detail_stats = payload.get("stats", {})
        n_detail = detail_stats.get("number_of_steps") if isinstance(detail_stats, dict) else None
        if not isinstance(n_detail, int):
            n_detail = len(extract_steps(payload))
        if not min_steps <= n_detail <= max_steps:
            skipped.append({"id": ident, "title": str(payload.get("title", "")),
                            "reason": f"steps_out_of_range ({n_detail})", "query": "direct"})
            continue
        if license_filter == "cc-by" and not is_cc_by(extract_license(payload)):
            skipped.append({"id": ident, "title": str(payload.get("title", "")),
                            "reason": f"license_not_cc_by ({extract_license(payload)})",
                            "query": "direct"})
            continue
        path = cache_protocol(payload, query="direct", sources_dir=sources_dir)
        cached.append({"id": str(payload.get("id", ident)),
                       "title": str(payload.get("title", "")), "path": str(path)})
    return {"cached": cached, "skipped": skipped, "keywords": {"direct": {
        "items": len(identifiers), "cached": len(cached)}}}


def list_cached(sources_dir: Path = SOURCES_DIR) -> list[dict]:
    """One summary row per cached protocol."""
    rows = []
    if not sources_dir.is_dir():
        return rows
    for path in sorted(sources_dir.glob("protocols_io_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = record.get("raw", {}) if isinstance(record.get("raw"), dict) else {}
        stats = raw.get("stats", {}) if isinstance(raw.get("stats"), dict) else {}
        rows.append({
            "id": str(record.get("protocol_id", "")),
            "title": str(record.get("title", ""))[:60],
            "license": str(record.get("license", "")),
            "version": record.get("version_id"),
            "steps": stats.get("number_of_steps", len(raw.get("steps", []))),
            "query": str(record.get("query", "")),
        })
    return rows


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
#: Tag-like markup only (<letter...>): never eats comparisons like "< 7".
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
#: Emphasis runs (**bold**, __underline__); single * and _ are kept.
_EMPHASIS = re.compile(r"\*{2,}|_{2,}")
_BARE_URL = re.compile(r"https?://\S+")
_WS_RUN = re.compile(r"\s+")


def clean_step_text(text: str) -> str:
    """Markdown/whitespace cleanup: images drop out, links collapse to their
    label, HTML tags and emphasis runs drop out, bare URLs drop out, and
    whitespace runs (including newlines) fold so each step stays on one line
    and never splits into phantom steps."""
    text = str(text or "")
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = _HTML_TAG.sub("", text)
    text = _BARE_URL.sub("", text)
    text = _EMPHASIS.sub("", text)
    return _WS_RUN.sub(" ", text).strip()


def extract_main_steps(payload: dict) -> list[str]:
    """Main step texts in order; substeps fold into their parent step."""
    steps = []
    for item in payload.get("steps", []):
        if not isinstance(item, dict):
            continue
        text = clean_step_text(item.get("step", ""))
        if not text:
            continue
        if item.get("is_substep") and steps:
            steps[-1] += " " + text
        else:
            steps.append(text)
    return steps


def record_to_raw(record: dict) -> RawProtocol:
    """A cached protocols.io record -> the RawProtocol ground/plan consume."""
    raw = record.get("raw", {})
    raw = raw if isinstance(raw, dict) else {}
    steps = extract_main_steps(raw)
    description = clean_step_text(raw.get("description") or "")
    before_start = clean_step_text(raw.get("before_start") or "")
    if before_start:
        description = f"{description} {before_start}".strip()
    return RawProtocol(
        protocol_id=str(record.get("protocol_id", "")),
        dataset="protocolsio",
        title=str(record.get("title", "") or raw.get("title", "")),
        description=description,
        steps_raw="\n".join(f"{i}. {text}" for i, text in enumerate(steps, 1)),
        source_url=str(record.get("url", "") or raw.get("url", "")),
        license=str(record.get("license", "")),
        number_of_steps=len(steps),
    )


def load_cached_protocol(protocol_id, sources_dir: Path = SOURCES_DIR) -> RawProtocol:
    """Read one cached protocols.io file back into a RawProtocol."""
    path = Path(sources_dir) / f"protocols_io_{protocol_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise KeyError(f"protocols.io id {protocol_id!r} not cached in {sources_dir}") from None
    except ValueError as e:
        raise ProtocolsIOError(f"Cached file {path} is not valid JSON: {e}") from None
    return record_to_raw(record)


def iter_cached_protocols(sources_dir: Path = SOURCES_DIR):
    """Yield a RawProtocol per cached file, sorted by file name."""
    base = Path(sources_dir)
    paths = sorted(base.glob("protocols_io_*.json")) if base.is_dir() else []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        yield record_to_raw(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache protocols.io protocols.")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="search and cache protocols")
    fetch.add_argument("--n", type=int, default=30)
    fetch.add_argument("--keywords", default=",".join(SEARCH_KEYWORDS))
    fetch.add_argument("--sources-dir", default=str(SOURCES_DIR))
    fetch.add_argument("--min-steps", type=int, default=MIN_STEPS)
    fetch.add_argument("--max-steps", type=int, default=MAX_STEPS)
    fetch.add_argument("--license-filter", default="cc-by", choices=["cc-by", "any"])
    fetch.add_argument("--ids", default="",
                       help="comma-separated ids/URIs/URLs to fetch directly (skips search)")
    sub.add_parser("list", help="list cached protocols")
    load = sub.add_parser("load", help="read cached files back as RawProtocol")
    load.add_argument("protocol_id", nargs="?", default="",
                      help="cached id to show (omit with --all)")
    load.add_argument("--all", action="store_true",
                      help="one summary row per loadable cached file")
    load.add_argument("--sources-dir", default=str(SOURCES_DIR))
    args = parser.parse_args(argv)

    if args.command == "list":
        rows = list_cached()
        print(f"{len(rows)} cached protocols:")
        for row in rows:
            print(f"  {row['id']}  {row['license']}  steps={row['steps']}  {row['title']}")
        return 0

    if args.command == "load":
        sources = Path(args.sources_dir)
        if args.all:
            raws = list(iter_cached_protocols(sources))
            print(f"{len(raws)} loadable protocols:")
            for raw in raws:
                print(f"  {raw.protocol_id}  steps={raw.number_of_steps}  {raw.title[:60]}")
            return 0
        if not args.protocol_id:
            parser.error("load needs a protocol id or --all")
        raw = load_cached_protocol(args.protocol_id, sources)
        print(f"{raw.dataset}:{raw.protocol_id}  {raw.title}")
        print(f"  url: {raw.source_url}\n  license: {raw.license}")
        print(raw.steps_raw)
        return 0

    client = ProtocolsIOClient()
    if client.token:
        print(f"auth: token detected ({len(client.token)} chars)")
    else:
        print(f"auth: anonymous ({TOKEN_ENV} not set)")
    if args.ids.strip():
        result = fetch_ids(client, [i.strip() for i in args.ids.split(",") if i.strip()],
                           sources_dir=Path(args.sources_dir),
                           min_steps=args.min_steps, max_steps=args.max_steps,
                           license_filter=args.license_filter)
        print(f"cached {len(result['cached'])}, skipped {len(result['skipped'])}")
        for skip in result["skipped"]:
            print(f"  skip {skip['id']} [{skip.get('query', '')}] {skip['reason']}")
        return 0
    result = fetch_many(client, [k.strip() for k in args.keywords.split(",") if k.strip()],
                        n=args.n, sources_dir=Path(args.sources_dir),
                        min_steps=args.min_steps, max_steps=args.max_steps,
                        license_filter=args.license_filter)
    print(f"cached {len(result['cached'])}, skipped {len(result['skipped'])}")
    for keyword, counts in result.get("keywords", {}).items():
        print(f"  [{keyword}] {counts['items']} new items, {counts['cached']} cached")
    for skip in result["skipped"]:
        print(f"  skip {skip['id']} [{skip.get('query', '')}] {skip['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

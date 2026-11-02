"""Versioned dataset manifest: splits, provenance, licenses, episode labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


def build_manifest(dataset: str, task: str, protocol_id: str, source: dict,
                   episodes: list, train_ratio: float = 0.9,
                   extra: dict | None = None) -> dict:
    """Manifest payload over exported episodes.

    episodes are {"episode_index", "seed", "success", "failure_label",
    "frames"}. The first train_ratio share (by index) is train, the rest
    test. source carries url + license (rule 7).
    """
    episodes = sorted(episodes, key=lambda e: e["episode_index"])
    n_train = int(len(episodes) * train_ratio)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "task": task,
        "protocol_id": protocol_id,
        "source": {"url": source.get("url", ""),
                   "license": source.get("license", "")},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action_convention": "next_state",
        "granularity": "per-op",
        "splits": {
            "train": [e["episode_index"] for e in episodes[:n_train]],
            "test": [e["episode_index"] for e in episodes[n_train:]],
        },
        "episodes": episodes,
        "provenance": {"compiler": "lab2rob",
                       "exporter": "lab2rob.export.lerobot",
                       **(extra or {})},
    }


def validate_manifest(manifest: dict) -> list[str]:
    """Human-readable errors; [] means valid."""
    errors = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")
    for field in ("dataset", "task", "protocol_id"):
        if not manifest.get(field):
            errors.append(f"missing {field}")
    source = manifest.get("source", {})
    if not source.get("url"):
        errors.append("missing source.url")
    if not source.get("license"):
        errors.append("missing source.license")
    episodes = manifest.get("episodes", [])
    idxs = [e.get("episode_index") for e in episodes]
    if len(set(idxs)) != len(idxs):
        errors.append("episode_index values must be unique")
    for episode in episodes:
        for field in ("seed", "success", "frames"):
            if field not in episode:
                errors.append(f"episode {episode.get('episode_index')}: "
                              f"missing {field}")
        if episode.get("success") is False and not episode.get("failure_label"):
            errors.append(f"episode {episode.get('episode_index')}: failed "
                          f"episodes need a failure_label")
    splits = manifest.get("splits", {})
    split_idxs = sorted(splits.get("train", []) + splits.get("test", []))
    if split_idxs != sorted(i for i in idxs if i is not None):
        errors.append("splits must cover every episode exactly once")
    return errors


def save_manifest(out_dir: Path, manifest: dict) -> Path:
    path = Path(out_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_manifest(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))

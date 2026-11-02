"""XR-01: LeRobot export writes datasets + validating manifests."""

import json

import pytest

from lab2rob.build.scene import build_scene
from lab2rob.export.lerobot import (
    STATE_DIM,
    export_dataset,
    record_episode,
    validate_dataset,
)
from lab2rob.export.manifest import (
    build_manifest,
    validate_manifest,
)
from lab2rob.specify.program import PlannedProgram, PlannedStep

SOURCE = {"url": "https://example.test/p/1", "license": "CC-BY"}


def prog(ops=(("aspirate", {"volume_ul": 25.0}),
              ("dispense", {"volume_ul": 25.0}))):
    step = PlannedStep(idx=1, source_text="Move 25 uL", status="planned",
                       ops=[{"skill": s, "args": dict(a), "at": None,
                             "assumed": {}}
                            for s, a in ops])
    return PlannedProgram(protocol_id="t", title="Move 25 uL", steps=[step])


def test_record_episode_captures_state_images_subtask():
    episode = record_episode(prog(), seed=0, cameras=["overhead"],
                             width=64, height=48)
    assert episode["success"] is True
    assert [f["subtask"] for f in episode["frames"]] == [0, 1, 1]
    first = episode["frames"][0]
    assert len(first["state"]) == STATE_DIM == 14
    assert first["images"]["overhead"].shape == (48, 64, 3)


def test_record_episode_keeps_failure_with_label():
    episode = record_episode(prog(ops=(("teleport", {"where": "mars"}),)),
                             seed=0, cameras=["overhead"],
                             width=64, height=48)
    assert episode["success"] is False
    assert episode["failure_label"] == "no_executor"
    assert len(episode["frames"]) == 1  # initial frame kept


def test_export_small_dataset_validates(tmp_path):
    manifest = export_dataset(prog(), SOURCE, tmp_path / "ds", n_episodes=2,
                              cameras=["overhead"], width=64, height=48)
    assert validate_manifest(manifest) == []
    assert manifest["splits"] == {"train": [0], "test": [1]}
    assert manifest["source"] == SOURCE
    assert validate_dataset(tmp_path / "ds") == []
    import pyarrow.parquet as pq

    table = pq.read_table(
        tmp_path / "ds" / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == 3
    assert table.column("subtask_index").to_pylist() == [0, 1, 1]
    assert len(table.column("observation.state").to_pylist()[0]) == 14
    import imageio.v2 as imageio

    reader = imageio.get_reader(
        tmp_path / "ds" / "videos" / "chunk-000" / "observation.images.overhead"
        / "episode_000000.mp4")
    assert reader.count_frames() == 3
    reader.close()
    stats_rows = [json.loads(line) for line in
                  (tmp_path / "ds" / "meta" / "episodes_stats.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
    assert len(stats_rows) == 2
    assert stats_rows[0]["stats"]["frame_index"]["mean"] == [pytest.approx(1.0)]
    assert stats_rows[0]["stats"]["observation.state"]["count"] == [3]
    aggregate = json.loads((tmp_path / "ds" / "meta" / "stats.json")
                           .read_text(encoding="utf-8"))
    assert aggregate["frame_index"]["count"] == [6]


def test_converted_v30_loads_in_lerobot_loader(tmp_path, monkeypatch):
    import importlib.util

    if importlib.util.find_spec("lerobot.datasets.lerobot_dataset") is None:
        pytest.skip("needs the lerobot package")
    # Redirect before importing: datasets snapshots its cache config at import.
    cache = tmp_path / "hf_cache"  # lerobot's converter needs a writable cache
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache / "datasets"))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache / "hub"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lab2rob.export.lerobot import convert_to_v30
    export_dataset(prog(), SOURCE, tmp_path / "ds", n_episodes=2,
                   cameras=["overhead"], width=64, height=48)
    v30 = convert_to_v30(tmp_path / "ds")
    ds = LeRobotDataset("ds", root=str(v30))
    assert (ds.num_episodes, ds.num_frames) == (2, 6)
    item = ds[0]
    assert tuple(item["observation.state"].shape) == (14,)
    assert tuple(item["action"].shape) == (14,)
    assert "subtask_index" in item


def test_export_mixed_success_and_validation_catches_gaps(tmp_path):
    ok = prog()
    bad = prog(ops=(("teleport", {"where": "mars"}),))
    manifest = export_dataset(ok, SOURCE, tmp_path / "ds", n_episodes=1,
                              cameras=["overhead"], width=64, height=48)
    assert manifest["episodes"][0]["success"] is True
    manifest = export_dataset(bad, SOURCE, tmp_path / "bad", n_episodes=1,
                              cameras=["overhead"], width=64, height=48)
    assert manifest["episodes"][0]["success"] is False
    assert manifest["episodes"][0]["failure_label"] == "no_executor"
    assert validate_dataset(tmp_path / "bad") == []
    (tmp_path / "ds" / "data" / "chunk-000"
     / "episode_000000.parquet").unlink()
    errors = validate_dataset(tmp_path / "ds")
    assert any("missing episode_000000.parquet" in e for e in errors)


def test_manifest_rejects_bad_payloads():
    good = build_manifest("ds", "task", "t", SOURCE,
                          [{"episode_index": 0, "seed": 0, "success": True,
                            "failure_label": "", "frames": 3}])
    assert validate_manifest(good) == []
    assert good["action_convention"] == "next_state"
    no_license = build_manifest("ds", "task", "t", {"url": "u", "license": ""},
                                good["episodes"])
    assert any("license" in e for e in validate_manifest(no_license))
    unlabeled = build_manifest("ds", "task", "t", SOURCE,
                               [{"episode_index": 0, "seed": 0,
                                 "success": False, "failure_label": "",
                                 "frames": 1}])
    assert any("failure_label" in e for e in validate_manifest(unlabeled))
    bad_split = dict(good)
    bad_split["splits"] = {"train": [0, 1], "test": []}
    assert any("splits" in e for e in validate_manifest(bad_split))


def test_cli_exports_one_episode(tmp_path):
    from lab2rob.export.lerobot import main

    out = tmp_path / "cli"
    assert main(["--protocol", "bioprot:1779", "--n", "1",
                 "--out", str(out), "--no-v30"]) == 0
    assert validate_dataset(out) == []
    assert "1779" in (out / "manifest.json").read_text(encoding="utf-8")

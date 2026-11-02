"""LeRobot export: replay a reference program over seeds into a dataset.

One episode per seed: build the scene, execute the planned ops through
the certifier's executor, and record one frame per op (initial frame +
post-op frames). Each frame carries the bimanual joint state (6 joints
+ gripper opening per arm), the joint target for the next frame
(next-state-as-action: action[t] is the configuration the op between
t and t+1 servoed to; the final frame holds), camera images, and the
subtask (step) index. Failed episodes are kept with their abort reason
as failure_label.

Layout follows the LeRobot datasets spec (data/chunk-000/,
videos/chunk-000/<image-key>/, meta/), plus our manifest.json at the
root. Needs pyarrow + imageio (+ imageio-ffmpeg for mp4) at export
time; writing never needs the lerobot package itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab2rob.build.scene import build_scene
from lab2rob.certify.run import (
    RunAborted,
    compute_prefill,
    execute_op,
    prepare_env,
)
from lab2rob.export.manifest import build_manifest, save_manifest

CAMERAS = ["overhead", "front"]
WIDTH, HEIGHT = 320, 240
FPS = 2
CHUNK_SIZE = 1000

STATE_NAMES = ([f"left_j{i}" for i in range(1, 7)] + ["left_gripper"]
               + [f"right_j{i}" for i in range(1, 7)] + ["right_gripper"])
STATE_DIM = len(STATE_NAMES)


def _state_vector(env) -> list[float]:
    """14-dim bimanual state: 6 joints + gripper opening per arm."""
    left = [float(v) for v in env.data.qpos[env._qadr["left"]]]
    right = [float(v) for v in env.data.qpos[env._qadr["right"]]]
    return left + [float(env.ee_pose("left")["opening"])] + right + [
        float(env.ee_pose("right")["opening"])]


def _frames(env, api, grip_above: float, program, cameras, width, height):
    """Replay planned ops; yields (subtask_idx, state, images) per frame.

    The first frame is the initial state (subtask 0); each op appends its
    step's frame. Raises RunAborted like the certifier on failure.
    """
    yield 0, _state_vector(env), {
        cam: env.render(camera=cam, width=width, height=height)
        for cam in cameras}
    for step in program.steps:
        if step.status != "planned":
            continue
        for op_idx, op in enumerate(step.ops):
            slot = getattr(program, "_export_slots", {}).get(
                f"{step.idx}:{op_idx}")
            execute_op(api, env, op, slot, grip_above)
            yield step.idx, _state_vector(env), {
                cam: env.render(camera=cam, width=width, height=height)
                for cam in cameras}


def record_episode(program, seed: int, cameras=CAMERAS, width: int = WIDTH,
                   height: int = HEIGHT) -> dict:
    """One replay: frames + success flag + failure label (no files)."""
    built = build_scene(program, seed=seed)
    program._export_slots = dict(built.assignments)
    frames: list = []
    try:
        prefill = compute_prefill(program, built)
        env, api, grip_above = prepare_env(built, prefill, seed=seed)
        for sub, state, images in _frames(env, api, grip_above, program,
                                          cameras, width, height):
            frames.append({"subtask": sub, "state": state, "images": images})
        return {"seed": seed, "success": True, "failure_label": "",
                "frames": frames}
    except RunAborted as aborted:
        return {"seed": seed, "success": False,
                "failure_label": aborted.reason, "frames": frames}
    finally:
        if hasattr(program, "_export_slots"):
            del program._export_slots


def _chunk_dir(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // CHUNK_SIZE:03d}"


def _video_dir(root: Path, episode_index: int, key: str) -> Path:
    return (root / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}" / key)


def _feature_arrays(frames: list, episode_index: int, task_index: int,
                    fps: int, start_index: int) -> dict:
    """Numeric feature columns as (n_frames, dim) float lists, one source.

    Parquet writing and stats both read this, so the stats file can never
    disagree with the stored frames.
    """
    import numpy as np

    n = len(frames)
    states = np.array([f["state"] for f in frames], dtype=np.float64)
    actions = np.vstack([states[1:], states[-1]])  # last frame holds
    return {
        "observation.state": states,
        "action": actions,
        "subtask_index": np.array([[f["subtask"]] for f in frames],
                                  dtype=np.float64),
        "timestamp": (np.arange(n, dtype=np.float64) / fps).reshape(n, 1),
        "frame_index": np.arange(n, dtype=np.float64).reshape(n, 1),
        "episode_index": np.full((n, 1), float(episode_index)),
        "index": np.arange(start_index, start_index + n,
                           dtype=np.float64).reshape(n, 1),
        "task_index": np.full((n, 1), float(task_index)),
    }


def _write_parquet(path: Path, columns: dict) -> None:
    """One episode's feature columns -> parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "observation.state": pa.array(
            columns["observation.state"].astype("float32").tolist(),
            type=pa.list_(pa.float32(), STATE_DIM)),
        "action": pa.array(columns["action"].astype("float32").tolist(),
                           type=pa.list_(pa.float32(), STATE_DIM)),
        "subtask_index": pa.array(columns["subtask_index"].reshape(-1).astype("int64").tolist(),
                                 type=pa.int64()),
        "timestamp": pa.array(columns["timestamp"].reshape(-1).astype("float32").tolist(),
                              type=pa.float32()),
        "frame_index": pa.array(columns["frame_index"].reshape(-1).astype("int64").tolist(),
                                type=pa.int64()),
        "episode_index": pa.array(columns["episode_index"].reshape(-1).astype("int64").tolist(),
                                  type=pa.int64()),
        "index": pa.array(columns["index"].reshape(-1).astype("int64").tolist(),
                          type=pa.int64()),
        "task_index": pa.array(columns["task_index"].reshape(-1).astype("int64").tolist(),
                               type=pa.int64()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


#: v2.1 stats keys per feature (matches lerobot's RunningQuantileStats).
_QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90),
              ("q99", 0.99))


def _stats_for(columns: dict) -> dict:
    """{feature: {min/max/mean/std/count/quantiles}} as JSON-safe lists."""
    import numpy as np

    stats = {}
    for key, values in columns.items():
        entry = {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "count": [int(values.shape[0])],
        }
        for name, level in _QUANTILES:
            entry[name] = np.quantile(values, level, axis=0).tolist()
        stats[key] = entry
    return stats


def _write_videos(root: Path, episode_index: int, frames: list, cameras: list,
                  fps: int, width: int, height: int):
    import imageio.v2 as imageio

    for pos, camera in enumerate(cameras):
        key = f"observation.images.{camera}"
        path = (_video_dir(root, episode_index, key)
                / f"episode_{episode_index:06d}.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=fps, macro_block_size=16)
        try:
            for frame in frames:
                writer.append_data(frame["images"][camera])
        finally:
            writer.close()


def _write_meta(root: Path, episodes: list, task: str, cameras: list,
                fps: int, width: int, height: int, total_frames: int):
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM],
                              "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
        "subtask_index": {"dtype": "int64", "shape": [1],
                          "names": ["subtask_index"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
        "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
        "episode_index": {"dtype": "int64", "shape": [1],
                          "names": ["episode_index"]},
        "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channel"]}
    n_train = int(len(episodes) * 0.9)
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "lab2rob_yam_bimanual",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(cameras),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_train}", "test": f"{n_train}:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }, indent=2), encoding="utf-8")
    with open(meta / "tasks.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with open(meta / "episodes.jsonl", "w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps({
                "episode_index": episode["episode_index"],
                "tasks": [task],
                "length": episode["frames"],
                "success": episode["success"],
                "failure_label": episode["failure_label"],
            }) + "\n")


def export_dataset(program, source: dict, out_dir, n_episodes: int = 100,
                   start_seed: int = 0, dataset: str | None = None,
                   cameras=CAMERAS, width: int = WIDTH, height: int = HEIGHT,
                   fps: int = FPS) -> dict:
    """Replay a program over seeds into a LeRobot dataset + manifest.

    source carries url + license (rule 7). Returns the manifest payload.
    """
    root = Path(out_dir)
    dataset = dataset or f"{program.protocol_id}_lerobot"
    task = "\n".join(f"{s.idx}. {s.source_text}" for s in program.steps
                     if s.status == "planned")
    episodes_meta, episodes_stats, index = [], [], 0
    for episode_index in range(n_episodes):
        seed = start_seed + episode_index
        episode = record_episode(program, seed, cameras, width, height)
        frames = episode["frames"] or []
        if frames:
            columns = _feature_arrays(frames, episode_index, 0, fps, index)
            path = (_chunk_dir(root, episode_index)
                    / f"episode_{episode_index:06d}.parquet")
            _write_parquet(path, columns)
            _write_videos(root, episode_index, frames, cameras, fps,
                          width, height)
            episodes_stats.append({"episode_index": episode_index,
                                   "stats": _stats_for(columns)})
            index += len(frames)
        episodes_meta.append({"episode_index": episode_index, "seed": seed,
                              "success": episode["success"],
                              "failure_label": episode["failure_label"],
                              "frames": len(frames)})
    total_frames = sum(e["frames"] for e in episodes_meta)
    _write_meta(root, episodes_meta, task, cameras, fps, width, height,
                total_frames)
    _write_stats(root, episodes_stats)
    manifest = build_manifest(dataset, task, program.protocol_id, source,
                              episodes_meta)
    save_manifest(root, manifest)
    return manifest


def _write_stats(root: Path, episodes_stats: list) -> None:
    """v2.1 stats files: per-episode jsonl + aggregated stats.json."""
    import numpy as np

    meta = root / "meta"
    with open(meta / "episodes_stats.jsonl", "w", encoding="utf-8") as handle:
        for row in episodes_stats:
            handle.write(json.dumps(row) + "\n")
    aggregate: dict = {}
    if episodes_stats:
        keys = episodes_stats[0]["stats"]
        for key in keys:
            stacked = [np.array(row["stats"][key]["mean"]) *
                       row["stats"][key]["count"][0]
                       for row in episodes_stats]
            counts = [row["stats"][key]["count"][0]
                      for row in episodes_stats]
            total = sum(counts)
            mean = sum(stacked) / total
            var = sum((np.array(row["stats"][key]["std"]) ** 2 + (
                np.array(row["stats"][key]["mean"]) - mean) ** 2)
                * row["stats"][key]["count"][0]
                for row in episodes_stats) / total
            aggregate[key] = {
                "min": np.min([row["stats"][key]["min"]
                               for row in episodes_stats], axis=0).tolist(),
                "max": np.max([row["stats"][key]["max"]
                               for row in episodes_stats], axis=0).tolist(),
                "mean": mean.tolist(),
                "std": np.sqrt(np.maximum(0, var)).tolist(),
                "count": [total],
            }
    (meta / "stats.json").write_text(json.dumps(aggregate, indent=2),
                                     encoding="utf-8")


def convert_to_v30(out_dir) -> Path:
    """Convert a v2.1 export to v3.0 with lerobot's own converter.

    Needs the lerobot package (pip install lerobot). The converter works
    in place: the v3.0 dataset ends up at out_dir, the v2.1 files move to
    <name>_old next to it. Our manifest.json is copied back onto the v3.0
    tree (episode indices are stable across the conversion). Returns the
    v3.0 directory (== out_dir).
    """
    try:
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
            convert_dataset,
        )
    except ImportError:
        raise RuntimeError(
            "Converting to v3.0 needs the lerobot package: pip install lerobot"
        ) from None
    out = Path(out_dir)
    stats_path = out / "meta" / "episodes_stats.jsonl"
    have_stats = set()
    if stats_path.exists():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                have_stats.add(json.loads(line)["episode_index"])
    manifest_episodes = json.loads((out / "manifest.json").read_text(
        encoding="utf-8"))["episodes"]
    frameless = [e["episode_index"] for e in manifest_episodes
                 if e["episode_index"] not in have_stats]
    if frameless:
        raise RuntimeError(
            f"Cannot convert: episodes {frameless} have no frames "
            f"(env failed before recording). Re-export without them.")
    convert_dataset(repo_id=out.name, root=str(out.parent),
                    push_to_hub=False)
    old_manifest = out.parent / f"{out.name}_old" / "manifest.json"
    if old_manifest.exists() and not (out / "manifest.json").exists():
        (out / "manifest.json").write_text(
            old_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def validate_dataset(root) -> list[str]:
    """Offline checks (no lerobot import): layout + counts + manifest."""
    import pyarrow.parquet as pq

    from lab2rob.export.manifest import load_manifest, validate_manifest

    errors = []
    root = Path(root)
    try:
        manifest = load_manifest(root)
    except (OSError, ValueError) as e:
        return [f"manifest unreadable: {e}"]
    errors.extend(validate_manifest(manifest))
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return errors + [f"meta/info.json unreadable: {e}"]
    total_frames = 0
    for episode in manifest.get("episodes", []):
        idx, want_frames = episode["episode_index"], episode["frames"]
        if want_frames == 0:
            continue  # frameless episodes keep the manifest entry only
        path = (_chunk_dir(root, idx) / f"episode_{idx:06d}.parquet")
        if not path.exists():
            errors.append(f"episode {idx}: missing {path.name}")
            continue
        rows = pq.read_table(path).num_rows
        total_frames += rows
        if rows != want_frames:
            errors.append(f"episode {idx}: parquet has {rows} rows, "
                          f"manifest says {want_frames}")
    if total_frames != info.get("total_frames"):
        errors.append(f"info total_frames {info.get('total_frames')} != "
                      f"parquet total {total_frames}")
    if len(manifest.get("episodes", [])) != info.get("total_episodes"):
        errors.append("info total_episodes != manifest episode count")
    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Export a LeRobot dataset")
    parser.add_argument("--protocol", default="bioprot:1779",
                        help="bioprot:<id> reference task to replay")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="runs/lerobot_export")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--v30", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="convert the export to v3.0 (needs lerobot)")
    args = parser.parse_args(argv)
    from lab2rob.ground.rules import ground_protocol
    from lab2rob.ingest.bioprot import load_bioprot_id
    from lab2rob.specify.program import plan_protocol

    kind, _, ident = args.protocol.partition(":")
    if kind != "bioprot":
        print(f"only bioprot:<id> protocols are supported, got {args.protocol!r}")
        return 1
    raw = load_bioprot_id(ident)
    program = plan_protocol(ground_protocol(raw))
    manifest = export_dataset(
        program, {"url": raw.source_url, "license": raw.license},
        args.out, n_episodes=args.n, start_seed=args.start_seed)
    succeeded = sum(1 for e in manifest["episodes"] if e["success"])
    print(f"exported {len(manifest['episodes'])} episodes "
          f"({succeeded} succeeded) to {args.out}")
    errors = validate_dataset(args.out)
    if errors:
        print("validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("dataset validates")
    if args.v30:
        try:
            v30 = convert_to_v30(args.out)
        except RuntimeError as e:
            print(f"v3.0 conversion skipped: {e}")
            return 0
        print(f"v3.0 dataset at {v30}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

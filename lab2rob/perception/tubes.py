"""ST-03: tube poses from camera images for the tube_rack task.

Replaces the ground-truth tube entries of perceive() with vision-derived
estimates: park the right arm for a clear overhead view, find the caps as
closest-point blobs in overhead depth, deproject the closest pixel of each
with metric depth, and report 3D tops. Two segmentation backends share one
interface: a deterministic depth backend (offline) and a vision-model backend
(job "visual", needs MODEL_API_KEY).

Also evaluates: pose error against ground truth over seeds, plus a
vision-driven grasp check (approach the estimated top, close, lift) next
to the same check driven by truth, so the accuracy gap is a grasp rate,
not just millimetres.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

_BIOLAB_ROOT = Path(__file__).resolve().parents[2] / "Code-as-a-Policy-Evolving-Skill"

CAP_DIAMETER_PX = 15
LOCAL_MEDIAN_PX = 45
PROMINENCE_M = 0.008
#: Floor, not a target: a fully visible cap is ~200 plateau pixels, but an
#: oblique view can shrink one to ~10 (seed 3's blue cap). The tube-height
#: band and prominence ranking do the real filtering; this only drops
#: 1-2 px depth noise.
MIN_PIXELS = 10
BORDER_MARGIN_PX = 20
FAR_PLANE_M = 1.5
TUBE_Z_MIN = 0.05
TUBE_Z_MAX = 0.12
WORKSPACE_XY_M = 0.8
MERGE_DIST_M = 0.02
CAMERAS = ("overhead", "front")
DEPTH_WIN = 5
PROPOSE_MAX_TOKENS = 2048
GRASP_LIFT_M = 0.02
TRAVEL_Z = 0.152
#: Right-gripper pinch point folded low near its base: reachable on every
#: seed, and the whole arm stays ~20 cm off the overhead sightline to the
#: racks, so parked captures show both tubes.
PARK_POS = (-0.10, -0.28, 0.14)
PARK_YAW = -1.52
#: Snap half-window: regressed cap pixels land up to ~25 px off the cap,
#: so each is re-centred on the nearest cap plateau in a ±35 px crop.
SNAP_HALF_PX = 35

VISION_SYSTEM = """Two vials sit in the teal source rack, one with a RED cap and one
with a BLUE cap (tube_0 is red, tube_1 is blue, on every episode). Reply
with exactly one JSON object, no other text:
{"points": [{"u": <int>, "v": <int>}, ...]} with the centre pixel of the
red cap first, then the blue cap centre (u right, v down, origin top
left). Aim for the middle of each cap disk, not above it."""


def _biolab():
    root = str(_BIOLAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from biolab import env as biolab_env

    return biolab_env


@contextlib.contextmanager
def parked_view(env, api=None):
    """Fold the right arm aside for an unoccluded capture, then go home.

    At its resting pose the right arm hangs over the source rack and hides
    a tube from the overhead camera on most seeds; no backend can see
    through it. Parking is active perception with no net state change: the
    arm ends back home and nothing else is touched.
    """
    if api is None:
        _biolab()
        from biolab.api import TubeRackApi
        api = TubeRackApi(env)
    if not api.move_to("right", np.array(PARK_POS), PARK_YAW):
        raise RuntimeError("could not park the right arm for a clear view")
    try:
        yield api
    finally:
        api.go_home("right")


def segment_tube_tops(depth: np.ndarray, intrinsics: np.ndarray,
                        pose_mat: np.ndarray, n_tubes: int = 2,
                        prominence_m: float = PROMINENCE_M,
                        border_margin_px: int = BORDER_MARGIN_PX) -> list:
    """Closest-point blobs (tube caps) as [{u, v, pixels, depth, pos}].

    Tube caps are flat disks standing ~75 mm proud of the rack, so from
    overhead each cap is a plateau of local depth minima. Candidates are
    minima standing clear of their surroundings by prominence_m, ranked by
    prominence; the top n_tubes win. Color is deliberately unused: cap
    appearance varies with viewing angle (dark cap tops, translucent
    walls), while geometry does not. Border pixels (filter artifacts) and
    far-plane pixels (past the table) are excluded up front, and only
    blobs deprojecting into the tube-height band survive, which drops rack
    surfaces below and homed grippers above. Crops pass border_margin_px=0:
    the margin guards full-image filter edges, not crop edges.
    """
    depth = np.asarray(depth, dtype=float)
    depth = depth[..., 0] if depth.ndim == 3 else depth
    rows, cols = depth.shape
    valid = np.ones_like(depth, dtype=bool)
    if border_margin_px:
        valid[:border_margin_px, :] = False
        valid[-border_margin_px:, :] = False
        valid[:, :border_margin_px] = False
        valid[:, -border_margin_px:] = False
    valid &= depth < FAR_PLANE_M
    local_min = ndimage.minimum_filter(depth, size=CAP_DIAMETER_PX) == depth
    background = ndimage.median_filter(depth, size=LOCAL_MEDIAN_PX)
    sure = (valid & local_min & (depth < background - prominence_m)
            & (background < FAR_PLANE_M))
    labeled, count = ndimage.label(sure)
    blobs = []
    for blob_id in range(1, count + 1):
        ys, xs = np.argwhere(labeled == blob_id).T
        if len(xs) < MIN_PIXELS:
            continue
        # Centroid, not the first minimum: argmin returns the first plateau
        # pixel in scan order (a cap edge, ~15 mm off), while the centroid
        # of the flat top is the cap centre the grasp needs.
        u, v = float(np.mean(xs)), float(np.mean(ys))
        cap_depth = float(np.median(depth[ys, xs]))
        pos = deproject(u, v, cap_depth, intrinsics, pose_mat)
        if not (TUBE_Z_MIN <= pos[2] <= TUBE_Z_MAX):
            continue
        blobs.append({"u": u, "v": v, "pixels": len(xs),
                      "depth": cap_depth, "pos": pos,
                      "prominence": float(np.median(background[ys, xs])
                                          - cap_depth)})
    blobs.sort(key=lambda b: -b["prominence"])
    return blobs[:max(0, n_tubes)]


def deproject(u: float, v: float, depth_m: float, intrinsics: np.ndarray,
              pose_mat: np.ndarray) -> list:
    """Pixel + metric depth to world XYZ (OpenCV camera axes)."""
    point = np.linalg.inv(np.asarray(intrinsics, dtype=float)) @ np.array([u, v, 1.0])
    cam = np.asarray(depth_m, dtype=float) * point
    world = np.asarray(pose_mat, dtype=float) @ np.array([*cam, 1.0])
    return [float(x) for x in world[:3]]


def _window_depth(depth: np.ndarray, u: float, v: float) -> float:
    rows, cols = depth.shape[:2]
    v0, v1 = max(0, int(v) - DEPTH_WIN), min(rows, int(v) + DEPTH_WIN + 1)
    u0, u1 = max(0, int(u) - DEPTH_WIN), min(cols, int(u) + DEPTH_WIN + 1)
    return float(np.median(depth[v0:v1, u0:u1]))


def snap_to_caps(pixels: list, obs: dict,
                 half_px: int = SNAP_HALF_PX) -> list:
    """Re-centre each regressed pixel on the nearest cap plateau.

    Vision models find the right neighbourhood but not the pixel: a point
    ~25 px off a cap reads background depth and fails the lift. Each pixel
    is re-segmented in a ±half_px crop (same geometry as the depth
    backend, principal point shifted to crop origin) and moved to the
    nearest blob centre; pixels with no cap in reach are dropped.
    """
    depth = np.asarray(obs["images"]["depth"])
    depth = depth[..., 0] if depth.ndim == 3 else depth
    rows, cols = depth.shape
    base_k = np.asarray(obs["intrinsics"], dtype=float)
    out = []
    for point in pixels:
        u0 = max(0, int(point["u"]) - half_px)
        u1 = min(cols, int(point["u"]) + half_px + 1)
        v0 = max(0, int(point["v"]) - half_px)
        v1 = min(rows, int(point["v"]) + half_px + 1)
        crop = depth[v0:v1, u0:u1]
        if crop.size == 0:
            continue
        intrinsics = base_k.copy()
        intrinsics[0, 2] -= u0
        intrinsics[1, 2] -= v0
        blobs = segment_tube_tops(crop, intrinsics, obs["pose_mat"],
                                  n_tubes=2, border_margin_px=0)
        if not blobs:
            continue
        best = min(blobs, key=lambda b: (b["u"] + u0 - point["u"]) ** 2
                   + (b["v"] + v0 - point["v"]) ** 2)
        out.append({"u": best["u"] + u0, "v": best["v"] + v0})
    return out


class DepthBackend:
    """Deterministic cap-top geometry; needs no model and no key."""

    name = "depth"

    def __init__(self, n_tubes: int = 2):
        self.n_tubes = n_tubes

    def points(self, obs: dict) -> list:
        return [{"u": b["u"], "v": b["v"]} for b in segment_tube_tops(
            obs["images"]["depth"], obs["intrinsics"], obs["pose_mat"],
            n_tubes=self.n_tubes)]


class VisionBackend:
    """Vision model (job 'visual') regresses cap pixels, snapped to caps."""

    name = "vision"

    def __init__(self, client=None):
        self.client = client
        self.calls = 0

    def _client(self):
        if self.client is None:
            from lab2rob.llm import Lab2RobClient
            self.client = Lab2RobClient()
        return self.client

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required to send renders") from None
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
        return buffer.getvalue()

    def _messages(self, rgb: np.ndarray, width: int, height: int, error: str = "") -> list:
        blob = base64.b64encode(self._encode_png(rgb)).decode("ascii")
        return [{"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"The render is {width}x{height} pixels (u right, v down)."
                             + error},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{blob}"}}]}]

    @staticmethod
    def _parse(text: str, width: int, height: int) -> list:
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"not JSON: {e}") from None
        points = data.get("points") if isinstance(data, dict) else None
        if not isinstance(points, list) or not points:
            raise ValueError("points must be a non-empty list")
        out = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("each point must be an object")
            u, v = point.get("u"), point.get("v")
            if (not isinstance(u, int) or not isinstance(v, int)
                    or not (0 <= u < width and 0 <= v < height)):
                raise ValueError(f"bad pixel {point!r} for a {width}x{height} render")
            out.append({"u": float(u), "v": float(v)})
        return out

    def points(self, obs: dict) -> list:
        rgb = np.asarray(obs["images"]["rgb"])
        height, width = rgb.shape[:2]
        error = ""
        for _ in range(2):
            self.calls += 1
            text = self._client().complete(
                "visual", self._messages(rgb, width, height, error),
                max_tokens=PROPOSE_MAX_TOKENS, temperature=0.0).text
            try:
                return snap_to_caps(self._parse(text, width, height), obs)
            except ValueError as e:
                error = f" Your last reply was invalid ({e}). Reply with valid JSON only."
        raise ValueError(f"vision backend failed twice;{error}")


def _merge_close(points: list, dist_m: float = MERGE_DIST_M) -> list:
    """Greedy 3D clustering (mean per cluster); one tube seen twice merges."""
    clusters: list = []
    for point in points:
        pos = np.asarray(point["pos"], dtype=float)
        for cluster in clusters:
            if float(np.linalg.norm(np.asarray(cluster["pos"]) - pos)) < dist_m:
                cluster["members"].append(pos)
                cluster["pos"] = [float(x) for x in
                                  sum(cluster["members"]) / len(cluster["members"])]
                break
        else:
            clusters.append({"u": point["u"], "v": point["v"],
                             "pos": [float(x) for x in pos], "members": [pos]})
    return [{"u": c["u"], "v": c["v"], "pos": c["pos"]} for c in clusters]


def _lift_point(u: float, v: float, obs: dict) -> list | None:
    """Pixel -> world XYZ top, or None when the lift is not a tube top.

    A regressed pixel a few pixels off a cap reads background or sky
    depth and would deproject metres away, so such lifts are dropped, not
    emitted: a miss the scorer can see beats a garbage point the grasp
    would chase.
    """
    depth = np.asarray(obs["images"]["depth"])
    depth = depth[..., 0] if depth.ndim == 3 else depth
    dist = _window_depth(depth, u, v)
    if not np.isfinite(dist) or dist <= 0 or dist >= FAR_PLANE_M:
        return None
    pos = deproject(u, v, dist, obs["intrinsics"], obs["pose_mat"])
    if not (TUBE_Z_MIN <= pos[2] <= TUBE_Z_MAX):
        return None
    if abs(pos[0]) > WORKSPACE_XY_M or abs(pos[1]) > WORKSPACE_XY_M:
        return None
    return pos


def perceive_points(env, backend, cameras=CAMERAS, park: bool = True) -> list:
    """3D top estimates [{u, v, pos}] merged over camera views.

    The arm parks for the captures (see parked_view) because at rest it
    hides a tube from the overhead camera on most seeds. Points within
    MERGE_DIST_M of each other are one tube; lifts that miss the tube
    band are dropped. Pass park=False when the arm is already aside
    (tests, scripted callers).
    """
    if isinstance(cameras, str):
        cameras = (cameras,)
    out = []
    with parked_view(env) if park else contextlib.nullcontext():
        for camera in cameras:
            obs = env.camera_observation(camera)
            for point in backend.points(obs):
                pos = _lift_point(point["u"], point["v"], obs)
                if pos is None:
                    continue
                out.append({"u": point["u"], "v": point["v"], "pos": pos,
                            "camera": camera})
    return _merge_close(out)


def score_points(predicted: list, truth: list) -> dict:
    """Optimal-assignment match errors in mm (truth used for scoring only).

    Hungarian assignment, not greedy: with a missed tube, greedy matching
    pairs the surviving point with the wrong tube and reports ~115 mm for
    a point sitting on its own tube. Unmatched truth tubes count as
    misses; unmatched predictions are ignored.
    """
    from scipy.optimize import linear_sum_assignment

    pred = [np.asarray(p, dtype=float) for p in predicted]
    targets = [np.asarray(t, dtype=float) for t in truth]
    errors = []
    if pred and targets:
        costs = np.array([[float(np.linalg.norm(p - t)) for p in pred]
                          for t in targets])
        rows, cols = linear_sum_assignment(costs)
        errors = [float(costs[r, c]) * 1000.0 for r, c in zip(rows, cols)]
    return {"n_truth": len(truth), "n_pred": len(predicted),
            "matched": len(errors), "misses": len(truth) - len(errors),
            "mean_err_mm": (sum(errors) / len(errors)) if errors else None,
            "max_err_mm": max(errors) if errors else None}


def grasp_lift(env, api, tube: str, target: list) -> dict:
    """Grasp check at a target top: approach, pinch below the cap, lift.

    Mirrors the reference policy's grasp. Lift success is read from ground
    truth, which is legitimate here: this scores perception, it does not
    feed the grasp.
    """
    target = np.asarray(target, dtype=float)
    grasp = target.copy()
    grasp[2] -= 0.018
    z0 = float(env.data.body(tube).xpos[2])
    api.set_gripper("right", 0.05)
    moves = [
        api.move_to("right", np.array([grasp[0], grasp[1], TRAVEL_Z])),
        api.move_to("right", grasp, speed=0.08),
    ]
    api.set_gripper("right", 0.0)
    moves.append(api.move_to("right", np.array([grasp[0], grasp[1], TRAVEL_Z])))
    lifted = float(env.data.body(tube).xpos[2]) - z0 > GRASP_LIFT_M
    return {"moved": all(moves), "lifted": lifted and all(moves)}


def evaluate_task(seeds: list, backends: dict, runs_dir: Path,
                  task: str = "tube_rack", cameras=CAMERAS) -> dict:
    """Per-seed pose errors + vision-vs-truth grasp rates for one task."""
    biolab_env = _biolab()
    env_cls = {"tube_rack": biolab_env.TubeRackEnv}[task]
    from biolab.api import TubeRackApi

    per_seed = []
    for seed in seeds:
        env = env_cls(seed=seed)
        truth = env.perceive()["tubes"]
        truth_tops = [truth[name]["top"] for name in sorted(truth)]
        task_tube = TubeRackApi(env).get_task()["tube"]
        row: dict = {"seed": seed, "backends": {}}
        for name, backend in backends.items():
            try:
                points = perceive_points(env, backend, cameras=cameras)
            except (ValueError, RuntimeError) as e:
                row["backends"][name] = {"error": str(e)}
                continue
            errors = score_points([p["pos"] for p in points], truth_tops)
            grasp = {"moved": False, "lifted": False, "detail": "no points"}
            if points:
                target = np.asarray(truth[task_tube]["top"], dtype=float)
                nearest = min(points, key=lambda p: float(
                    np.linalg.norm(np.asarray(p["pos"]) - target)))
                grasp_env = env_cls(seed=seed)
                grasp = grasp_lift(grasp_env, TubeRackApi(grasp_env), task_tube,
                                   nearest["pos"])
            row["backends"][name] = {
                "n_points": len(points),
                "errors_mm": {k: errors[k] for k in
                              ("matched", "misses", "mean_err_mm", "max_err_mm")},
                "grasp": grasp}
        truth_env = env_cls(seed=seed)
        row["grasp_truth"] = grasp_lift(
            truth_env, TubeRackApi(truth_env), task_tube,
            list(truth[task_tube]["top"]))
        per_seed.append(row)
    summary = {}
    for name in backends:
        errs = [r["backends"][name]["errors_mm"]["mean_err_mm"] for r in per_seed
                if "errors_mm" in r["backends"].get(name, {})]
        errs = [e for e in errs if e is not None]
        grasps = [r["backends"][name].get("grasp", {}).get("lifted", False)
                  for r in per_seed]
        summary[name] = {"mean_err_mm": (sum(errs) / len(errs)) if errs else None,
                         "grasp_rate": (sum(1 for g in grasps if g) / len(grasps)
                                        if grasps else None)}
    summary["truth"] = {"grasp_rate": (
        sum(1 for r in per_seed if r["grasp_truth"]["lifted"]) / len(per_seed)
        if per_seed else None)}
    report = {"task": task, "seeds": list(seeds), "cameras": list(cameras),
              "per_seed": per_seed, "summary": summary}
    out_dir = Path(runs_dir) / f"perception_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accuracy.json").write_text(json.dumps(report, indent=2,
                                                      default=str),
                                           encoding="utf-8")
    return {"dir": str(out_dir), "report": report}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Camera pose perception accuracy for one task.")
    parser.add_argument("--task", default="tube_rack", choices=("tube_rack",))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--backends", default="depth",
                        help="comma list of depth,vision (vision needs MODEL_API_KEY)")
    parser.add_argument("--cameras", default=",".join(CAMERAS),
                        help="comma list of cameras (front sees walls, not tops)")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)
    try:
        seeds = [int(s) for s in args.seeds.split(",")]
    except ValueError:
        print(f"error: bad seeds {args.seeds!r}")
        return 1
    backends = {}
    for name in args.backends.split(","):
        if name == "depth":
            backends[name] = DepthBackend()
        elif name == "vision":
            backends[name] = VisionBackend()
        else:
            print(f"error: unknown backend {name!r}")
            return 1
    try:
        out = evaluate_task(seeds, backends, Path(args.runs_dir),
                            task=args.task,
                            cameras=tuple(args.cameras.split(",")))
    except RuntimeError as e:
        print(f"error: {e}")
        return 1
    for name, stats in out["report"]["summary"].items():
        print(f"{name}: mean_err_mm={stats.get('mean_err_mm')} "
              f"grasp_rate={stats.get('grasp_rate')}")
    print(f"wrote {out['dir']}/accuracy.json")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main(_sys.argv[1:]))

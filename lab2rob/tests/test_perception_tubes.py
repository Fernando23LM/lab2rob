"""ST-03: tube-top perception geometry, validation, and scoring.

Offline offline: synthetic depth renders and stub backends stand in for
the MuJoCo env and the vision model, so these tests need no simulator,
no key, and no network.
"""

import numpy as np
import pytest

from lab2rob.perception.tubes import (
    PARK_POS,
    PARK_YAW,
    VisionBackend,
    DepthBackend,
    _lift_point,
    _merge_close,
    deproject,
    parked_view,
    perceive_points,
    score_points,
    segment_tube_tops,
    snap_to_caps,
)

PINHOLE = np.array([[100.0, 0.0, 0.0],
                    [0.0, 100.0, 0.0],
                    [0.0, 0.0, 1.0]])
IDENTITY_POSE = np.eye(4)


def make_obs(depth, intrinsics=PINHOLE, pose_mat=IDENTITY_POSE):
    return {"images": {"depth": np.asarray(depth, dtype=float)},
            "intrinsics": np.asarray(intrinsics, dtype=float),
            "pose_mat": np.asarray(pose_mat, dtype=float)}


def test_deproject_identity_pinhole():
    # f=100, centre origin, identity pose: (10, 20) at 2 m -> (0.2, 0.4, 2).
    assert deproject(10.0, 20.0, 2.0, PINHOLE, IDENTITY_POSE) == pytest.approx(
        [0.2, 0.4, 2.0])


def test_segment_tube_tops_finds_two_plateaus():
    # Two flat cap disks proud of a rack plane, in the tube z-band via a
    # -0.34 m pose shift (cap depth 0.42 m -> world z 0.08 m).
    depth = np.full((200, 200), 0.50)
    yy, xx = np.mgrid[:200, :200]
    for cu, cv in ((60, 60), (140, 140)):
        depth[(xx - cu) ** 2 + (yy - cv) ** 2 <= 12 ** 2] = 0.42
    pose = np.eye(4)
    pose[2, 3] = -0.34
    K = np.array([[200.0, 0.0, 100.0],
                  [0.0, 200.0, 100.0],
                  [0.0, 0.0, 1.0]])
    blobs = segment_tube_tops(depth, K, pose, n_tubes=2)
    assert len(blobs) == 2
    for blob, (cu, cv) in zip(sorted(blobs, key=lambda b: b["u"]),
                              [(60, 60), (140, 140)]):
        # Centroid of the plateau: the cap centre, not a scan-order edge.
        assert abs(blob["u"] - cu) <= 2
        assert abs(blob["v"] - cv) <= 2
        assert blob["pos"][2] == pytest.approx(0.08, abs=1e-6)


def test_segment_tube_tops_empty_scene_finds_none():
    depth = np.full((200, 200), 0.50)
    assert segment_tube_tops(depth, PINHOLE, IDENTITY_POSE) == []


def test_lift_point_keeps_cap_rejects_background_and_sky():
    # Identity pose: depth reads as world z, so 0.08 m is a cap top.
    cap = make_obs(np.full((50, 50), 0.08))
    assert _lift_point(25.0, 25.0, cap) == pytest.approx([0.02, 0.02, 0.08])
    table = make_obs(np.full((50, 50), 0.001))  # below the tube band
    assert _lift_point(25.0, 25.0, table) is None
    sky = make_obs(np.full((50, 50), 84.0))  # far plane / background
    assert _lift_point(25.0, 25.0, sky) is None


def test_lift_point_rejects_outside_workspace():
    # f=100 at (u=7900, v=0): x = 79 * 0.08 = 6.3 m, far off the table.
    cap = make_obs(np.full((50, 8000), 0.08))
    assert _lift_point(7900.0, 25.0, cap) is None


def test_merge_close_clusters_one_tube_seen_twice():
    points = [{"u": 1.0, "v": 1.0, "pos": [0.0, 0.0, 0.08]},
              {"u": 2.0, "v": 2.0, "pos": [0.005, 0.005, 0.08]},
              {"u": 3.0, "v": 3.0, "pos": [0.05, 0.0, 0.08]}]
    merged = _merge_close(points)
    assert len(merged) == 2
    assert merged[0]["pos"][:2] == pytest.approx([0.0025, 0.0025])


def test_score_points_reports_errors_and_misses():
    score = score_points([[0.0, 0.0, 0.08], [0.05, 0.0, 0.08]],
                         [[0.003, 0.004, 0.08], [0.053, 0.004, 0.08]])
    assert score["matched"] == 2
    assert score["misses"] == 0
    assert score["mean_err_mm"] == pytest.approx(5.0, abs=0.01)
    short = score_points([[0.0, 0.0, 0.08]],
                         [[0.0, 0.0, 0.08], [0.05, 0.0, 0.08]])
    assert (short["matched"], short["misses"]) == (1, 1)


def test_score_points_matches_survivor_to_its_own_tube():
    # Optimal assignment: the surviving point sits on tube_1 and must not
    # be charged against tube_0 (greedy reported ~115 mm here).
    score = score_points([[0.05, 0.0, 0.08]],
                         [[0.0, 0.0, 0.08], [0.05, 0.0, 0.08]])
    assert (score["matched"], score["misses"]) == (1, 1)
    assert score["mean_err_mm"] == pytest.approx(0.0, abs=0.01)
    assert score_points([], [[0.0, 0.0, 0.08]])["misses"] == 1


def test_vision_parse_accepts_points_rejects_garbage():
    good = VisionBackend._parse('{"points": [{"u": 98, "v": 200}]}', 640, 480)
    assert good == [{"u": 98.0, "v": 200.0}]
    for bad in ('not json',
                '{"points": []}',
                '{"points": [{"u": 1}]}',
                '{"points": [{"u": 640, "v": 0}]}',
                '{"points": [{"u": 1.5, "v": 0}]}'):
        with pytest.raises(ValueError):
            VisionBackend._parse(bad, 640, 480)


class FakeApi:
    def __init__(self, park_ok=True):
        self.calls = []
        self.park_ok = park_ok

    def move_to(self, arm, position, yaw=0.0):
        self.calls.append(("move_to", arm, tuple(position), yaw))
        return self.park_ok

    def go_home(self, arm):
        self.calls.append(("go_home", arm))


def test_parked_view_parks_then_restores_home():
    api = FakeApi()
    with parked_view(env=None, api=api):
        pass
    assert api.calls == [("move_to", "right", PARK_POS, PARK_YAW),
                         ("go_home", "right")]


def test_parked_view_restores_home_on_body_error():
    api = FakeApi()
    with pytest.raises(RuntimeError, match="boom"):
        with parked_view(env=None, api=api):
            raise RuntimeError("boom")
    assert api.calls[-1] == ("go_home", "right")


def test_parked_view_refuses_when_park_unreachable():
    with pytest.raises(RuntimeError, match="could not park"):
        with parked_view(env=None, api=FakeApi(park_ok=False)):
            pass  # pragma: no cover


class StubBackend:
    name = "stub"

    def __init__(self, pixels):
        self.pixels = pixels

    def points(self, obs):
        return list(self.pixels)


class StubEnv:
    def __init__(self, obs):
        self.obs = obs
        self.cameras = []

    def camera_observation(self, camera):
        self.cameras.append(camera)
        return self.obs


def test_perceive_points_lifts_and_drops_without_parking():
    # One cap pixel and one sky pixel: only the cap survives the lift.
    depth = np.full((50, 50), 84.0)
    depth[20:30, 20:30] = 0.08
    obs = make_obs(depth)
    env = StubEnv(obs)
    backend = StubBackend([{"u": 25.0, "v": 25.0}, {"u": 5.0, "v": 5.0}])
    points = perceive_points(env, backend, cameras=("overhead",), park=False)
    assert len(points) == 1
    assert points[0]["pos"] == pytest.approx([0.02, 0.02, 0.08])
    assert env.cameras == ["overhead"]


def _plateau_obs(centre=(100, 100), radius=12, size=200):
    depth = np.full((size, size), 0.50)
    yy, xx = np.mgrid[:size, :size]
    cu, cv = centre
    depth[(xx - cu) ** 2 + (yy - cv) ** 2 <= radius ** 2] = 0.42
    pose = np.eye(4)
    pose[2, 3] = -0.34
    K = np.array([[200.0, 0.0, 100.0],
                  [0.0, 200.0, 100.0],
                  [0.0, 0.0, 1.0]])
    return make_obs(depth, K, pose)


def test_snap_to_caps_recentres_nearby_pixel():
    obs = _plateau_obs()
    snapped = snap_to_caps([{"u": 120.0, "v": 115.0}], obs)
    assert len(snapped) == 1
    assert abs(snapped[0]["u"] - 100) <= 2
    assert abs(snapped[0]["v"] - 100) <= 2


def test_snap_to_caps_drops_pixel_with_no_cap_in_reach():
    obs = _plateau_obs()
    assert snap_to_caps([{"u": 10.0, "v": 10.0}], obs) == []


def test_depth_backend_returns_blob_pixels():
    depth = np.full((200, 200), 0.50)
    yy, xx = np.mgrid[:200, :200]
    depth[(xx - 100) ** 2 + (yy - 100) ** 2 <= 12 ** 2] = 0.42
    pose = np.eye(4)
    pose[2, 3] = -0.34
    K = np.array([[200.0, 0.0, 100.0],
                  [0.0, 200.0, 100.0],
                  [0.0, 0.0, 1.0]])
    pixels = DepthBackend(n_tubes=2).points(make_obs(depth, K, pose))
    assert len(pixels) == 1
    assert abs(pixels[0]["u"] - 100) <= 2
    assert abs(pixels[0]["v"] - 100) <= 2

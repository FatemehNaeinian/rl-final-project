"""Fast checks of the shared pieces. Run with:  pytest -q tests/"""
import numpy as np
import pytest

from drive_compare.env import make_env, split_seeds
from drive_compare.metrics import classify_outcome, clearance, wilson_ci
from drive_compare.obs import OBS_DIM, StructuredObsWrapper
from drive_compare.prediction import predict_constant_velocity
from drive_compare.vehicle_model import NX, load_params, rollout, step

P = load_params()
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- vehicle model
def _random_states(n):
    return np.column_stack([
        RNG.uniform(-50, 50, n), RNG.uniform(-50, 50, n), RNG.uniform(-np.pi, np.pi, n),
        RNG.uniform(0, 20, n), RNG.uniform(-1, 1, n)])


def test_backends_agree():
    """The MPC's CasADi model must be numerically the same as the NumPy reference."""
    import casadi as ca
    X, U = _random_states(64), RNG.uniform(-1, 1, (64, 2))
    ref = step(P, X, U, "numpy")
    xs, us = ca.SX.sym("x", NX), ca.SX.sym("u", 2)
    f = ca.Function("f", [xs, us], [step(P, xs, us, "casadi")])
    cas = np.array([np.array(f(X[i], U[i])).ravel() for i in range(len(X))])
    np.testing.assert_allclose(cas, ref, rtol=1e-9, atol=1e-9)


def test_model_physics_sanity():
    x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    assert step(P, x0, np.array([0.0, 1.0]))[3] > 10.0  # throttle speeds up
    assert step(P, x0, np.array([0.0, -1.0]))[3] < 10.0  # brake slows down
    stopped = rollout(P, np.array([0, 0, 0, 1.0, 0]), np.tile([0.0, -1.0], (30, 1)))
    assert np.all(stopped[:, 3] >= 0)  # no reversing
    # steering acts one step late, then turns left for positive steer
    xs = rollout(P, x0, np.tile([0.5, 0.0], (5, 1)))
    assert xs[1, 2] == pytest.approx(0.0) and xs[-1, 2] > 0.05


# ---------------------------------------------------------------- prediction / metrics
def test_constant_velocity_prediction():
    from drive_compare.state import Route, WorldState
    route = Route(points=np.zeros((2, 3)), s=np.array([0.0, 1.0]), lane_width=3.5, turn="straight")
    ws = WorldState(t=0, x=0, y=0, yaw=0, v=0, steer_prev=0, length=4.5, width=1.9, s=0, lateral=0,
                    heading_error=0, route=route, traffic_xy=np.array([[10.0, 0.0]]),
                    traffic_vel=np.array([[-5.0, 1.0]]), traffic_yaw=np.array([np.pi]),
                    traffic_speed=np.array([5.1]), traffic_length=np.array([4.5]), traffic_width=np.array([1.9]))
    pred = predict_constant_velocity(ws, horizon=10, dt=0.1)
    assert pred["xy"].shape == (10, 1, 2)
    np.testing.assert_allclose(pred["xy"][-1, 0], [5.0, 1.0])
    assert clearance(ws) == pytest.approx(10.0 - 4.5)


def test_wilson_and_outcomes():
    lo, hi = wilson_ci(50, 50)
    assert 0.9 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_ci(25, 50)
    assert lo == pytest.approx(0.366, abs=0.01) and hi == pytest.approx(0.634, abs=0.01)
    assert classify_outcome({"arrive_dest": True, "crash_vehicle": True}) == "success"
    assert classify_outcome({"crash_vehicle": True, "out_of_road": True}) == "crash_vehicle"
    assert classify_outcome({"crash_sidewalk": True}) == "out_of_road"
    assert classify_outcome({"max_step": True}) == "timeout"


# ---------------------------------------------------------------- environment interface (needs MetaDrive)
@pytest.fixture  # function scope: MetaDrive allows only one engine per process
def env():
    e = StructuredObsWrapper(make_env("high", "dev"))
    yield e
    e.close()


def test_observation_and_world_state(env):
    obs, info = env.reset(seed=split_seeds("dev")[0])
    ws = info["world_state"]
    assert obs.shape == (OBS_DIM, ) and np.all(np.isfinite(obs))
    assert 100 < ws.route.length < 140 and ws.route.turn in ("left", "straight", "right")
    assert ws.s < 10.0 and abs(ws.lateral) < 0.5  # the ego spawns ~5 m into its first lane
    assert ws.route.lat_min < ws.lateral - ws.width / 2 and ws.lateral + ws.width / 2 < ws.route.lat_max
    assert ws.route.lat_max - ws.route.lat_min == pytest.approx(3 * ws.route.lane_width)  # 3-lane roads
    obs, _, _, _, info = env.step([0.0, 1.0])
    assert info["world_state"].steer_prev == 0.0


def test_same_seed_same_traffic(env):
    """Matched initial conditions: a seed must reproduce the same route and initial traffic."""
    seed = split_seeds("dev")[3]
    a = env.reset(seed=seed)[1]["world_state"]
    for _ in range(20):
        env.step([0.0, 0.5])
    b = env.reset(seed=seed)[1]["world_state"]
    np.testing.assert_allclose(a.route.points, b.route.points)
    np.testing.assert_allclose(a.traffic_xy, b.traffic_xy)


def test_seed_outside_split_is_rejected(env):
    with pytest.raises(AssertionError):
        env.reset(seed=split_seeds("test")[0])


def test_tracker_reaches_goal_without_traffic():
    from drive_compare.controllers import make_controller
    from drive_compare.evaluate import run_episode
    e = StructuredObsWrapper(make_env(0.0, "dev"))
    try:
        rows = [run_episode(e, make_controller("tracker"), s)[0] for s in split_seeds("dev", 3)]
    finally:
        e.close()
    assert all(r["outcome"] == "success" for r in rows), rows


def test_mpc_reaches_goal_without_traffic():
    from drive_compare.controllers import make_controller
    from drive_compare.evaluate import run_episode
    e = StructuredObsWrapper(make_env(0.0, "dev"))
    try:
        rows = [run_episode(e, make_controller("mpc"), s)[0] for s in split_seeds("dev", 2)]
    finally:
        e.close()
    assert all(r["outcome"] == "success" for r in rows), rows

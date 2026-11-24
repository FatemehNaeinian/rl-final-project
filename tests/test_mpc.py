"""Behavioural checks of the MPC controller in controlled situations with a known correct answer.

The 'world' here is the MPC's own vehicle model on a straight 3-lane road, so these tests check the optimizer
and cost logic in isolation (no model mismatch, no MetaDrive). Run:  pytest -q tests/test_mpc.py
"""
import numpy as np
import pytest

from drive_compare.controllers.mpc import MPCController
from drive_compare.metrics import clearance
from drive_compare.state import Route, WorldState
from drive_compare.vehicle_model import load_params, step

P = load_params()
LANE_W = 3.5


def straight_route(length=200.0):
    s = np.arange(0.0, length, 0.5)
    pts = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    # ego drives in the middle lane: one lane to the left, one to the right
    return Route(points=pts, s=s, lane_width=LANE_W, turn="straight", lat_min=-1.5 * LANE_W, lat_max=1.5 * LANE_W)


def world(x, route, traffic=()):
    """WorldState from model state x and traffic rows (x, y, yaw, speed)."""
    s, lat, ryaw, _ = route.project(x[:2])
    t = np.array(traffic, dtype=float).reshape(-1, 4)
    return WorldState(
        t=0.0, x=x[0], y=x[1], yaw=x[2], v=x[3], steer_prev=x[4], length=P.length, width=P.width, s=s,
        lateral=lat, heading_error=x[2] - ryaw, route=route, traffic_xy=t[:, :2],
        traffic_vel=np.column_stack([t[:, 3] * np.cos(t[:, 2]), t[:, 3] * np.sin(t[:, 2])]),
        traffic_yaw=t[:, 2], traffic_speed=t[:, 3], traffic_length=np.full(len(t), 4.5),
        traffic_width=np.full(len(t), 1.85))


def simulate(x0, traffic, steps, ctrl=None, u_guess=None):
    """Closed loop on the model. Traffic moves at constant velocity (exactly what the MPC predicts)."""
    route = straight_route()
    ctrl = ctrl or MPCController()
    x = np.asarray(x0, dtype=float)
    traffic = [list(r) for r in traffic]
    ctrl.reset(None, world(x, route, traffic))
    if u_guess is not None:
        ctrl.U_guess = np.tile(u_guess, (ctrl.w.horizon, 1))
    xs, us, clear = [x], [], []
    for _ in range(steps):
        ws = world(x, route, traffic)
        clear.append(clearance(ws))
        u = ctrl.act(None, ws)
        x = step(P, x, u)
        for r in traffic:
            r[0] += r[3] * np.cos(r[2]) * P.dt
            r[1] += r[3] * np.sin(r[2]) * P.dt
        xs.append(x)
        us.append(u)
    return np.array(xs), np.array(us), np.array(clear), ctrl


def test_drives_off_from_standstill_after_braking():
    """Stopped, previous plan was full braking, road is clear: the car must pull away.

    Guards against the zero-gradient trap of the speed clamp (v' = max(v + a*dt, 0)) at standstill.
    """
    xs, us, _, _ = simulate([0, 0, 0, 0.0, 0], [], steps=40, u_guess=[0.0, -1.0])
    assert xs[-1, 3] > 3.0, f"still stuck: final speed {xs[-1, 3]:.2f} m/s"


def test_tracks_route_and_target_speed():
    xs, _, _, ctrl = simulate([0, 1.0, 0.1, 5.0, 0], [], steps=80)
    assert abs(xs[-1, 1]) < 0.3  # converged back onto the lane centre
    assert xs[-1, 3] == pytest.approx(ctrl.w.target_speed, abs=0.5)


def test_stops_behind_stalled_car():
    """A car is stopped in the ego lane 30 m ahead, with the other lanes blocked too: stop, don't crash."""
    blocked = [(40.0, 0.0, 0.0, 0.0), (40.0, LANE_W, 0.0, 0.0), (40.0, -LANE_W, 0.0, 0.0)]
    xs, _, clear, _ = simulate([10, 0, 0, 8.0, 0], blocked, steps=80)
    assert clear.min() > 0.5, f"min clearance {clear.min():.2f} m"
    assert xs[-1, 3] < 0.5  # came to a stop


def test_yields_to_crossing_car():
    """A car crosses the ego's path at a right angle, timed to collide if the ego keeps its speed."""
    # ego reaches x=40 in ~3.75 s at 8 m/s; the crossing car reaches y=0 at the same time
    crossing = [(40.0, -30.0, np.pi / 2, 8.0)]
    _, _, clear, _ = simulate([10, 0, 0, 8.0, 0], crossing, steps=80)
    assert clear.min() > 0.5, f"min clearance {clear.min():.2f} m"


def test_solution_respects_model_and_bounds():
    route = straight_route()
    ctrl = MPCController()
    x = np.array([0, 0.5, 0.05, 6.0, 0.1])
    ws = world(x, route, [(25.0, 0.0, 0.0, 2.0)])
    ctrl.reset(None, ws)
    ctrl.act(None, ws)
    U = np.array(ctrl.opti.debug.value(ctrl.U)).T
    X = np.array(ctrl.opti.debug.value(ctrl.X)).T
    assert np.all(np.abs(U) <= 1 + 1e-6)
    np.testing.assert_allclose(X[0], x, atol=1e-6)
    for k in range(len(U)):  # dynamics constraints hold along the plan
        np.testing.assert_allclose(X[k + 1], step(P, X[k], U[k], smooth=ctrl.smooth), atol=1e-4)


def test_smooth_model_matches_exact_model_away_from_kinks():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), rng.uniform(-3, 3, 200),
                         rng.uniform(2, 20, 200), rng.uniform(-1, 1, 200)])
    U = np.column_stack([rng.uniform(-1, 1, 200), rng.choice([-1, 1], 200) * rng.uniform(0.2, 1, 200)])
    np.testing.assert_allclose(step(P, X, U, smooth=0.02), step(P, X, U), atol=1e-3)

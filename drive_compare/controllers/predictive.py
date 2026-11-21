"""Ingredients shared by the two predictive controllers (MPC in mpc.py, MPPI below).

Both must use the SAME ingredients so the comparison isolates the optimizer, not the model:
  * dynamics:    drive_compare.vehicle_model.step(params, x, u, backend="casadi" | "torch")
  * reference:   build_reference(ws, ...)        -> route points, headings and desired speeds per step
  * traffic:     padded_traffic_prediction(ws, ...) (constant velocity, drive_compare.prediction)
  * cost:        CostWeights and the stage cost described below
  * actions:     MetaDrive's normalized [steer, throttle_brake] in [-1, 1]

Stage cost at step k (state x_k = [px, py, yaw, v, steer_prev], action u_k):
    w_lateral  * e_lat_k^2                      lateral offset from the reference point (contouring-style:
                                                along-track error is NOT penalized, so slowing for traffic
                                                is not fought by the tracking term)
  + w_heading  * (yaw_k - yaw_ref_k)^2
  + w_speed    * (v_k - v_des_k)^2              v_des is capped by route curvature (max_lat_accel)
  + w_steer * steer_k^2 + w_throttle * tb_k^2   control effort
  + w_steer_rate * dsteer_k^2 + w_throttle_rate * dtb_k^2      abrupt control changes
  + w_road * (relu(e_lat_k - hi)^2 + relu(lo - e_lat_k)^2)
                                                stay on the road: [lo, hi] = route.lat_min/lat_max shrunk by
                                                the car's half-width + road_margin (touching the solid line or
                                                road edge ends the episode)
  + w_proximity * sum_j exp(-q_jk / 2)          predicted proximity to traffic vehicle j, where q_jk is
                                                the squared ellipse distance in vehicle j's frame with
                                                semi-axes (L_j + L_ego)/2 + margin_long, (W_j + W_ego)/2 + margin_lat
MPPI starting point:
  pytorch_mppi.MPPI(dynamics, running_cost, nx=5, noise_sigma=..., u_min=-1, u_max=1, device="cuda",
  step_dependent_dynamics=True), where dynamics(x, u, t) = vehicle_model.step(params, x, u, backend="torch")
  and running_cost(x, u, t) implements the stage cost above (BOTH receive the step index t with that flag).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..prediction import predict_constant_velocity
from ..state import WorldState, wrap_angle
from ..vehicle_model import load_params
from .base import Controller

FAR = 1e4  # where padded (absent) traffic vehicles are placed


@dataclass
class CostWeights:
    horizon: int = 20  # steps of 0.1 s (2 s lookahead)
    target_speed: float = 8.0  # m/s on straight road
    max_lat_accel: float = 3.0  # m/s^2: v_des <= sqrt(max_lat_accel / curvature)
    w_lateral: float = 2.0
    w_heading: float = 2.0
    w_speed: float = 0.5
    w_steer: float = 0.1
    w_throttle: float = 0.05
    w_steer_rate: float = 2.0
    w_throttle_rate: float = 0.5
    w_proximity: float = 200.0
    w_road: float = 500.0
    road_margin: float = 0.5  # m kept from the solid line / road edge (covers corners when the car yaws)
    margin_long: float = 2.0  # m added to the ellipse's along-track semi-axis
    margin_lat: float = 0.8  # m added to the ellipse's cross-track semi-axis
    max_vehicles: int = 10  # traffic slots in the optimization (nearest first, padded with far-away dummies)


def build_reference(ws: WorldState, w: CostWeights, dt: float, v_plan=None):
    """Reference for steps k = 1..H.

    Points are spaced along the route by the planned speeds (previous solution, else target speed), so the
    reference moves with the car. Returns (xy (H, 2), yaw (H,) unwrapped near the ego yaw, v_des (H,)).
    """
    H = w.horizon
    if v_plan is None:
        v_plan = np.full(H, max(ws.v, w.target_speed))
    dist = np.cumsum(np.maximum(np.asarray(v_plan[:H], dtype=float), 0.5) * dt)
    pts = ws.route.lookahead(ws.s, dist)
    yaw = ws.yaw + wrap_angle(pts[:, 2] - ws.yaw)  # unwrap so yaw_k - yaw_ref_k has no 2*pi jumps
    # curvature from heading change along the route, measured over a few metres ahead of each point
    ahead = ws.route.lookahead(ws.s, dist + 3.0)
    curv = np.abs(wrap_angle(ahead[:, 2] - pts[:, 2])) / 3.0
    v_des = np.minimum(w.target_speed, np.sqrt(w.max_lat_accel / np.maximum(curv, 1e-3)))
    return pts[:, :2], yaw, v_des


def road_band(ws: WorldState, w: CostWeights) -> tuple[float, float]:
    """Allowed range of the car centre's lateral offset from the route (lo, hi)."""
    half = ws.width / 2 + w.road_margin
    return ws.route.lat_min + half, ws.route.lat_max - half


def padded_traffic_prediction(ws: WorldState, w: CostWeights, dt: float):
    """Constant-velocity prediction for the `max_vehicles` nearest cars, padded to a fixed size.

    Returns xy (H, M, 2), cos/sin of yaw (M,), ellipse semi-axes a, b (M,). Padding sits FAR away.
    """
    H, M = w.horizon, w.max_vehicles
    xy = np.full((H, M, 2), FAR)
    cy, sy = np.ones(M), np.zeros(M)
    a, b = np.ones(M), np.ones(M)
    if ws.num_traffic:
        pred = predict_constant_velocity(ws, H, dt)
        order = np.argsort(np.linalg.norm(ws.traffic_xy - [ws.x, ws.y], axis=1))[:M]
        n = len(order)
        xy[:, :n] = pred["xy"][:, order]
        cy[:n], sy[:n] = np.cos(ws.traffic_yaw[order]), np.sin(ws.traffic_yaw[order])
        a[:n] = (ws.traffic_length[order] + ws.length) / 2 + w.margin_long
        b[:n] = (ws.traffic_width[order] + ws.width) / 2 + w.margin_lat
    return xy, cy, sy, a, b


class MPPIController(Controller):
    name = "mppi"

    def __init__(self, weights: CostWeights | None = None, num_samples: int = 1000, device: str = "cuda"):
        self.p = load_params()
        self.w = weights or CostWeights()
        self.num_samples = num_samples
        self.device = device

    def reset(self, env, ws):
        raise NotImplementedError("Create the pytorch_mppi.MPPI object here, then call .command(state) in act().")

    def act(self, obs, ws):
        raise NotImplementedError

"""Shared observation interface.

`StructuredObsWrapper` wraps the raw MetaDrive env so that every method sees the same information:
  * observation (for PPO): a fixed-size vector encoding of the WorldState (ego, route ahead, K nearest cars)
  * info["world_state"] (for MPC / MPPI / baselines): the WorldState itself (drive_compare.state)

It also tracks the previous steering command (needed by the vehicle model, because steering acts with a
one-step delay) and can add the plan's "abrupt control change" penalty to the reward for PPO training.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .state import SENSE_RADIUS, Route, WorldState, build_route, extract_world_state, wrap_angle

N_ROUTE_POINTS = 10
ROUTE_POINT_SPACING = 4.0  # m: route points at 4, 8, ..., 40 m ahead
K_NEAREST = 6
TRAFFIC_FEATURES = 7  # present, rel x, rel y, rel vx, rel vy, sin(rel yaw), cos(rel yaw)
EGO_FEATURES = 8
OBS_DIM = EGO_FEATURES + 2 * N_ROUTE_POINTS + 1 + K_NEAREST * TRAFFIC_FEATURES

MAX_SPEED = 80 / 3.6
VEL_SCALE = 20.0
OBS_CLIP = 5.0


def _to_ego_frame(dxy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * dxy[..., 0] + s * dxy[..., 1], -s * dxy[..., 0] + c * dxy[..., 1]], -1)


def encode(ws: WorldState, last_throttle: float) -> np.ndarray:
    """Fixed-size vector (OBS_DIM,) derived only from the WorldState."""
    ego = [
        ws.v / MAX_SPEED,
        ws.steer_prev,
        last_throttle,
        ws.lateral / ws.route.lane_width,
        np.sin(ws.heading_error),
        np.cos(ws.heading_error),
        (ws.lateral - ws.route.lat_min) / ws.route.lane_width,  # distance to the right road edge
        (ws.route.lat_max - ws.lateral) / ws.route.lane_width,  # distance to the solid line on the left
    ]
    ahead = ws.route.lookahead(ws.s, ROUTE_POINT_SPACING * np.arange(1, N_ROUTE_POINTS + 1))
    route_feat = _to_ego_frame(ahead[:, :2] - np.array([ws.x, ws.y]), ws.yaw) / (ROUTE_POINT_SPACING * N_ROUTE_POINTS)
    remaining = [ws.remaining / ws.route.length]

    traffic = np.zeros((K_NEAREST, TRAFFIC_FEATURES))
    if ws.num_traffic:
        rel = ws.traffic_xy - np.array([ws.x, ws.y])
        order = np.argsort(np.linalg.norm(rel, axis=1))[:K_NEAREST]
        ego_vel = ws.v * np.array([np.cos(ws.yaw), np.sin(ws.yaw)])
        rel_p = _to_ego_frame(rel[order], ws.yaw) / SENSE_RADIUS
        rel_v = _to_ego_frame(ws.traffic_vel[order] - ego_vel, ws.yaw) / VEL_SCALE
        rel_yaw = wrap_angle(ws.traffic_yaw[order] - ws.yaw)
        n = len(order)
        traffic[:n] = np.column_stack([np.ones(n), rel_p, rel_v, np.sin(rel_yaw), np.cos(rel_yaw)])

    vec = np.concatenate([ego, route_feat.ravel(), remaining, traffic.ravel()]).astype(np.float32)
    return np.clip(vec, -OBS_CLIP, OBS_CLIP)


class StructuredObsWrapper(gym.Wrapper):
    """Wraps a MetaDrive env from drive_compare.env.make_env.

    Args:
        smoothness_penalty: weight w of the extra reward term -w * ||a_t - a_{t-1}||^2 (PPO training only;
            leave at 0 for evaluation so rewards stay comparable to MetaDrive's).
        include_world_state: put the WorldState in info (needed by planners and the evaluator; training
            turns it off to avoid sending it between worker processes every step).
    """

    def __init__(self, env, smoothness_penalty: float = 0.0, include_world_state: bool = True):
        super().__init__(env)
        self.include_world_state = include_world_state
        self.observation_space = gym.spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.smoothness_penalty = smoothness_penalty
        self.route: Route | None = None
        self.last_action = np.zeros(2)

    def _observe(self, info):
        ws = extract_world_state(self.env, self.route, steer_prev=float(self.last_action[0]))
        info = dict(info)
        if self.include_world_state:
            info["world_state"] = ws
        return encode(ws, float(self.last_action[1])), info

    def reset(self, *, seed=None, options=None):
        # MetaDrive's reset() takes only `seed`; accept the standard gymnasium signature and drop `options`
        _, info = self.env.reset(seed=seed)
        self.route = build_route(self.env)
        self.last_action = np.zeros(2)
        return self._observe(info)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        _, reward, terminated, truncated, info = self.env.step(action)
        if self.smoothness_penalty:
            penalty = self.smoothness_penalty * float(np.sum((action - self.last_action)**2))
            reward -= penalty
            info["smoothness_penalty"] = penalty
        self.last_action = action
        if terminated or truncated:
            info["is_success"] = bool(info.get("arrive_dest", False))  # read by SB3's EvalCallback / Monitor
        obs, info = self._observe(info)
        return obs, reward, terminated, truncated, info

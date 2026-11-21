"""Reference controllers that need no training or tuning.

* IDMController:     MetaDrive's rule-based driver (lane-keeping PID + Intelligent Driver Model), the
                     same logic the traffic vehicles use. A floor any learned/planned method should beat.
* ExpertController:  the PPO policy pretrained by the MetaDrive authors (metadrive/examples/ppo_expert).
                     It uses MetaDrive's own lidar observation, NOT the shared interface, so report it as an
                     external reference rather than as "our PPO".
* TrackerController: pure-pursuit steering + P speed control on the shared route, using the identified
                     vehicle model. It ignores traffic entirely; use it to sanity-check the route and the
                     action mapping (it should succeed on every route with no traffic).

IDM and the expert read the simulator directly (that is how MetaDrive implements them); they are
baselines, not part of the comparison, so the same-information rule does not apply to them.
"""
from __future__ import annotations

import numpy as np

from ..state import WorldState, wrap_angle
from ..vehicle_model import load_params, steer_for_angle, throttle_for_accel
from .base import Controller


class IDMController(Controller):
    name = "idm"

    def reset(self, env, ws):
        from metadrive.policy.idm_policy import IDMPolicy
        self._policy = IDMPolicy(env.unwrapped.agent, random_seed=0)

    def act(self, obs, ws):
        return np.clip(np.asarray(self._policy.act(), dtype=float), -1, 1)


class ExpertController(Controller):
    name = "expert"

    def reset(self, env, ws):
        self._vehicle = env.unwrapped.agent

    def act(self, obs, ws):
        from metadrive.examples import expert
        return np.clip(np.asarray(expert(self._vehicle, deterministic=True), dtype=float), -1, 1)


class TrackerController(Controller):
    name = "tracker"

    def __init__(self, target_speed: float = 8.0, lookahead_time: float = 1.0, min_lookahead: float = 4.0):
        self.p = load_params()
        self.target_speed = target_speed
        self.lookahead_time = lookahead_time
        self.min_lookahead = min_lookahead

    def act(self, obs, ws: WorldState):
        p = self.p
        # steering acts one step late, so aim from where the car will be after this step
        ld = max(self.min_lookahead, self.lookahead_time * ws.v)
        x_pred = ws.x + ws.v * np.cos(ws.yaw) * p.dt
        y_pred = ws.y + ws.v * np.sin(ws.yaw) * p.dt
        tx, ty, _ = ws.route.lookahead(ws.s, [ld + ws.v * p.dt])[0]
        alpha = wrap_angle(np.arctan2(ty - y_pred, tx - x_pred) - ws.yaw)
        delta = np.arctan2(2.0 * p.wheelbase * np.sin(alpha), ld)
        steer = steer_for_angle(p, delta * (1.0 + p.k_understeer * ws.v**2))
        # slow down for the turn and near the end of the route
        v_ref = min(self.target_speed, np.sqrt(max(2.0 * 2.0 * (ws.remaining + 5.0), 0.0)) + 2.0)
        throttle = throttle_for_accel(p, 1.0 * (v_ref - ws.v), ws.v)
        return np.array([steer, throttle])

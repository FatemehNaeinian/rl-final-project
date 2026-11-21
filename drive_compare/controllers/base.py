from __future__ import annotations

import numpy as np

from ..state import WorldState


class Controller:
    """Interface every method implements so drive_compare.evaluate can run it.

    The evaluator calls `reset` once per episode (after env.reset) and then `act` once per 0.1 s step.
    Only the time spent inside `act` is reported as decision time, so do all per-episode setup in `reset`.
    """

    name = "base"

    def reset(self, env, ws: WorldState) -> None:
        """`env` is the StructuredObsWrapper; most controllers only need `ws`."""

    def act(self, obs: np.ndarray, ws: WorldState) -> np.ndarray:
        """Return MetaDrive's action [steer, throttle_brake], each in [-1, 1].

        obs: the fixed-size vector (what PPO sees). ws: the same information as a WorldState
        (what MPC/MPPI use). Do not read anything else from the simulator.
        """
        raise NotImplementedError

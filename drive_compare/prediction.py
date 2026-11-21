"""Traffic motion prediction shared by MPC and MPPI.

The plan specifies one simple model for both predictive controllers: constant velocity. Each visible
vehicle keeps its current velocity vector and heading over the planning horizon.
"""
from __future__ import annotations

import numpy as np

from .state import WorldState


def predict_constant_velocity(ws: WorldState, horizon: int, dt: float) -> dict[str, np.ndarray]:
    """Predicted traffic for steps k = 1..horizon.

    Returns:
        xy:     (horizon, N, 2) predicted centre positions
        yaw:    (horizon, N)    headings (constant)
        length: (N,), width: (N,)
    N is the number of visible vehicles (may be 0, giving empty arrays).
    """
    k = np.arange(1, horizon + 1, dtype=float)[:, None, None] * dt
    xy = ws.traffic_xy[None] + k * ws.traffic_vel[None]
    yaw = np.broadcast_to(ws.traffic_yaw[None], (horizon, ws.num_traffic)).copy()
    return dict(xy=xy, yaw=yaw, length=ws.traffic_length.copy(), width=ws.traffic_width.copy())

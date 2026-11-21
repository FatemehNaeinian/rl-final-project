"""Evaluation metrics from the plan: outcome rates, travel time, minimum clearance, decision time."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .state import WorldState

OUTCOMES = ("success", "crash_vehicle", "crash_other", "out_of_road", "timeout")


def classify_outcome(info: dict) -> str:
    """Exactly one terminal outcome per episode, checked in priority order."""
    if info.get("arrive_dest"):
        return "success"
    if info.get("crash_vehicle"):
        return "crash_vehicle"
    if info.get("out_of_road") or info.get("crash_sidewalk"):
        return "out_of_road"
    if info.get("crash_object") or info.get("crash_building") or info.get("crash_human"):
        return "crash_other"
    return "timeout"


def _box(x, y, yaw, length, width) -> Polygon:
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in corners])


def clearance(ws: WorldState) -> float:
    """Smallest gap (m) between the ego's bounding box and any visible vehicle's box; inf if none visible.

    Box-to-box distance, not centre distance: centre distance overstates safety by roughly a car length.
    """
    if ws.num_traffic == 0:
        return float("inf")
    ego = _box(ws.x, ws.y, ws.yaw, ws.length, ws.width)
    return float(
        min(
            ego.distance(_box(x, y, yaw, l, w)) for (x, y), yaw, l, w in zip(
                ws.traffic_xy, ws.traffic_yaw, ws.traffic_length, ws.traffic_width)))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate (well-behaved near 0 and 1, unlike the normal approximation)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(centre - half, 0.0)), float(min(centre + half, 1.0)))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes (same seeds, two methods).

    a_only: seeds where method A succeeded and B failed; b_only: the reverse.
    """
    from scipy.stats import binomtest
    n = a_only + b_only
    if n == 0:
        return 1.0
    return float(binomtest(a_only, n, 0.5).pvalue)

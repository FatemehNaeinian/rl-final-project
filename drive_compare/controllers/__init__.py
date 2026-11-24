from __future__ import annotations

from .base import Controller
from .baselines import ExpertController, IDMController, TrackerController
from .mpc import MPCController
from .predictive import CostWeights


def make_controller(name: str, **kwargs) -> Controller:
    """Factory used by drive_compare.evaluate."""
    registry = {
        "idm": IDMController,
        "expert": ExpertController,
        "tracker": TrackerController,
        "mpc": MPCController,
    }
    return registry[name](**kwargs)


__all__ = ["Controller", "CostWeights", "make_controller"]

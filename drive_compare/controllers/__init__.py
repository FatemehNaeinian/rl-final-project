from __future__ import annotations

from .base import Controller
from .baselines import ExpertController, IDMController, TrackerController
from .mpc import MPCController
from .predictive import CostWeights, MPPIController


def make_controller(name: str, **kwargs) -> Controller:
    """Factory used by drive_compare.evaluate. `ppo` needs model_path=..."""
    if name == "ppo":
        from .ppo import PPOController
        return PPOController(**kwargs)
    registry = {
        "idm": IDMController,
        "expert": ExpertController,
        "tracker": TrackerController,
        "mpc": MPCController,
        "mppi": MPPIController,
    }
    return registry[name](**kwargs)


__all__ = ["Controller", "CostWeights", "make_controller"]

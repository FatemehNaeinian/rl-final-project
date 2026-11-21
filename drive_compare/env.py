"""Shared MetaDrive environment configuration for all three controllers.

Every method (PPO, MPC, MPPI, baselines) must build its environment through `make_env` so that the
map, traffic, termination rules, control period and seed splits are identical.

Scenario: MetaDrive's 4-way intersection block ("X"). The seed decides which exit the route takes
and how traffic is spawned; with `random_traffic=False` the same seed gives the same initial traffic.
"""
from __future__ import annotations

import logging

from metadrive.envs import MetaDriveEnv

# ---- control period ----------------------------------------------------------------------------------
PHYSICS_DT = 0.02
DECISION_REPEAT = 5
DT = PHYSICS_DT * DECISION_REPEAT  # 0.1 s per env.step(): the real-time budget for one decision

# ---- traffic settings compared in the report --------------------------------------------------------------
# On the "X" map, density 0.1 spawns ~3 vehicles and 0.3 spawns ~9.
DENSITIES = {"low": 0.1, "high": 0.3}

# ---- seed splits (disjoint) ---------------------------------------------------------------------------------
# MetaDrive only accepts seeds inside [start_seed, start_seed + num_scenarios), so each split has its own range.
SPLITS = {
    "test": (0, 200),  # held-out evaluation; the report uses the first 50 (more if time allows)
    "dev": (200, 300),  # tuning MPC/MPPI cost weights and PPO hyperparameters
    "train": (1000, 6000),  # PPO training only
}

HORIZON = 600  # steps (60 s); routes are ~110-125 m, so this only cuts off stuck episodes


def quiet_metadrive_config(**overrides) -> dict:
    cfg = dict(
        use_render=False,
        log_level=logging.WARNING,
        physics_world_step_size=PHYSICS_DT,
        decision_repeat=DECISION_REPEAT,
    )
    cfg.update(overrides)
    return cfg


def env_config(density: float | str = "low", split: str = "test", **overrides) -> dict:
    if isinstance(density, str):
        density = DENSITIES[density]
    start, end = SPLITS[split]
    cfg = quiet_metadrive_config(
        map="X",
        traffic_density=density,
        random_traffic=False,
        start_seed=start,
        num_scenarios=end - start,
        horizon=HORIZON,
        crash_vehicle_done=True,
        crash_object_done=True,
        out_of_road_done=True,
    )
    cfg.update(overrides)
    return cfg


def make_env(density: float | str = "low", split: str = "test", **overrides) -> MetaDriveEnv:
    """Raw MetaDrive env. Wrap with `drive_compare.obs.StructuredObsWrapper` for the shared observation."""
    return MetaDriveEnv(env_config(density, split, **overrides))


def split_seeds(split: str, n: int | None = None) -> list[int]:
    start, end = SPLITS[split]
    seeds = list(range(start, end))
    return seeds if n is None else seeds[:n]

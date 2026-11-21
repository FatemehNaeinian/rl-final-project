from __future__ import annotations

import numpy as np

from .base import Controller


class PPOController(Controller):
    """Runs a Stable-Baselines3 PPO policy trained with scripts/train_ppo.py on the shared observation."""

    name = "ppo"

    def __init__(self, model_path: str, device: str = "cpu"):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device=device)

    def act(self, obs, ws):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

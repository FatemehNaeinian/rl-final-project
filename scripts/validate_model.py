"""How far off is the shared vehicle model over a planning horizon?

Drives the intersection (no traffic) with the tracker baseline plus random action noise, and at many
points along each episode compares an open-loop model rollout of the actions that were actually applied
against what MetaDrive did. Reports position / heading / speed error at 0.5 s, 1 s and 2 s ahead.
These numbers belong in the report's discussion of "how the approximate vehicle model affects MPC/MPPI".

Usage:  python scripts/validate_model.py [--episodes 15]
"""
import argparse

import numpy as np

from drive_compare.controllers import make_controller
from drive_compare.env import make_env, split_seeds
from drive_compare.obs import StructuredObsWrapper
from drive_compare.state import wrap_angle
from drive_compare.vehicle_model import load_params, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.15, help="std of action noise added to the tracker")
    args = ap.parse_args()

    p = load_params()
    rng = np.random.default_rng(0)
    horizons = {"0.5s": 5, "1.0s": 10, "2.0s": 20}
    errs = {h: [] for h in horizons}
    env = StructuredObsWrapper(make_env(0.0, "dev"))
    ctrl = make_controller("tracker")
    try:
        for seed in split_seeds("dev", args.episodes):
            obs, info = env.reset(seed=seed)
            ws = info["world_state"]
            ctrl.reset(env, ws)
            states, actions = [ws.ego_state], []
            while True:
                a = np.clip(ctrl.act(obs, ws) + rng.normal(0, args.noise, 2), -1, 1)
                obs, _, term, trunc, info = env.step(a)
                ws = info["world_state"]
                states.append(ws.ego_state)
                actions.append(a)
                if term or trunc:
                    break
            states, actions = np.array(states), np.array(actions)
            for t0 in range(0, len(actions) - max(horizons.values()), 3):
                pred = rollout(p, states[t0], actions[t0:t0 + max(horizons.values())])
                for name, h in horizons.items():
                    true, est = states[t0 + h], pred[h]
                    errs[name].append((np.hypot(*(est[:2] - true[:2])), abs(wrap_angle(est[2] - true[2])),
                                       abs(est[3] - true[3])))
    finally:
        env.close()
    print(f"open-loop model error over {args.episodes} intersection episodes (median / 90th percentile)")
    print(f"{'horizon':>8} {'position m':>18} {'heading deg':>18} {'speed m/s':>18}")
    for name in horizons:
        e = np.array(errs[name])
        med, p90 = np.median(e, 0), np.percentile(e, 90, 0)
        print(f"{name:>8} {med[0]:8.2f} / {p90[0]:6.2f} {np.degrees(med[1]):9.2f} / {np.degrees(p90[1]):6.2f} "
              f"{med[2]:9.2f} / {p90[2]:6.2f}")


if __name__ == "__main__":
    main()

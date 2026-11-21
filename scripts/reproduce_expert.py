"""Reproduce the documented performance of MetaDrive's pretrained PPO expert.

MetaDrive's own tests (metadrive/tests/test_policy/test_expert_performance.py) state the expected results.
This script runs the SAME evaluation function on the same four configurations (10 episodes each) and
records the measured numbers next to the expected ranges:

  CCC road, traffic, flat terrain:  episode reward in (300, 350)
  CCC road, traffic, mesh terrain:  episode reward in (300, 350)
  CCC road, no traffic:             success rate 1.0, reward in [300, 350]
  XTXTS intersections, no traffic:  success rate 1.0, reward > 400

Usage:  python scripts/reproduce_expert.py [--out results/reproduction/expert.json]
"""
import argparse
import json
import pathlib

from metadrive.tests.test_policy.test_expert_performance import _evaluate

CASES = [
    ("CCC, traffic, use_mesh_terrain=True",
     dict(num_scenarios=1, map="CCC", use_mesh_terrain=True, start_seed=2, random_traffic=False),
     dict(need_on_same_lane=False), "300 < reward < 350", lambda r, s: 300 < r < 350),
    ("CCC, traffic, use_mesh_terrain=False",
     dict(num_scenarios=1, map="CCC", use_mesh_terrain=False, start_seed=2, random_traffic=False),
     dict(need_on_same_lane=False), "300 < reward < 350", lambda r, s: 300 < r < 350),
    ("CCC, no traffic",
     dict(num_scenarios=1, random_agent_model=False, map="CCC", start_seed=2, traffic_density=0,
          random_traffic=False),
     dict(has_traffic=False), "success = 1.0, 300 <= reward <= 350", lambda r, s: s == 1.0 and 300 <= r <= 350),
    ("XTXTS intersections, no traffic",
     dict(num_scenarios=1, random_agent_model=False, map="XTXTS", start_seed=2, traffic_density=0,
          random_traffic=False),
     dict(has_traffic=False), "success = 1.0, reward > 400", lambda r, s: s == 1.0 and r > 400),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reproduction/expert.json")
    args = ap.parse_args()
    rows = []
    for name, cfg, kw, expected, check in CASES:
        reward, success = _evaluate(dict(cfg, log_level=50), num_episode=10, **kw)
        rows.append(dict(case=name, episodes=10, mean_episode_reward=round(float(reward), 1),
                         success_rate=float(success), expected=expected, reproduced=bool(check(reward, success))))
        print(f"{name:34s} reward={reward:7.1f} success={success:.2f}  expected: {expected:38s} "
              f"{'REPRODUCED' if rows[-1]['reproduced'] else 'NOT reproduced'}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

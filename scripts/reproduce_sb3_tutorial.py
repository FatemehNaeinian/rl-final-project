"""Reproduce MetaDrive's official Stable-Baselines3 PPO tutorial (metadrive/documentation/source/training.ipynb).

Published result: after 300k steps, one deterministic evaluation episode scores episode_reward = 168.7.
This script copies the tutorial's env config, PPO hyperparameters, step budget and evaluation loop exactly,
minus the GIF rendering. The tutorial reports a single training seed, so run several seeds to see where 168.7
falls in the spread.

Usage:  python scripts/reproduce_sb3_tutorial.py --seed 0 [--out results/reproduction/sb3_tutorial_s0.json]
"""
import argparse
import json
import pathlib
import time
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

PUBLISHED_EPISODE_REWARD = 168.7058092369063


def create_env(need_monitor=False):
    from metadrive.envs import MetaDriveEnv
    env = MetaDriveEnv(
        dict(
            map="C",
            discrete_action=True,
            discrete_throttle_dim=3,
            discrete_steering_dim=3,
            horizon=500,
            random_spawn_lane_index=False,
            num_scenarios=1,
            start_seed=5,
            traffic_density=0,
            accident_prob=0,
            log_level=50,
        ))
    if need_monitor:
        env = Monitor(env)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out")
    args = ap.parse_args()

    t0 = time.time()
    set_random_seed(args.seed)
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(4)])
    # like the tutorial, seed only via set_random_seed: PPO(seed=...) would also reseed the envs, and MetaDrive
    # rejects seeds outside the tutorial's single scenario (start_seed=5, num_scenarios=1)
    model = PPO("MlpPolicy", train_env, n_steps=4096, verbose=0, device="cpu")
    model.learn(total_timesteps=args.timesteps, log_interval=4)
    train_env.close()
    train_minutes = (time.time() - t0) / 60

    # evaluation exactly as in the tutorial: one deterministic episode, stop on `done` (terminated)
    env = create_env()
    total_reward, info = 0.0, {}
    try:
        obs, _ = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:  # the tutorial only checks `done`; truncation at horizon=500 also ends it
                break
    finally:
        env.close()

    result = dict(
        seed=args.seed,
        timesteps=args.timesteps,
        episode_reward=float(total_reward),
        published_episode_reward=PUBLISHED_EPISODE_REWARD,
        arrive_dest=bool(info.get("arrive_dest", False)),
        crash=bool(info.get("crash", False)),
        out_of_road=bool(info.get("out_of_road", False)),
        train_minutes=round(train_minutes, 1),
    )
    print(json.dumps(result, indent=2))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

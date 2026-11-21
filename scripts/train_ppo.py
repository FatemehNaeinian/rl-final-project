"""Train PPO (Stable-Baselines3) on the shared observation, adapted from MetaDrive's training tutorial
(metadrive/documentation/source/training.ipynb).

* Training uses the "train" seed split; periodic evaluation uses the "dev" split (never "test").
* Half of the worker envs use low traffic density and half high, so one policy covers both settings.
* Reward: MetaDrive's default (route progress + speed bonus, +10 success, -5 crash / off-road) plus the
  plan's abrupt-control-change penalty (--smoothness).

Usage:
  python scripts/train_ppo.py --timesteps 2000000 --envs 16 --out runs/ppo
  tensorboard --logdir runs/                        # curves: rollout/ep_rew_mean, eval/success_rate
Then evaluate the best checkpoint on held-out seeds:
  python -m drive_compare.evaluate --controller ppo --model runs/ppo/best_model.zip --n 50
"""
import argparse
import pathlib
from functools import partial

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class SeedIntoSplit(gym.Wrapper):
    """SB3 seeds worker i with (seed + i), but MetaDrive only accepts seeds inside the split's range.
    Map any requested seed into that range; unseeded resets let MetaDrive sample a scenario in the range."""

    def __init__(self, env, split):
        super().__init__(env)
        from drive_compare.env import SPLITS
        self.start, self.end = SPLITS[split]

    def reset(self, *, seed=None, options=None):
        if seed is not None and not self.start <= seed < self.end:
            seed = self.start + seed % (self.end - self.start)
        return self.env.reset(seed=seed, options=options)


def make_train_env(density, split, smoothness, rank, seed, reward_cfg):
    # imports inside so each worker process builds its own MetaDrive engine
    from drive_compare.env import make_env
    from drive_compare.obs import StructuredObsWrapper
    set_random_seed(seed + rank)
    env = StructuredObsWrapper(make_env(density, split, **reward_cfg), smoothness_penalty=smoothness,
                               include_world_state=False)
    return Monitor(SeedIntoSplit(env, split), info_keywords=("is_success", ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=16, help="parallel MetaDrive workers (one CPU core each)")
    ap.add_argument("--smoothness", type=float, default=0.02)
    # MetaDrive's reward terms (defaults = MetaDrive's own). With the defaults, speeding off the road at a turn
    # (-5) is nearly as good as finishing, so the first 1M-step run never learned to turn: raise the penalties.
    ap.add_argument("--success-reward", type=float, default=10.0)
    ap.add_argument("--crash-penalty", type=float, default=5.0, help="crash_vehicle / crash_object penalty")
    ap.add_argument("--out-of-road-penalty", type=float, default=5.0)
    ap.add_argument("--speed-reward", type=float, default=0.1, help="per-step bonus * speed / max_speed")
    ap.add_argument("--eval-every", type=int, default=50_000, help="timesteps between dev evaluations")
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--device", default="cpu", help="cpu is usually faster than GPU for small MLP policies")
    ap.add_argument("--resume", help="continue training from this checkpoint .zip (the worker count may differ); "
                    "--timesteps is then the TOTAL target, counting the steps already done")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reward_cfg = dict(success_reward=args.success_reward, crash_vehicle_penalty=args.crash_penalty,
                      crash_object_penalty=args.crash_penalty, out_of_road_penalty=args.out_of_road_penalty,
                      speed_reward=args.speed_reward)
    densities = ["low" if i % 2 == 0 else "high" for i in range(args.envs)]
    train_env = SubprocVecEnv(
        [partial(make_train_env, d, "train", args.smoothness, i, args.seed, reward_cfg)
         for i, d in enumerate(densities)])
    # evaluation alternates densities too: one low and one high worker on the dev split
    eval_env = SubprocVecEnv([partial(make_train_env, d, "dev", 0.0, 100 + i, args.seed, reward_cfg)
                              for i, d in enumerate(["low", "high"])])

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device, tensorboard_log=str(out.parent))
        print(f"resumed from {args.resume} at {model.num_timesteps} timesteps")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=1024,  # per env -> 1024 * envs samples per update
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=str(out.parent),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    callbacks = [
        EvalCallback(eval_env, n_eval_episodes=args.eval_episodes, eval_freq=max(args.eval_every // args.envs, 1),
                     best_model_save_path=str(out), log_path=str(out), deterministic=True),
        CheckpointCallback(save_freq=max(250_000 // args.envs, 1), save_path=str(out / "checkpoints"),
                           name_prefix="ppo"),
    ]
    try:
        remaining = max(args.timesteps - model.num_timesteps, 0)
        model.learn(total_timesteps=remaining, callback=callbacks, tb_log_name=out.name,
                    reset_num_timesteps=not args.resume)
    finally:
        model.save(out / "final_model")
        train_env.close()
        eval_env.close()
    print(f"saved -> {out / 'final_model.zip'} (best on dev: {out / 'best_model.zip'})")


if __name__ == "__main__":
    main()

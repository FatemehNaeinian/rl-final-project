"""System identification: map MetaDrive's normalized actions to physical quantities.

MetaDrive's action is [steer, throttle_brake] in [-1, 1]^2, but MPC/MPPI reason about a kinematic
bicycle with front-wheel angle (rad) and acceleration (m/s^2). This script drives the ego car with
random piecewise-constant commands on an empty straight road, logs the response, and fits:

  longitudinal:  a = c_throttle * u+ + c_brake * u- + c_drag * v + c_const      (u+ = max(u,0), u- = min(u,0))
  lateral:       yaw_rate = v / (lf + lr) * tan(k_steer * steer_prev) / (1 + k_understeer * v^2)

The steering command acts with a ONE-STEP DELAY: the yaw during step t is driven by the command sent at
step t-1 (R^2 ~0.94 with the delay vs ~0.75 without), so the shared model keeps the previous steering
command in its state. The fit also reports the no-delay R^2 for comparison.

Results are written to drive_compare/vehicle_params.json, which drive_compare.vehicle_model loads.

Usage:  python scripts/sysid.py [--episodes 40]
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

from drive_compare.env import DT, quiet_metadrive_config
from metadrive.envs import MetaDriveEnv

OUT = pathlib.Path(__file__).resolve().parents[1] / "drive_compare" / "vehicle_params.json"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def collect(episodes, seed):
    rng = np.random.default_rng(seed)
    env = MetaDriveEnv(
        quiet_metadrive_config(
            map="SSSSSSSS",  # long straight road, no traffic
            traffic_density=0.0,
            num_scenarios=1,
            start_seed=0,
            horizon=2000,
            out_of_road_done=True,
        ))
    lon, lat = [], []
    try:
        for ep in range(episodes):
            env.reset(seed=0)
            v = env.agent
            lateral_mode = ep % 2 == 1  # odd episodes: steering excitation at roughly constant speed
            step = 0
            while True:
                if lateral_mode:
                    # accelerate straight to a target speed, then hold it with a P-controller and excite steering
                    if step == 0:
                        target_v, steer, prev_steer = rng.uniform(3, 18), 0.0, 0.0
                    if step >= 60 and step % 8 == 0:
                        steer = rng.uniform(-0.25, 0.25)
                    u = float(np.clip(0.5 * (target_v - v.speed), -1, 1))
                    action = [steer, u]
                else:
                    if step % 10 == 0:
                        u = rng.uniform(-1, 1) if rng.random() < 0.8 else rng.choice([-1.0, 1.0])
                    action = [0.0, u]
                s0 = dict(x=v.position[0], y=v.position[1], th=v.heading_theta, spd=v.speed)
                _, _, term, trunc, _ = env.step(action)
                s1 = dict(spd=v.speed, th=v.heading_theta)
                if lateral_mode:
                    if step >= 60:
                        lat.append((s0["spd"], action[0], prev_steer, wrap(s1["th"] - s0["th"]) / DT))
                    prev_steer = action[0]
                else:
                    lon.append((s0["spd"], action[1], (s1["spd"] - s0["spd"]) / DT))
                step += 1
                if term or trunc or step > 400:
                    break
        lf, lr = env.agent.FRONT_WHEELBASE, env.agent.REAR_WHEELBASE
        max_speed = env.agent.max_speed_m_s
        length, width = env.agent.LENGTH, env.agent.WIDTH
    finally:
        env.close()
    return np.array(lon), np.array(lat), dict(lf=lf, lr=lr, max_speed=max_speed, length=length, width=width)


def fit_longitudinal(lon, max_speed):
    v, u, a = lon.T
    # exclude samples where the speed cap or standstill makes the response non-smooth
    keep = (v > 0.5) & (v < max_speed - 0.5)
    v, u, a = v[keep], u[keep], a[keep]
    X = np.stack([np.maximum(u, 0), np.minimum(u, 0), v, np.ones_like(v)], 1)
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - a)**2)))
    r2 = float(1 - np.sum((pred - a)**2) / np.sum((a - a.mean())**2))
    return dict(c_throttle=coef[0], c_brake=coef[1], c_drag=coef[2], c_const=coef[3]), rmse, r2


def fit_lateral(lat, lf, lr):
    v, s, s_prev, w = lat.T
    keep = v > 1.0
    v, s, s_prev, w = v[keep], s[keep], s_prev[keep], w[keep]
    L = lf + lr

    def fit(cmd):
        model = lambda p: v / L * np.tan(p[0] * cmd) / (1 + p[1] * v**2)
        res = least_squares(lambda p: model(p) - w, x0=[0.6, 0.001], bounds=([0.05, 0.0], [1.5, 0.1]))
        r2 = float(1 - np.sum(res.fun**2) / np.sum((w - w.mean())**2))
        return res.x, float(np.sqrt(np.mean(res.fun**2))), r2

    (k, k_us), rmse, r2 = fit(s_prev)
    _, _, r2_no_delay = fit(s)
    return float(k), float(k_us), rmse, r2, r2_no_delay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lon, lat, geom = collect(args.episodes, args.seed)
    lon_coef, lon_rmse, lon_r2 = fit_longitudinal(lon, geom["max_speed"])
    k_steer, k_understeer, lat_rmse, lat_r2, lat_r2_no_delay = fit_lateral(lat, geom["lf"], geom["lr"])

    params = dict(
        dt=DT,
        lf=geom["lf"],
        lr=geom["lr"],
        length=geom["length"],
        width=geom["width"],
        max_speed=geom["max_speed"],
        k_steer=k_steer,
        k_understeer=k_understeer,
        steer_delay_steps=1,
        **{k: float(v) for k, v in lon_coef.items()},
        fit=dict(
            n_longitudinal=len(lon),
            n_lateral=len(lat),
            longitudinal_rmse=lon_rmse,
            longitudinal_r2=lon_r2,
            lateral_rmse=lat_rmse,
            lateral_r2=lat_r2,
            lateral_r2_without_delay=lat_r2_no_delay,
        ),
    )
    OUT.write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps(params, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

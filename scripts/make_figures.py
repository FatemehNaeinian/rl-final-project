"""Example figures for the report, rendered with MetaDrive's top-down renderer (headless).

  figures/scenario_densities.png  the intersection at low vs high traffic density (same seed)
  figures/mpc_left_turn.png       filmstrip: MPC completing a left turn in high-density traffic
  figures/mpc_road_cost.png       same seed, MPC without vs with the road-edge cost (the off-road failure)
  figures/idm_crash.png           filmstrip: a failure of the rule-based IDM baseline

The ego vehicle's planned route is overlaid as a dashed line and the ego is circled. Development seeds only.

Usage:  python scripts/make_figures.py [--out figures]
"""
import argparse
import os
import pathlib

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drive_compare.controllers import make_controller  # noqa: E402
from drive_compare.controllers.mpc import MPCController  # noqa: E402
from drive_compare.controllers.predictive import CostWeights  # noqa: E402
from drive_compare.env import DT, make_env  # noqa: E402
from drive_compare.metrics import classify_outcome  # noqa: E402
from drive_compare.obs import StructuredObsWrapper  # noqa: E402

SCREEN = 600
SCALING = 4.5  # pixels per metre -> 133 m field of view
CAMERA = (70.0, 8.0)  # world point at the image centre (the intersection)
OUTCOME_TEXT = {"success": "reached goal", "crash_vehicle": "crashed into a vehicle", "out_of_road": "left the road",
                "crash_other": "crashed", "timeout": "timed out"}


def to_pixels(xy):
    """World (x, y) -> image (col, row) for the fixed camera above (y points up in the image)."""
    xy = np.atleast_2d(xy)
    return np.column_stack([SCREEN / 2 + (xy[:, 0] - CAMERA[0]) * SCALING,
                            SCREEN / 2 - (xy[:, 1] - CAMERA[1]) * SCALING])


def run(controller, density, seed, max_steps=600):
    """Run one episode, rendering every step. Returns frames, ego positions, route, outcome."""
    env = StructuredObsWrapper(make_env(density, "dev"))
    frames, ego, speeds = [], [], []
    try:
        obs, info = env.reset(seed=seed)
        ws = info["world_state"]
        controller.reset(env, ws)
        route = ws.route.points[:, :2].copy()
        for _ in range(max_steps):
            obs, _, term, trunc, info = env.step(controller.act(obs, ws))
            ws = info["world_state"]
            frames.append(env.unwrapped.render(mode="topdown", window=False, screen_size=(SCREEN, SCREEN),
                                               camera_position=CAMERA, scaling=SCALING,
                                               draw_target_vehicle_trajectory=True).copy())
            ego.append((ws.x, ws.y))
            speeds.append(ws.v)
            if term or trunc:
                break
        outcome = classify_outcome(info)
    finally:
        env.close()
    return dict(frames=frames, ego=np.array(ego), speeds=np.array(speeds), route=route, outcome=outcome,
                turn=ws.route.turn)


def draw(ax, ep, k, title):
    ax.imshow(ep["frames"][k])
    r = to_pixels(ep["route"])
    ax.plot(r[:, 0], r[:, 1], "--", color="tab:blue", lw=1.2, alpha=0.7)
    e = to_pixels(ep["ego"][k])[0]
    ax.add_patch(plt.Circle(e, 14, fill=False, color="red", lw=2))
    ax.set_title(title, fontsize=10)
    ax.set_xlim(40, SCREEN - 40)
    ax.set_ylim(SCREEN - 40, 40)
    ax.axis("off")


def filmstrip(ep, n, suptitle, path, frac=(0.15, 0.45, 0.7, 1.0)):
    idx = [min(int(f * (len(ep["frames"]) - 1)), len(ep["frames"]) - 1) for f in frac[:n]]
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9))
    for ax, k in zip(axes, idx):
        extra = f" ({OUTCOME_TEXT[ep['outcome']]})" if k == len(ep["frames"]) - 1 else ""
        draw(ax, ep, k, f"t = {(k + 1) * DT:.1f} s, {ep['speeds'][k]:.1f} m/s{extra}")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. scenario at both densities: same seed, ~5 s in (MetaDrive spawns traffic as the ego approaches)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, density in zip(axes, ["low", "high"]):
        ep = run(make_controller("mpc"), density, seed=203, max_steps=50)
        draw(ax, ep, len(ep["frames"]) - 1, f"{density} density ({'0.1' if density == 'low' else '0.3'})")
    fig.suptitle("Scenario: 4-way intersection, ego (red circle) and its route (dashed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "scenario_densities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / "scenario_densities.png")

    # 2. MPC left turn in high density
    ep = run(make_controller("mpc"), "high", seed=214)
    filmstrip(ep, 4, f"MPC, left turn, high density (dev seed 214): {OUTCOME_TEXT[ep['outcome']]}",
              out / "mpc_left_turn.png")

    # 3. road-edge cost: same seed, without vs with, zoomed on the ego while it waits for crossing traffic
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6))
    without = run(MPCController(CostWeights(w_road=0.0)), "high", seed=214)
    with_cost = run(MPCController(CostWeights()), "high", seed=214)
    k_fail = len(without["frames"]) - 1  # the moment the episode ended
    for ax, ep, k, label in [
        (axes[0], without, k_fail, f"without road-edge cost\nsteers away from traffic, {OUTCOME_TEXT[without['outcome']]}"),
        (axes[1], with_cost, k_fail, f"with road-edge cost\nbrakes and waits, then {OUTCOME_TEXT[with_cost['outcome']]}"),
    ]:
        draw(ax, ep, k, f"MPC {label}\n(t = {(k + 1) * DT:.1f} s)")
        c = to_pixels(ep["ego"][k])[0]
        ax.set_xlim(c[0] - 110, c[0] + 170)
        ax.set_ylim(c[1] + 110, c[1] - 170)
    fig.suptitle("Left turn with crossing traffic (dev seed 214, high density)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "mpc_road_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / "mpc_road_cost.png")

    # 4. an IDM failure in high density (first dev seed where IDM crashes)
    for seed in range(200, 220):
        ep = run(make_controller("idm"), "high", seed=seed)
        if ep["outcome"] == "crash_vehicle":
            filmstrip(ep, 3, f"Rule-based IDM baseline, {ep['turn']} route, high density (dev seed {seed}): "
                      f"{OUTCOME_TEXT[ep['outcome']]}", out / "idm_crash.png", frac=(0.3, 0.75, 1.0))
            break


if __name__ == "__main__":
    main()

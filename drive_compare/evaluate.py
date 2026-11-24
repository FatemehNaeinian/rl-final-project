"""Run controllers over held-out seeds and traffic densities; write per-episode CSV and a summary.

Examples:
  python -m drive_compare.evaluate --controller idm --n 50
  python -m drive_compare.evaluate --controller mpc --n 50
  python -m drive_compare.evaluate --controller tracker --densities 0 --n 20     # no-traffic sanity check
  python -m drive_compare.evaluate --summarize results/*.csv                      # compare saved runs

Per episode it records: outcome (success / crash_vehicle / out_of_road / crash_other / timeout), travel
time (successes), route completion, minimum box-to-box clearance, and per-step decision time (only the
time inside controller.act). With --save-traj it also stores each trajectory for failure analysis.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from .controllers import make_controller
from .env import DENSITIES, DT, make_env, split_seeds
from .metrics import OUTCOMES, classify_outcome, clearance, mcnemar_exact, wilson_ci
from .obs import StructuredObsWrapper


def run_episode(env, controller, seed: int, save_traj: bool = False) -> tuple[dict, dict | None]:
    obs, info = env.reset(seed=seed)
    ws = info["world_state"]
    controller.reset(env, ws)
    decision_times, min_clear = [], float("inf")
    traj = {"ego": [], "action": [], "traffic": []} if save_traj else None
    while True:
        t0 = time.perf_counter()
        action = controller.act(obs, ws)
        decision_times.append(time.perf_counter() - t0)
        obs, _, terminated, truncated, info = env.step(action)
        ws = info["world_state"]
        min_clear = min(min_clear, clearance(ws))
        if save_traj:
            traj["ego"].append(ws.ego_state)
            traj["action"].append(np.asarray(action, dtype=float))
            traj["traffic"].append(np.column_stack([ws.traffic_xy, ws.traffic_yaw]))
        if terminated or truncated:
            break
    outcome = classify_outcome(info)
    steps = len(decision_times)
    dts = np.asarray(decision_times) * 1000.0
    row = dict(
        seed=seed,
        turn=ws.route.turn,
        outcome=outcome,
        success=outcome == "success",
        steps=steps,
        travel_time=steps * DT if outcome == "success" else np.nan,
        route_completion=min(ws.s / ws.route.length, 1.0),
        min_clearance=min_clear,
        decision_ms_mean=float(dts.mean()),
        decision_ms_p95=float(np.percentile(dts, 95)),
        decision_ms_max=float(dts.max()),
    )
    return row, traj


def evaluate(controller_name: str, densities, split: str, n: int, controller_kwargs=None, save_traj_dir=None):
    rows = []
    for density in densities:
        controller = make_controller(controller_name, **(controller_kwargs or {}))
        env = StructuredObsWrapper(make_env(density, split))
        try:
            for seed in split_seeds(split, n):
                row, traj = run_episode(env, controller, seed, save_traj=save_traj_dir is not None)
                row.update(method=controller_name, density=density)
                rows.append(row)
                if traj is not None:
                    out = pathlib.Path(save_traj_dir) / f"{controller_name}_d{density}_s{seed}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out, ego=np.array(traj["ego"]), action=np.array(traj["action"]),
                        traffic=np.array(traj["traffic"], dtype=object), outcome=row["outcome"])
                print(f"[{controller_name} d={density}] seed {seed:4d} {row['turn']:8s} {row['outcome']:13s} "
                      f"steps={row['steps']:4d} clear={row['min_clearance']:.2f}m "
                      f"act={row['decision_ms_mean']:.2f}ms", flush=True)
        finally:
            env.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, density), g in df.groupby(["method", "density"], sort=False):
        n = len(g)
        rec = dict(method=method, density=density, n=n)
        for o in OUTCOMES:
            k = int((g["outcome"] == o).sum())
            lo, hi = wilson_ci(k, n)
            rec[o] = f"{k / n:.2f} [{lo:.2f},{hi:.2f}]"
        rec["travel_time_s"] = round(float(g["travel_time"].mean()), 2)
        rec["min_clearance_median_m"] = round(float(g["min_clearance"].replace(np.inf, np.nan).median()), 2)
        rec["decision_ms_mean"] = round(float(g["decision_ms_mean"].mean()), 3)
        rec["decision_ms_p95"] = round(float(g["decision_ms_p95"].max()), 3)
        out.append(rec)
    return pd.DataFrame(out)


def paired_success_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Exact McNemar test on success for every pair of methods, matched by (density, seed)."""
    methods = list(df["method"].unique())
    rows = []
    for density, g in df.groupby("density"):
        piv = g.pivot_table(index="seed", columns="method", values="success", aggfunc="first")
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                if a not in piv or b not in piv:
                    continue
                both = piv[[a, b]].dropna().astype(bool)
                a_only = int((both[a] & ~both[b]).sum())
                b_only = int((~both[a] & both[b]).sum())
                rows.append(dict(density=density, a=a, b=b, n=len(both), a_only=a_only, b_only=b_only,
                                 p_value=round(mcnemar_exact(a_only, b_only), 4)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=["idm", "expert", "tracker", "mpc"])
    ap.add_argument("--densities", nargs="+", default=["low", "high"],
                    help=f"names from {list(DENSITIES)} or numbers, e.g. 0 for no traffic")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--n", type=int, default=50, help="number of seeds per density")
    ap.add_argument("--out", help="CSV path (default results/<controller>_<split>.csv)")
    ap.add_argument("--save-traj", metavar="DIR", help="also save per-episode trajectories (.npz) here")
    ap.add_argument("--summarize", nargs="+", metavar="CSV", help="only summarize existing result CSVs")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    if args.summarize:
        df = pd.concat([pd.read_csv(p) for p in args.summarize], ignore_index=True)
    else:
        if not args.controller:
            ap.error("--controller is required unless --summarize is given")
        densities = [d if d in DENSITIES else float(d) for d in args.densities]
        kwargs = {}
        df = evaluate(args.controller, densities, args.split, args.n, kwargs, args.save_traj)
        out = pathlib.Path(args.out or f"results/{args.controller}_{args.split}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nsaved {len(df)} episodes -> {out}")
    print("\n" + summarize(df).to_string(index=False))
    if df["method"].nunique() > 1:
        print("\nPaired success tests (exact McNemar):\n" + paired_success_tests(df).to_string(index=False))


if __name__ == "__main__":
    main()

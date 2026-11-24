"""Solver and prediction diagnostics for the MPC in the real simulator.

Per step it records IPOPT's return status, iteration count and wall time, and stores the planned ego
trajectory so it can be compared with what actually happened (closed-loop prediction error at 0.5/1/2 s).

Usage:  python scripts/diagnose_mpc.py --density high --n 20 [--split dev] [--smooth 0.02 --time-limit wall
        --warm-start-duals]
"""
import argparse
import collections
import time

import numpy as np

from drive_compare.controllers.mpc import MPCController
from drive_compare.env import make_env, split_seeds
from drive_compare.metrics import classify_outcome
from drive_compare.obs import StructuredObsWrapper
from drive_compare.state import wrap_angle


class InstrumentedMPC(MPCController):

    def reset(self, env, ws):
        super().reset(env, ws)
        self.log = []

    def act(self, obs, ws):
        t0 = time.perf_counter()
        u = super().act(obs, ws)
        dt = time.perf_counter() - t0
        st = self.opti.stats()
        self.log.append(dict(status=st.get("return_status", "?"), iters=st.get("iter_count", -1), ms=dt * 1e3,
                             plan=self.X_guess.copy(), v=ws.v))  # X_guess = plan shifted by one step
        return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", default="high")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--time-limit", choices=["cpu", "wall"], default="cpu")
    ap.add_argument("--warm-start-duals", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ctrl = InstrumentedMPC(smooth=args.smooth, time_limit=args.time_limit, warm_start_duals=args.warm_start_duals)
    env = StructuredObsWrapper(make_env(args.density, args.split))
    status, iters, times, outcomes = collections.Counter(), [], [], collections.Counter()
    pred_err = {5: [], 10: [], 20: []}
    stopped_steps = stopped_non_success = 0
    try:
        for seed in split_seeds(args.split, args.n):
            obs, info = env.reset(seed=seed)
            ws = info["world_state"]
            ctrl.reset(env, ws)
            actual = []
            while True:
                obs, _, term, trunc, info = env.step(ctrl.act(obs, ws))
                ws = info["world_state"]
                actual.append(ws.ego_state)
                if term or trunc:
                    break
            outcomes[classify_outcome(info)] += 1
            actual = np.array(actual)
            for t, rec in enumerate(ctrl.log):
                status[rec["status"]] += 1
                iters.append(rec["iters"])
                times.append(rec["ms"])
                if rec["v"] < 0.3:
                    stopped_steps += 1
                    stopped_non_success += rec["status"] not in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
                # rec["plan"][j] is the planned state j+1 steps after this decision, i.e. actual[t + j]
                for h in pred_err:
                    if t + h - 1 < len(actual):
                        p, a = rec["plan"][h - 1], actual[t + h - 1]
                        pred_err[h].append((np.hypot(*(p[:2] - a[:2])), abs(wrap_angle(p[2] - a[2]))))
    finally:
        env.close()

    n = sum(status.values())
    conv = status["Solve_Succeeded"] + status["Solved_To_Acceptable_Level"]
    ms_all = np.array(times)
    print(f"SUMMARY | {args.label:28s} | {args.density:4s} | success {outcomes['success']}/{args.n} | "
          f"converged {100 * conv / n:5.1f}% | ms median {np.median(ms_all):4.1f} p95 {np.percentile(ms_all, 95):5.1f} "
          f"max {ms_all.max():5.0f} | >100ms {(ms_all > 100).sum()}")
    print(f"\n{args.n} {args.split} seeds, density={args.density}: outcomes {dict(outcomes)}")
    print(f"solves: {n}")
    for s, c in status.most_common():
        print(f"  {s:32s} {c:6d}  ({100 * c / n:.1f}%)")
    it, ms = np.array(iters), np.array(times)
    print(f"IPOPT iterations: median {np.median(it):.0f}, p95 {np.percentile(it, 95):.0f}, max {it.max()}")
    print(f"decision time ms: median {np.median(ms):.1f}, p95 {np.percentile(ms, 95):.1f}, max {ms.max():.1f}, "
          f"> 100 ms: {(ms > 100).sum()}")
    print(f"steps at standstill (v < 0.3 m/s): {stopped_steps}, of which solver not converged: {stopped_non_success}")
    print("closed-loop prediction error of the MPC plan vs what happened (median / p90):")
    for h, e in pred_err.items():
        e = np.array(e)
        print(f"  {h * 0.1:.1f} s: position {np.median(e[:, 0]):.2f} / {np.percentile(e[:, 0], 90):.2f} m, "
              f"heading {np.degrees(np.median(e[:, 1])):.1f} / {np.degrees(np.percentile(e[:, 1], 90)):.1f} deg")


if __name__ == "__main__":
    main()

"""Nonlinear MPC with CasADi (Opti stack + IPOPT) on the shared vehicle model and traffic prediction.

The optimization problem is built once; each step only updates its parameters (current state, reference,
predicted traffic) and warm-starts from the previous solution shifted by one step. The first action of the
optimal sequence is applied (receding horizon). See predictive.py for the stage cost.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from ..vehicle_model import NU, NX, load_params, step
from .base import Controller
from .predictive import CostWeights, build_reference, padded_traffic_prediction, road_band


class MPCController(Controller):
    name = "mpc"

    def __init__(self, weights: CostWeights | None = None, max_iter: int = 60, max_cpu_time: float = 0.08):
        self.p = load_params()
        self.w = weights or CostWeights()
        self.max_iter = max_iter
        self.max_cpu_time = max_cpu_time
        self._build()
        self.stats = {"solves": 0, "failures": 0}

    # ------------------------------------------------------------------ problem construction
    def _build(self):
        p, w = self.p, self.w
        H, M = w.horizon, w.max_vehicles
        opti = ca.Opti()
        X = opti.variable(NX, H + 1)
        U = opti.variable(NU, H)
        x0 = opti.parameter(NX)
        u_prev = opti.parameter(NU)
        ref_xy = opti.parameter(2, H)
        ref_yaw = opti.parameter(1, H)
        v_des = opti.parameter(1, H)
        tr_x = opti.parameter(M, H)
        tr_y = opti.parameter(M, H)
        tr_c = opti.parameter(M)
        tr_s = opti.parameter(M)
        tr_a = opti.parameter(M)
        tr_b = opti.parameter(M)
        lat_lo = opti.parameter()
        lat_hi = opti.parameter()

        opti.subject_to(X[:, 0] == x0)
        cost = 0
        for k in range(H):
            opti.subject_to(X[:, k + 1] == step(p, X[:, k], U[:, k], backend="casadi"))
            xk = X[:, k + 1]
            # contouring-style lateral error w.r.t. the reference point
            dx, dy = xk[0] - ref_xy[0, k], xk[1] - ref_xy[1, k]
            e_lat = -ca.sin(ref_yaw[k]) * dx + ca.cos(ref_yaw[k]) * dy
            cost += w.w_lateral * e_lat**2
            cost += w.w_road * (ca.fmax(e_lat - lat_hi, 0)**2 + ca.fmax(lat_lo - e_lat, 0)**2)
            cost += w.w_heading * (xk[2] - ref_yaw[k])**2
            cost += w.w_speed * (xk[3] - v_des[k])**2
            cost += w.w_steer * U[0, k]**2 + w.w_throttle * U[1, k]**2
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += w.w_steer_rate * du[0]**2 + w.w_throttle_rate * du[1]**2
            # proximity to each predicted traffic vehicle (ellipse in that vehicle's frame)
            rx, ry = xk[0] - tr_x[:, k], xk[1] - tr_y[:, k]
            lon = (tr_c * rx + tr_s * ry) / tr_a
            lat = (-tr_s * rx + tr_c * ry) / tr_b
            cost += w.w_proximity * ca.sum1(ca.exp(-0.5 * (lon**2 + lat**2)))
        opti.subject_to(opti.bounded(-1, U, 1))
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "expand": True}, {
            "print_level": 0,
            "sb": "yes",
            "max_iter": self.max_iter,
            "max_cpu_time": self.max_cpu_time,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "warm_start_init_point": "yes",
        })
        self.opti, self.X, self.U = opti, X, U
        self.par = dict(x0=x0, u_prev=u_prev, ref_xy=ref_xy, ref_yaw=ref_yaw, v_des=v_des, tr_x=tr_x, tr_y=tr_y,
                        tr_c=tr_c, tr_s=tr_s, tr_a=tr_a, tr_b=tr_b, lat_lo=lat_lo, lat_hi=lat_hi)

    # ------------------------------------------------------------------ control loop
    def reset(self, env, ws):
        H = self.w.horizon
        self.U_guess = np.tile([0.0, 0.3], (H, 1))
        self.X_guess = None
        self.v_plan = None
        self.u_prev = np.zeros(NU)

    def act(self, obs, ws):
        p, w, H = self.p, self.w, self.w.horizon
        x0 = ws.ego_state
        ref_xy, ref_yaw, v_des = build_reference(ws, w, p.dt, self.v_plan)
        txy, tc, ts, ta, tb = padded_traffic_prediction(ws, w, p.dt)
        opti, par = self.opti, self.par
        opti.set_value(par["x0"], x0)
        opti.set_value(par["u_prev"], self.u_prev)
        opti.set_value(par["ref_xy"], ref_xy.T)
        opti.set_value(par["ref_yaw"], ref_yaw[None])
        opti.set_value(par["v_des"], v_des[None])
        opti.set_value(par["tr_x"], txy[:, :, 0].T)
        opti.set_value(par["tr_y"], txy[:, :, 1].T)
        lo, hi = road_band(ws, w)
        for name, val in (("tr_c", tc), ("tr_s", ts), ("tr_a", ta), ("tr_b", tb), ("lat_lo", lo), ("lat_hi", hi)):
            opti.set_value(par[name], val)
        # warm start: previous solution shifted by one step (or a rollout of the guess at the start)
        if self.X_guess is None:
            xs = [x0]
            for k in range(H):
                xs.append(step(p, xs[-1], self.U_guess[k]))
            self.X_guess = np.array(xs)
        X_init = self.X_guess.copy()
        X_init[0] = x0
        opti.set_initial(self.X, X_init.T)
        opti.set_initial(self.U, self.U_guess.T)

        self.stats["solves"] += 1
        try:
            sol = opti.solve()
            U_opt, X_opt = np.array(sol.value(self.U)).T, np.array(sol.value(self.X)).T
        except RuntimeError:
            # IPOPT hit max_iter / max_cpu_time: its last iterate is usually still a usable plan
            self.stats["failures"] += 1
            U_opt, X_opt = np.array(opti.debug.value(self.U)).T, np.array(opti.debug.value(self.X)).T
            if not np.all(np.isfinite(U_opt)):
                U_opt = np.tile([0.0, -1.0], (H, 1))
                X_opt = self.X_guess
        U_opt = np.clip(U_opt, -1.0, 1.0)
        self.U_guess = np.vstack([U_opt[1:], U_opt[-1:]])
        self.X_guess = np.vstack([X_opt[1:], X_opt[-1:]])
        self.v_plan = X_opt[1:, 3]
        self.u_prev = U_opt[0]
        return U_opt[0].copy()

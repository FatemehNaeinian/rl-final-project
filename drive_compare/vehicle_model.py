"""Approximate ego-vehicle model used by the MPC, identified from MetaDrive by scripts/sysid.py.

State  x = [px, py, yaw, v, steer_prev]   (m, m, rad, m/s, normalized steering command of the previous step)
Action u = [steer, throttle_brake]        (MetaDrive's own normalized action, each in [-1, 1])

Planners optimize directly in MetaDrive's action space, so there is no separate inverse mapping in the
control loop and every controller outputs exactly the same kind of action.

Dynamics over one control period DT (0.1 s), fitted on an empty straight road:
  a        = c_throttle * max(u_tb, 0) + c_brake * min(u_tb, 0) + c_drag * v + c_const
  yaw_rate = v / (lf + lr) * tan(k_steer * steer_prev) / (1 + k_understeer * v^2)   # ONE-STEP steering delay
  px'  = px + v * cos(yaw) * DT,   py' = py + v * sin(yaw) * DT
  yaw' = yaw + yaw_rate * DT,      v'  = max(v + a * DT, 0),     steer_prev' = u_steer

The same `step` function runs on NumPy arrays (batched) and CasADi symbols
(for MPC): it only uses arithmetic, abs, and the trig functions of the backend passed in.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

PARAMS_FILE = pathlib.Path(__file__).with_name("vehicle_params.json")

NX = 5  # state dimension
NU = 2  # action dimension


@dataclass(frozen=True)
class VehicleParams:
    dt: float
    lf: float
    lr: float
    length: float
    width: float
    max_speed: float
    k_steer: float  # rad of effective front-wheel angle per unit of normalized steering
    k_understeer: float
    c_throttle: float  # m/s^2 at full throttle
    c_brake: float  # m/s^2 per unit of brake (multiplies a negative command)
    c_drag: float
    c_const: float

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


def load_params(path: str | pathlib.Path = PARAMS_FILE) -> VehicleParams:
    raw = json.loads(pathlib.Path(path).read_text())
    fields = VehicleParams.__dataclass_fields__
    return VehicleParams(**{k: raw[k] for k in fields})


class _Backend:
    def __init__(self, cos, sin, tan, stack):
        self.cos, self.sin, self.tan, self.stack = cos, sin, tan, stack


def _numpy_backend():
    return _Backend(np.cos, np.sin, np.tan, lambda xs: np.stack(xs, axis=-1))


def _casadi_backend():
    import casadi as ca
    return _Backend(ca.cos, ca.sin, ca.tan, lambda xs: ca.vertcat(*xs))


BACKENDS = {"numpy": _numpy_backend, "casadi": _casadi_backend}


def _abs(z, eps=0.0):
    """|z|, or the smooth sqrt(z^2 + eps^2) when eps > 0 (differentiable everywhere, for gradient-based MPC)."""
    return abs(z) if eps <= 0 else (z * z + eps * eps)**0.5


def _pos(z, eps=0.0):
    return (z + _abs(z, eps)) * 0.5


def _neg(z, eps=0.0):
    return (z - _abs(z, eps)) * 0.5


def acceleration(p: VehicleParams, v, u_tb, smooth: float = 0.0):
    """Longitudinal acceleration (m/s^2) for throttle/brake command u_tb at speed v."""
    return p.c_throttle * _pos(u_tb, smooth) + p.c_brake * _neg(u_tb, smooth) + p.c_drag * v + p.c_const


def yaw_rate(p: VehicleParams, v, steer_prev, tan):
    return v / p.wheelbase * tan(p.k_steer * steer_prev) / (1.0 + p.k_understeer * v * v)


def step(p: VehicleParams, x, u, backend: str = "numpy", smooth: float = 0.0):
    """One DT step. x: (..., 5), u: (..., 2) for numpy (batched over leading dims); 5x1 / 2x1 for casadi.

    smooth > 0 replaces the two kinks (throttle/brake switch at u=0, speed clamp at v=0) with smooth
    approximations of width ~smooth. Gradient-based solvers (IPOPT) converge better on the smooth model;
    Default 0 = the exact identified model.
    """
    B = BACKENDS[backend]()
    px, py, yaw, v, steer_prev = (x[..., i] for i in range(NX)) if backend == "numpy" else (x[i] for i in range(NX))
    us, utb = (u[..., 0], u[..., 1]) if backend == "numpy" else (u[0], u[1])
    a = acceleration(p, v, utb, smooth)
    v_next = _pos(v + a * p.dt, smooth)  # no reversing: MetaDrive brakes to a stop when enable_reverse=False
    return B.stack([
        px + v * B.cos(yaw) * p.dt,
        py + v * B.sin(yaw) * p.dt,
        yaw + yaw_rate(p, v, steer_prev, B.tan) * p.dt,
        v_next,
        us + 0.0 * v,  # keep dtype/device/shape consistent across backends
    ])


def rollout(p: VehicleParams, x0, U, backend: str = "numpy"):
    """Roll out an action sequence U (H, 2) from x0 (5,). Returns states (H+1, 5). numpy only."""
    xs = [x0]
    for k in range(len(U)):
        xs.append(step(p, xs[-1], U[k], backend))
    return np.stack(xs)


# ---- convenience inverses (for hand-written controllers / analysis, not needed by the MPC) ----

def throttle_for_accel(p: VehicleParams, a_des, v) -> float:
    """Normalized throttle/brake that produces acceleration a_des at speed v under the fitted model."""
    rest = p.c_drag * v + p.c_const
    if a_des >= rest:
        u = (a_des - rest) / p.c_throttle
    else:
        u = (a_des - rest) / p.c_brake
    return float(np.clip(u, -1.0, 1.0))


def steer_for_angle(p: VehicleParams, delta) -> float:
    """Normalized steering command for an effective front-wheel angle delta (rad)."""
    return float(np.clip(delta / p.k_steer, -1.0, 1.0))

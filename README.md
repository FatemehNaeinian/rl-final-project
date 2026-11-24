# MPC for driving through traffic in MetaDrive

ROB-GY 6323, Project 9: the MPC part. A nonlinear model predictive controller (CasADi + IPOPT) drives through
a MetaDrive four-way intersection with moving traffic. This repo also holds the scenario setup, identified vehicle
model, traffic prediction, evaluation pipeline, and reference baselines it is evaluated against.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10–3.12
pip install metadrive-simulator                          # or: pip install -e <path to a metadrive clone>
pip install -e ".[dev]"
pytest -q tests/                                         # 15 tests, ~10 s
```

Everything runs headless; never set `use_render=True` on a server with no display.

## Layout

| Path | What it is |
|---|---|
| `drive_compare/env.py` | `make_env(density, split)`: the scenario config. Intersection map `"X"`, densities `low=0.1` (~3 cars) / `high=0.3` (~9 cars), seed splits, 0.1 s control period |
| `drive_compare/state.py` | `WorldState`: everything a controller may know (ego state, cars within 50 m, route polyline, road edges, progress) |
| `drive_compare/obs.py` | `StructuredObsWrapper`: puts the `WorldState` in `info["world_state"]` each step (plus a fixed-size vector encoding of it) |
| `drive_compare/vehicle_model.py` | Identified kinematic bicycle model; one `step()` for NumPy and CasADi |
| `drive_compare/prediction.py` | Constant-velocity traffic prediction |
| `drive_compare/metrics.py` | Outcome classification, box-to-box clearance, Wilson CIs, exact McNemar test |
| `drive_compare/evaluate.py` | Runs a controller over seeds × densities → CSV + summary table + paired tests |
| `drive_compare/controllers/mpc.py` | **The MPC** |
| `drive_compare/controllers/predictive.py` | MPC cost weights (`CostWeights`) and helpers: reference, road band, padded traffic prediction |
| `drive_compare/controllers/baselines.py` | Reference baselines: `idm`, `expert`, `tracker` |
| `scripts/sysid.py` | Re-fits `vehicle_params.json` from simulator data |
| `scripts/validate_model.py` | Open-loop multi-step model error |
| `scripts/diagnose_mpc.py` | MPC solver status, iterations, decision time, closed-loop prediction error |
| `scripts/reproduce_expert.py` | Reproduces the documented performance of MetaDrive's pretrained expert |
| `scripts/make_figures.py` | Top-down example figures in `figures/` |
| `tests/` | Shared-interface tests and MPC behaviour tests |

## Evaluation rules

1. **Information.** Controllers only use what `act()` receives (`ws`, the `WorldState`); they never read the
   simulator directly. The `idm`/`expert` baselines are the exceptions; they are references.
2. **Actions.** Every controller outputs MetaDrive's normalized `[steer, throttle_brake] ∈ [-1, 1]²`. The MPC
   optimizes in that space through the identified model, so no conversion layer is needed.
3. **Seeds.** Tune on `dev` (200–299). Report on `test` (0–199; the plan's 50 = the first 50). Do not look at
   test results while tuning.
4. **Decision time** = time inside `act()` only, compared against the 100 ms control period. Report the median,
   95th and 99th percentiles and the fraction over budget, not just the mean.

## What we measured about the simulator

**Action → physics** (`scripts/sysid.py`, `drive_compare/vehicle_params.json`):

- Acceleration ≈ `3.36·throttle⁺ + 11.4·brake⁻ − 0.022·v − 0.42` m/s² (R² = 0.97). The speed cap is 80 km/h,
  and the car never reverses.
- **Steering acts with a one-step (0.1 s) delay.** Yaw rate ≈ `v/L · tan(0.68·steer_prev) / (1 + 0.002·v²)`
  (R² = 0.93; only 0.74 if the delay is ignored). The model therefore carries the previous steering command as
  a 5th state.
- Open-loop prediction error of the model (`scripts/validate_model.py`, median / 90th percentile):

  | horizon | position | heading | speed |
  |---|---|---|---|
  | 0.5 s | 0.08 / 0.26 m | 1.0 / 3.2° | 0.08 / 0.20 m/s |
  | 1.0 s | 0.16 / 0.46 m | 1.2 / 4.7° | 0.13 / 0.30 m/s |
  | 2.0 s | 0.46 / 0.98 m | 2.2 / 5.8° | 0.19 / 0.47 m/s |

**Scenario:** routes are 110–125 m (≈14 s). The seed picks the exit (left / straight / right; reported in the
`turn` column), and the same seed reproduces the same initial traffic. Crashes and leaving the road (including
touching a solid line) end the episode; there is a 60 s timeout.

## Results so far (dev seeds; not yet run on test seeds)

Success rate with 95% Wilson CIs:

| method | seeds per density | low density | high density | decision time |
|---|---|---|---|---|
| **`mpc`** (CasADi/IPOPT, 2 s horizon) | 100 | **1.00** [0.96, 1.00] | **0.97** [0.92, 0.99] | ~7 ms mean, 15–16 ms p95 |
| `expert` (MetaDrive's pretrained policy, own lidar observation) | 20 | 0.95 [0.76, 0.99] | 0.85 [0.64, 0.95] | 1.2–1.3 ms |
| `idm` (MetaDrive rule-based driver) | 20 | 0.75 [0.53, 0.89] | 0.50 [0.30, 0.70] | 0.1–0.2 ms |
| `tracker` (route following, no traffic awareness; 0 traffic) | 20 | 1.00 [0.84, 1.00] | n/a | 0.05 ms |

## The MPC

`drive_compare/controllers/mpc.py`. The optimization problem (2 s horizon, 20 steps) is built once. Each step
updates its parameters (state, reference, predicted traffic) and warm-starts from the previous solution shifted
by one step; the first action is applied (receding horizon). IPOPT is capped at 60 iterations / 80 ms of
wall-clock time and runs with one linear-algebra thread.

**Stage cost** (documented in `controllers/predictive.py`): lateral and heading error to a route reference
spaced by the planned speeds (along-track error is not penalized, so slowing for traffic is not fought);
curvature-limited target speed; control effort and control changes; a road-edge penalty that accounts for the
car's corners when it is yawed; and a Gaussian proximity cost around each predicted traffic vehicle (an ellipse
in that vehicle's frame).

**Model:** the identified model above, with its two kinks (throttle/brake switch, zero-speed clamp) slightly
smoothed (`vehicle_model.step(..., smooth=0.02)`) so the gradient-based solver converges reliably.

**Development findings:**
- **Road-edge cost.** The first version (no road-edge term) scored 0.95 / 0.70 on 20 seeds. Its failures were
  mostly *off road*: waiting for crossing traffic at a left turn, it steered away from the predicted cars and
  touched the solid road line, which MetaDrive counts as out of road. Adding the road-edge cost fixed that.
- **Solver robustness.** With the exact (non-smooth) model, IPOPT hit its iteration cap on 5–7% of steps,
  mostly around the intersection, and the unfinished plan was applied. Smoothing the model's two kinks raised
  convergence to 99.6–99.9%, cut the 95th-percentile decision time from ~43–60 ms to ~15 ms, and removed every
  timeout. On 100 dev seeds per density, success went from 0.97/0.91 to 1.00/0.97 (high density: 6 seeds fixed,
  0 broken, exact McNemar p = 0.03). Warm-starting IPOPT's multipliers as well gave no benefit.
- **Decision-time outliers.** About 0.25% of decisions exceed 100 ms. They are scheduling jitter outside IPOPT:
  replaying the same problem solves in ~4 ms.
- **Traffic cost shape.** A hinge cost (zero outside the safety ellipse) was much worse than the Gaussian
  (0.78/0.57 vs 1.00/0.97), almost all crashes. The Gaussian's never-zero tail makes the MPC react to traffic
  early, at the price of slight lateral wiggling next to adjacent-lane traffic.
- **Remaining high-density failures (3/100)** show the limits of the simple models:
  - Seed 288 crashes into an oncoming car turning left into the same exit road; constant-velocity prediction
    assumed it would go straight.
  - Seed 231 is hit by a crossing car while stopped mid-turn; MetaDrive's traffic does not yield.
  - Seed 200 leaves the road after wiggling next to oncoming adjacent-lane traffic.

**Checks.**
- `tests/test_mpc.py`, in controlled situations on the MPC's own model: pulling away from standstill, lane/speed
  tracking, stopping when the road is blocked, yielding to a car crossing at 90°, plan consistency with the
  model, and smooth vs exact model agreement.
- `scripts/diagnose_mpc.py` in the simulator: solver status, iterations, decision time, closed-loop prediction
  error. Example: `python scripts/diagnose_mpc.py --density high --n 20`.

## Reproducing published MetaDrive results

MetaDrive's pretrained expert against the expected values in MetaDrive's own tests
(`metadrive/tests/test_policy/test_expert_performance.py`; `scripts/reproduce_expert.py`, same evaluation
function, 10 episodes each). Results are in `results/reproduction/expert.json`.

| case | expected | measured | |
|---|---|---|---|
| CCC road, traffic, `use_mesh_terrain=True` | 300 < reward < 350 | 339.1 (success 1.00) | reproduced |
| CCC road, traffic, `use_mesh_terrain=False` | 300 < reward < 350 | 339.2 (success 1.00) | reproduced |
| CCC road, no traffic | success 1.0, 300–350 | 1.00, 339.3 | reproduced |
| XTXTS intersections, no traffic | success 1.0, reward > 400 | 1.00, 411.5 | reproduced |

## Commands

```bash
# MPC and baselines on held-out seeds (writes results/<name>_test.csv)
python -m drive_compare.evaluate --controller mpc --n 50
python -m drive_compare.evaluate --controller idm --n 50
python -m drive_compare.evaluate --controller expert --n 50

# compare saved runs (summary + paired McNemar tests per density)
python -m drive_compare.evaluate --summarize results/*_test.csv

# tuning / failure analysis on dev seeds, keeping trajectories
python -m drive_compare.evaluate --controller mpc --split dev --n 20 --save-traj results/trajectories

# solver diagnostics, figures
python scripts/diagnose_mpc.py --density high --n 20
python scripts/make_figures.py
```

## Code we build on (cite in the report)

MetaDrive (Li et al., TPAMI 2022): simulator, IDM policy, pretrained expert · CasADi (Andersson et al.,
Math. Prog. Comp. 2019) · IPOPT (Wächter & Biegler, Math. Prog. 2006) · PythonRobotics (Sakai et al., 2018):
reference kinematic MPC.

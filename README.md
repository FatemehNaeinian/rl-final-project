# PPO vs MPC vs MPPI in MetaDrive traffic

ROB-GY 6323, Project 9. This repo holds the **shared infrastructure** that all three controllers plug into,
so each method gets the same scenario, information, action space, vehicle model, traffic prediction and metrics.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10–3.12
pip install metadrive-simulator                          # or: pip install -e <path to a metadrive clone>
pip install torch --index-url https://download.pytorch.org/whl/cu126   # cu126 still supports GTX 10xx GPUs
pip install -e ".[dev]"
pytest -q tests/                                         # 9 tests, ~8 s
```

Everything runs headless; never set `use_render=True` on a server with no display.

## Layout

| Path | What it is |
|---|---|
| `drive_compare/env.py` | `make_env(density, split)`: the one scenario config. Intersection map `"X"`, densities `low=0.1` (~3 cars) / `high=0.3` (~9 cars), seed splits, 0.1 s control period |
| `drive_compare/state.py` | `WorldState`: everything a controller may know (ego state, cars within 50 m, route polyline and progress) |
| `drive_compare/obs.py` | `StructuredObsWrapper`: a 71-dimensional vector for PPO, plus `info["world_state"]` for planners, both built from the same `WorldState` |
| `drive_compare/vehicle_model.py` | Identified kinematic bicycle model; one `step()` for NumPy, PyTorch (MPPI) and CasADi (MPC) |
| `drive_compare/prediction.py` | Constant-velocity traffic prediction (shared by MPC and MPPI) |
| `drive_compare/metrics.py` | Outcome classification, box-to-box clearance, Wilson CIs, exact McNemar test |
| `drive_compare/evaluate.py` | Runs any controller over seeds × densities → CSV + summary table + paired tests |
| `drive_compare/controllers/` | `Controller` interface; baselines (`idm`, `expert`, `tracker`); `ppo`; `mpc` (working); **`mppi` skeleton to implement** |
| `scripts/sysid.py` | Re-fits `vehicle_params.json` from simulator data |
| `scripts/validate_model.py` | Open-loop multi-step model error (for the report) |
| `scripts/train_ppo.py` | Stable-Baselines3 PPO training (adapted from MetaDrive's training tutorial) |

## The rules that keep the comparison fair

1. **Same information.** Controllers only use `obs` (PPO) or `ws` (planners) passed to `act()`; never read the simulator
   directly. (The `idm`/`expert` baselines are the only exceptions; they are references, not compared methods.)
2. **Same actions.** Every controller outputs MetaDrive's normalized `[steer, throttle_brake] ∈ [-1, 1]²`. MPC/MPPI
   optimize in that space through the shared model, so no extra conversion layer is needed.
3. **Same model and prediction** for MPC and MPPI: `vehicle_model.step(..., backend=...)` and
   `prediction.predict_constant_velocity`. Use `CostWeights` in `controllers/predictive.py` for both.
4. **Seeds.** Tune on `dev` (200–299). PPO trains on `train` (1000–5999). Report on `test` (0–199; the plan's
   50 = the first 50). Do not look at test results while tuning.
5. **Decision time** = time inside `act()` only; it's compared against the 100 ms control period. Report the
   hardware (MPPI on GPU vs CasADi on CPU).

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
`turn` column), and the same seed reproduces the same initial traffic. Crashes and leaving the road end the
episode; there is a 60 s timeout.

**Results so far** (20 dev seeds per density; success rate with 95% Wilson CIs; not yet run on test seeds):

| method | low density success | high density success | mean decision time |
|---|---|---|---|
| `tracker` (route following, no traffic awareness; 0 traffic) | 1.00 [0.84, 1.00] | n/a | 0.05 ms |
| `idm` (MetaDrive rule-based driver) | 0.75 [0.53, 0.89] | 0.50 [0.30, 0.70] | 0.1–0.2 ms |
| `expert` (MetaDrive's pretrained PPO, own lidar observation) | 0.95 [0.76, 0.99] | 0.85 [0.64, 0.95] | 1.2–1.3 ms |
| first `ppo` run (1M steps, MetaDrive default reward)* | 0.20 [0.08, 0.42] | 0.10 [0.03, 0.30] | 0.3 ms |
| `mpc` (CasADi/IPOPT, 2 s horizon), **100 dev seeds** | 1.00 [0.96, 1.00] | 0.97 [0.92, 0.99] | ~7 ms mean, 15–16 ms p95 |

\* trained on the earlier 69-dim observation; it will not load with the current 71-dim one, so retrain.

**MPC notes.**
- **Road-edge cost.** The first version (no road-edge term) scored 0.95 / 0.70 on 20 seeds. Its failures were
  mostly *off road*: waiting for crossing traffic at a left turn, it steered away from the predicted cars and
  touched the solid road line, which MetaDrive counts as out of road. Adding the road-edge cost fixed that.
- **Solver robustness.** With the exact (non-smooth) model, IPOPT hit its iteration cap on 5–7% of steps,
  mostly around the intersection, and the unfinished plan was applied. Smoothing the model's two kinks
  (`smooth=0.02`) raised convergence to 99.6–99.9%, cut the 95th-percentile decision time from ~43–60 ms to
  ~15 ms, and removed every timeout. On 100 dev seeds per density, success went from 0.97/0.91 to 1.00/0.97
  (high density: 6 seeds fixed, 0 broken, exact McNemar p = 0.03). Rare >100 ms decisions (~0.25% of steps)
  are scheduling jitter outside IPOPT: replaying the same problem solves in ~4 ms.
- **Traffic cost shape.** A hinge cost (zero outside the safety ellipse) was much worse than the Gaussian
  (0.78/0.57 vs 1.00/0.97). The Gaussian's never-zero tail makes MPC cautious early, at the price of slight
  lateral wiggling next to adjacent-lane traffic.
- **Remaining high-density failures (3/100):**
  - Seed 200 leaves the road after wiggling next to oncoming adjacent-lane traffic.
  - Seed 288 crashes into an oncoming car turning left into the same exit road; constant-velocity prediction
    assumed it would go straight.
  - Seed 231 is hit by a crossing car while stopped mid-turn; the traffic doesn't yield.
- **Checks.** `tests/test_mpc.py` covers pulling away from standstill, lane/speed tracking, stopping behind a
  blocked road, yielding to a crossing car, and plan consistency with the model. `scripts/diagnose_mpc.py`
  reports solver status, iterations, decision time and closed-loop prediction error.

**First PPO run: it never learned to turn.** It succeeded on 6 of 12 straight routes and 0 of 34 left/right
routes, and its successful runs took 9.4 s against 14.5 s for `tracker`. It drives too fast and leaves the road
at the turn. With MetaDrive's default reward, leaving the road costs only −5, while getting ~70% of the way earns
~+80 of progress reward, so speeding is nearly as good as finishing. Dev success plateaued at 20–30% from 200k
steps. Next step: raise the penalties, e.g. `--crash-penalty 40 --out-of-road-penalty 40 --success-reward 50`,
and/or lower `--speed-reward`.

## Reproducing published MetaDrive baselines

These check that our MetaDrive + Stable-Baselines3 setup matches published numbers, before we trust our own.
Results are in `results/reproduction/`.

**1. MetaDrive's official SB3 PPO tutorial** (`metadrive/documentation/source/training.ipynb`;
`scripts/reproduce_sb3_tutorial.py`). We used the same config (curve map `C`, 3×3 discrete actions, no traffic,
horizon 500), the same PPO settings (`MlpPolicy`, `n_steps=4096`, 4 workers), 300k steps, and the same evaluation
(one deterministic episode).

| | episode reward | reached goal |
|---|---|---|
| published (seed 0) | 168.7 | yes |
| ours, seed 0 | 169.7 | yes |
| ours, seed 1 | 168.1 | yes |
| ours, seed 2 | 143.2 | no (left the road) |

Seed 0, the tutorial's seed, matches within 0.6%, and 2 of 3 seeds reproduce the result. The third shows the
seed-to-seed variance that a single-seed tutorial hides, which is one more reason to report several PPO seeds.
Each run took ~34 min on 4 CPU workers.

**2. MetaDrive's pretrained PPO expert** against the expected values in MetaDrive's own tests
(`metadrive/tests/test_policy/test_expert_performance.py`; `scripts/reproduce_expert.py`, same evaluation
function, 10 episodes each).

| case | expected | measured | |
|---|---|---|---|
| CCC road, traffic, `use_mesh_terrain=True` | 300 < reward < 350 | 339.1 (success 1.00) | reproduced |
| CCC road, traffic, `use_mesh_terrain=False` | 300 < reward < 350 | 339.2 (success 1.00) | reproduced |
| CCC road, no traffic | success 1.0, 300–350 | 1.00, 339.3 | reproduced |
| XTXTS intersections, no traffic | success 1.0, reward > 400 | 1.00, 411.5 | reproduced |

**Not reproduced: the MetaDrive paper's generalization experiment** (Li et al., Sec. V-B, Fig. 8: PPO/SAC trained
on N maps, tested on unseen maps). The paper gives those success rates only as a plot, with no numbers, and does
not state the N values, traffic density, block count or training budget. It used RLlib with 5 seeds (PPO settings
in its Table II). A comparison would rely on values read off a figure under guessed settings, so we cite it
qualitatively instead: more training maps → better test success; SAC was more stable than PPO.

## Commands

```bash
# baselines / any controller on held-out seeds (writes results/<name>_test.csv)
python -m drive_compare.evaluate --controller idm --n 50
python -m drive_compare.evaluate --controller expert --n 50

# PPO
python scripts/train_ppo.py --timesteps 2000000 --envs 12 --out runs/ppo \
    --crash-penalty 40 --out-of-road-penalty 40 --success-reward 50              # ~500-650 steps/s, ~1M steps in 30 min
tensorboard --logdir runs/
python -m drive_compare.evaluate --controller ppo --model runs/ppo/best_model.zip --n 50

# compare everything (summary + paired McNemar tests per density)
python -m drive_compare.evaluate --summarize results/*_test.csv

# failure analysis: keep trajectories
python -m drive_compare.evaluate --controller mppi --split dev --n 20 --save-traj results/trajectories
```

Training speed: about 500–650 env steps/s on the lab machine. Going from 12 to 28 workers did **not** speed it
up (514 vs ~500 steps/s), so the bottleneck is elsewhere, probably PPO's CPU gradient updates competing with the
workers. `--device cuda` or fewer `n_epochs` are untested options.

## MPC (done) / MPPI (to implement)

`drive_compare/controllers/mpc.py` is a working nonlinear MPC. The problem is built once, parameters are updated
each step, it warm-starts from the shifted previous solution, and IPOPT is capped at 60 iterations / 80 ms of
wall-clock time. MPC plans on a slightly smoothed copy of the shared model (`vehicle_model.step(..., smooth=0.02)`;
the default `smooth=0` is unchanged for everyone else) and limits the solver to one linear-algebra thread, but only
in processes that create an MPC controller.
The stage cost (lateral + heading tracking, curvature-limited speed, effort, control changes, road edges,
ellipse proximity to constant-velocity-predicted traffic) is documented in `controllers/predictive.py`,
together with the shared helpers `build_reference`, `road_band` and `padded_traffic_prediction`.

MPPI should reuse exactly those helpers and `CostWeights`, written in torch, with
`pytorch_mppi.MPPI(..., step_dependent_dynamics=True)`. Fill in `MPPIController` in `predictive.py` and run it
with `--controller mppi`. Develop on `--split dev`.

## Code we build on (cite in the report)

MetaDrive (Li et al., TPAMI 2022): simulator, IDM policy, pretrained expert, training tutorial ·
Stable-Baselines3 (Raffin et al., JMLR 2021): PPO · pytorch_mppi (UM-ARM Lab): MPPI ·
PythonRobotics (Sakai et al., 2018): reference kinematic MPC · CasADi (Andersson et al., 2019).

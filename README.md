# Comparing PPO, MPC, and MPPI for Autonomous Driving in Traffic

ROB-GY 6323 (Reinforcement Learning and Optimal Control for Autonomous Systems I), Project 9.
Sai Sriya Tulluri, Grishma Balgi, Stavan Christian, Fatemeh Naeinian.

We compare proximal policy optimization (PPO), model predictive control (MPC), and model predictive path
integral control (MPPI) for a car driving through a MetaDrive four-way intersection with moving traffic, at two
traffic densities. The question is how the three methods trade off driving performance, safety, and computation
time.

**This `main` branch only contains this file.** Each method lives on its own branch:

| Branch | Contents | Owner | Status |
|---|---|---|---|
| [`fatemeh`](../../tree/fatemeh) | **MPC** (CasADi + IPOPT), plus the scenario setup, identified vehicle model, traffic prediction, evaluation pipeline, and reference baselines it is evaluated against | Fatemeh Naeinian | Working; evaluated on development seeds, held-out test evaluation pending |
| *to be added* | **PPO** | | |
| *to be added* | **MPPI** | | |

Each branch has its own `README.md` with setup instructions, commands, and results.

## Getting a branch

```bash
git clone -b fatemeh https://github.com/FatemehNaeinian/rl-final-project.git   # one branch
# or, in an existing clone:
git fetch origin && git switch fatemeh
```

## Adding your branch

1. Create your branch (e.g. `git switch -c <name>`), commit your work, and push it:
   `git push -u origin <name>`.
2. Add a row for it in the table above: edit this file on the `main` branch and commit.

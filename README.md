# 🏦 Reinforcement Learning for Monetary Policy

**42578 Advanced Business Analytics — DTU Business Analytics, Term 3**  
**Group 8** · Final Project · 2025

---

## Overview

This project frames **US monetary policy** as a **Reinforcement Learning (RL) control problem**. A trained agent learns — purely through trial and error — to adjust interest rates in a simulated economy and keep inflation, unemployment, and GDP growth close to central-bank targets. We compare the agent against classical econometric baselines (Taylor Rule) and an unguided random policy.

The project also ships an **interactive web simulator** where you can act as the Fed Chair yourself, or watch PPO/DDPG agents run the economy in real time.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Economic Background](#2-economic-background)
3. [Reinforcement Learning Formulation](#3-reinforcement-learning-formulation)
4. [The Simulation Environment](#4-the-simulation-environment)
5. [Agents & Baselines](#5-agents--baselines)
6. [Results](#6-results)
7. [Interactive Simulator (Web Dashboard)](#7-interactive-simulator-web-dashboard)
8. [Project Structure](#8-project-structure)
9. [Setup & Running](#9-setup--running)
10. [Data Sources](#10-data-sources)
11. [Dependencies](#11-dependencies)
12. [Team](#12-team)

---

## 1. Project Goal

> **Can a reinforcement learning agent learn a monetary policy that stabilises the economy as well as — or better than — a classical Taylor Rule?**

The Federal Reserve (and other central banks) periodically adjust the federal funds rate to pursue a **dual mandate**: price stability (inflation ≈ 2%) and maximum employment (unemployment ≈ 4.5%). Choosing the right rate is hard because:

- The economy responds with **lags** (monetary policy takes 12–18 months to fully transmit).
- The world is **stochastic**: supply shocks, recessions, and pandemics are unpredictable.
- The objectives can **conflict**: cutting rates to fight unemployment may reignite inflation.

We train RL agents on a calibrated macroeconomic simulator and evaluate whether they can learn these trade-offs from scratch.

---

## 2. Economic Background

### 2.1 The Taylor Rule

The Taylor Rule (Taylor, 1993) is the textbook prescription for optimal interest-rate setting:

```
r_t = r* + π_t + 0.5(π_t − π*) + 0.5(y_t − y*)
```

Where:
- `r*` = neutral real interest rate (~2.5%, Laubach-Williams 2024)
- `π_t` = current inflation, `π*` = target (2%)
- `y_t` = current output/GDP, `y*` = potential output

We implement an **augmented version** that also responds to unemployment:

```
Δr_t = 0.5(π_t − 2.0) + 0.5(g_t − 0.25) + 0.25(u_t − 4.5)
```

This is our primary econometric **benchmark** — a deterministic rule designed by economists to represent optimal policy.

### 2.2 Key Macro Relationships Encoded in the Simulator

| Relationship | Description | Parameter |
|---|---|---|
| **AR(1) persistence** | Inflation, unemployment, GDP all mean-revert with inertia | ρ_π=0.75, ρ_u=0.92, ρ_g=0.40 |
| **IS Curve** | Higher rates → lower GDP demand (credit/investment channel) | IS_coeff=0.25 |
| **Phillips Curve** | Higher output gap → higher inflation | κ=0.12 |
| **Okun's Law** | Lower GDP growth → higher unemployment | Okun=0.15 |
| **Lagged transmission** | Inflation responds slowly to rate hikes (sticky prices) | W_π=[0.10, 0.20, 0.15] |
| **Supply shocks** | Correlated shocks: cost-push raises π and lowers g simultaneously | ρ_shock=−0.50 |
| **Recession risk** | 1.5% monthly probability of a recession shock | P_recession=0.015 |

### 2.3 Why RL Instead of Econometrics?

Traditional econometric approaches (OLS, VAR, DSGE models) **estimate relationships** from data and often derive an optimal rule analytically. RL instead **learns a reaction function** by interacting with a simulated environment.

| Aspect | Classical Econometrics | Reinforcement Learning |
|---|---|---|
| Method | Estimate → derive rule | Trial and error → learn rule |
| Uncertainty | Confidence intervals | Exploration-exploitation trade-off |
| Nonlinearity | Usually linearised | Naturally captured by neural networks |
| Policy | Closed-form (Taylor Rule) | Implicit in learned neural network |
| Evaluation | In-sample/out-of-sample R² | Cumulative episode reward |

---

## 3. Reinforcement Learning Formulation

The monetary policy problem is cast as a **Markov Decision Process (MDP)**:

```
(S, A, P, R, γ)
```

| Component | Definition |
|---|---|
| **State S** | `[inflation, unemployment, gdp_growth, interest_rate, Δr_{t-1}, Δr_{t-2}]` — 6-dimensional |
| **Action A** | Continuous interest-rate change `Δr ∈ [−1, +1]` percentage points per month |
| **Transition P** | Stochastic AR(1) economic dynamics (calibrated to US monthly data) |
| **Reward R** | Negative quadratic loss: `−(π−2)² − 0.5(u−4.5)² − 0.5(g−0.25)² − 0.1(Δr)²` |
| **Discount γ** | 0.99 |

### Reward function intuition

```
R_t = −[ 1.0 × (π − 2.0)²        ← inflation deviation (double weight: primary mandate)
        + 0.5 × (u − 4.5)²        ← unemployment deviation
        + 0.5 × (g − 0.25)²       ← GDP growth deviation
        + 0.1 × (Δr_t)²  ]        ← rate-change smoothing (penalises abrupt moves)
```

The reward is always ≤ 0 and reaches exactly 0 only when all targets are simultaneously met. The agent must balance four competing objectives at once.

### Why include lagged actions in the state?

Monetary policy transmits with a **lag**: a rate hike today reduces inflation in 1–2 months, not immediately. By including the two most-recent rate changes (`Δr_{t-1}`, `Δr_{t-2}`) in the observation, the agent can see the "policy pipeline" it has already initiated and avoid overshooting.

---

## 4. The Simulation Environment

**File:** `environment/economic_env.py`

The environment implements `gymnasium.Env` with custom AR(1) macroeconomic dynamics. Historical data **only seeds the starting state** — from reset, the simulation evolves on its own equations.

### Transition equations

```
g_{t+1}  = ρ_g × g_t + (1−ρ_g) × G_SS − IS_gap − f_g(actions) + ε_g
π_{t+1}  = ρ_π × π_t + (1−ρ_π) × π_SS + κ × (g−G_SS) − f_π(actions) + ε_π
u_{t+1}  = ρ_u × u_t + (1−ρ_u) × U_SS + Okun × (G_SS − g) + ε_u
r_{t+1}  = clip(r_t + action, −0.5, 20.0)
```

Where `f(actions)` is a **three-lag weighted + nonlinear (signed-quadratic)** policy transmission function.

### Multi-country extension

**File:** `environment/multi_country_env.py`

A two-country (US + Canada) MDP modelling **cross-country monetary spillovers**:
- Each country has its own state and rate change.
- A **spillover term** transmits neighbour rate hikes to own inflation (trade and capital flow channel).
- A single PPO agent observes both countries' states (8-D) and outputs two actions simultaneously.
- This mirrors the **GNN message-passing** concept: countries are nodes, trade links are edges.

---

## 5. Agents & Baselines

| Policy | Type | Description |
|---|---|---|
| **PPO** | RL Agent | Proximal Policy Optimisation. Stochastic policy with clipped update ratio. 500k timesteps, MlpPolicy. |
| **DDPG** | RL Agent | Deep Deterministic Policy Gradient. Deterministic actor-critic. Same budget. |
| **Taylor Rule** | Econometric | Augmented Taylor Rule reacting to inflation + GDP + unemployment gaps. No training required. |
| **Random** | Baseline | Uniform random `Δr ∈ [−1, +1]`. Pure lower bound. |

### PPO vs DDPG

| Feature | PPO | DDPG |
|---|---|---|
| Policy type | Stochastic (samples from Gaussian) | Deterministic |
| Update style | On-policy with clipped ratio | Off-policy with replay buffer |
| Stability | High (clip prevents large updates) | Moderate (sensitive to hyperparameters) |
| Exploration | Built into stochastic policy | Separate Gaussian noise |

Both use **VecNormalize** to standardise the 6-D observation to zero-mean unit-variance, preventing high-variance variables (unemployment) from dominating the gradient.

---

## 6. Results

### 6.1 Simulation Evaluation (20 stochastic episodes)

| Policy | Mean Reward | vs Random | vs Taylor Rule |
|---|---|---|---|
| 🥇 **Taylor Rule** | **−52.1** | +86.8% | — |
| 🥈 **DDPG** | **−61.4** | +84.4% | −17.9% |
| 🥉 **PPO** | **−71.7** | +81.8% | −37.6% |
| **Random** | **−394.1** | — | — |

*Higher (less negative) is better. All rewards are negative — zero is the theoretical optimum.*

**Key insight:** Both RL agents beat Random by ~83% — they have genuinely learned the structure of monetary policy. The Taylor Rule still leads, which is expected: it encodes economic knowledge that the agent must rediscover from reward alone.

### 6.2 Real-Data Backtest (Jan 2017 – Dec 2024)

Policy outputs were run on the historical observation sequence (no simulated transitions). Implied interest-rate levels were compared against the actual Fed Funds Rate.

| Model | MAE (pp) | RMSE (pp) |
|---|---|---|
| **PPO** | **0.6481** | **0.7635** |
| **DDPG** | 0.6769 | 0.7681 |

PPO's implied rate path is **slightly closer** to the historical Fed Funds Rate than DDPG's.

### 6.3 Explainability (SHAP)

SHAP analysis was performed on the COVID shock period (2020):
- Both agents correctly identified **inflation** as the dominant driver of rate decisions.
- During the COVID crash, the agents responded to **negative GDP growth** and **rising unemployment** by cutting rates — consistent with the actual Fed response.

---

## 7. Interactive Simulator (Web Dashboard)

A real-time web dashboard lets you **play around with the simulation**. You can act as the Fed Chair yourself or watch trained agents run the economy.

![Dashboard Preview](docs/dashboard_preview.png)

### Features

#### 🎮 Interactive Simulator Tab

| Feature | Description |
|---|---|
| **Manual Control** | You set the interest-rate change each month using a slider (−1 to +1 pp). Observe how the economy reacts. |
| **Agent Auto-Run** | Select PPO, DDPG, Taylor Rule, or Random. The agent runs automatically at slow/normal/fast speed. |
| **Starting Period** | Choose any month from Jan 2017 – Dec 2024. Start from COVID (2020-01) or the 2022 inflation surge for stress testing. |
| **Live Charts** | 6 real-time charts: Inflation, Unemployment, GDP Growth, Interest Rate, Rate Changes taken, Cumulative Reward. |
| **Stats Panel** | Current values colour-coded green/yellow/red against targets. |

#### 📊 Policy Comparison Tab

| Feature | Description |
|---|---|
| **Run All Policies** | Executes every available agent over the same 95-month episode (seed=42). |
| **Rankings Table** | Sorted by total reward with 🥇🥈🥉, vs-Random delta, and performance bar. |
| **Side-by-Side Charts** | All 4 policies overlaid on the same Inflation/Unemployment/GDP/Rate charts. |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   React Dashboard                        │
│   ControlPanel  │  ChartGrid  │  CompareView            │
│   (policy,      │  (Recharts  │  (multi-policy          │
│    slider,      │   live      │   comparison,           │
│    controls)    │   charts)   │   results table)        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP /api/*
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                         │
│   /api/reset   /api/step   /api/compare                 │
│   /api/run-episode         /api/trajectory              │
├─────────────────────────────────────────────────────────┤
│   EconomicEnv   │   PPO.load()   │   DDPG.load()        │
│   (Gymnasium)   │   VecNormalize │   VecNormalize       │
└─────────────────────────────────────────────────────────┘
```

### Running the Simulator

**Step 1 — Backend** (in one terminal):
```bash
cd simulator
python3 -m uvicorn api:app --reload --port 8000
```

**Step 2 — Frontend** (in a second terminal):
```bash
cd simulator/frontend
npm run dev
```

**Step 3** — Open [http://localhost:3000](http://localhost:3000) in your browser.

> The green dot in the top-right confirms the backend is connected.

---

## 8. Project Structure

```
ABA_Group_8/
│
├── README.md
├── requirements.txt               ← Python dependencies
│
├── data/
│   ├── us_macro_data_real.csv     ← US monthly macro data (Jan 2017 – Dec 2024, FRED)
│   ├── ca_macro_data_real.csv     ← Canada monthly macro data (same period)
│   ├── us_macro_data.csv          ← Synthetic quarterly data (testing only)
│   └── generate_sample_data.py    ← Script that generated the synthetic data
│
├── environment/
│   ├── economic_env.py            ← US single-country Gymnasium environment
│   └── multi_country_env.py       ← US + Canada two-country environment
│
├── models/
│   ├── ppo_economic_policy.zip    ← Trained PPO model weights
│   ├── vec_normalize.pkl          ← PPO VecNormalize observation stats
│   ├── ddpg_economic_policy.zip   ← Trained DDPG model weights
│   ├── ddpg_vec_normalize.pkl     ← DDPG VecNormalize observation stats
│   └── ppo_multi_country.zip      ← Trained multi-country PPO model
│
├── train.py                       ← Train PPO + evaluate vs baselines
├── train_multi_country.py         ← Train multi-country PPO
│
├── rl_monetary_policy.ipynb       ← Main report notebook (data, training,
│                                     evaluation, SHAP, backtest, plots)
│
└── simulator/
    ├── api.py                     ← FastAPI backend (wraps env + models)
    ├── requirements_sim.txt       ← Extra deps (fastapi, uvicorn)
    └── frontend/
        ├── package.json
        ├── vite.config.js
        ├── index.html
        └── src/
            ├── App.jsx            ← Root component, global state
            ├── api.js             ← Fetch wrappers
            ├── index.css          ← Dark Bloomberg-terminal theme
            └── components/
                ├── ControlPanel.jsx  ← Policy selector, slider, buttons
                ├── ChartGrid.jsx     ← 6 live Recharts charts
                └── CompareView.jsx   ← Multi-policy comparison + table
```

---

## 9. Setup & Running

> **Pre-trained models are included** in `models/` — you do not need to retrain anything to run the notebook or the simulator.

### Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Python | 3.10 | `python3 --version` |
| pip | any recent | `pip --version` |
| Node.js | 18 | `node --version` |
| npm | 9 | `npm --version` |

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/aba-group8-rl-monetary-policy.git
cd aba-group8-rl-monetary-policy
```

---

### Step 2 — Python environment

We recommend a virtual environment to avoid dependency conflicts:

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install all Python dependencies
pip install -r requirements.txt
pip install -r simulator/requirements_sim.txt
```

> **Tip:** If you see a `numpy._core` import error when loading the saved models, upgrade NumPy:
> ```bash
> pip install "numpy>=2.0"
> ```

---

### Step 3 — Run the main notebook

```bash
jupyter notebook rl_monetary_policy.ipynb
```

The notebook is self-contained and covers end-to-end:
data exploration → environment calibration → PPO training → DDPG training → policy comparison → real-data backtest → SHAP explainability.

All outputs are pre-rendered so you can read it without re-running any cells.

---

### Step 4 — Run the interactive web simulator

The simulator requires **two terminals running simultaneously**.

**Terminal 1 — start the Python API backend:**

```bash
cd simulator
python3 -m uvicorn api:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[data]  Loaded 96 monthly rows from us_macro_data_real.csv
[model] PPO loaded
[model] DDPG loaded
```

**Terminal 2 — start the React frontend:**

```bash
cd simulator/frontend
npm install        # first time only — installs React, Recharts, Vite
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in ...ms
  ➜  Local:   http://localhost:3000/
```

**Open [http://localhost:3000](http://localhost:3000) in your browser.**

The green dot in the top-right of the dashboard confirms the backend is connected. If it shows red, make sure the backend terminal (Terminal 1) is still running.

---

### Step 5 — Retrain (optional — models are already included)

The `models/` directory already contains trained weights. Only run these if you want to experiment with new hyperparameters:

```bash
# Single-country PPO + evaluate vs baselines
python3 train.py

# Single-country DDPG
python3 train_ddpg.py

# Multi-country PPO (US + Canada spillovers)
python3 train_multi_country.py
```

Training takes ~5–15 minutes per agent on a modern laptop (CPU).

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: gymnasium` | Run `pip install -r requirements.txt` inside your venv |
| `ModuleNotFoundError: fastapi` | Run `pip install -r simulator/requirements_sim.txt` |
| Backend starts but PPO/DDPG show as unavailable | Check `models/` directory exists and contains `.zip` files |
| Frontend shows "API Offline" | Ensure Terminal 1 backend is running on port 8000 |
| `npm: command not found` | Install Node.js 18+ from [nodejs.org](https://nodejs.org) |
| Port 8000 already in use | Run `lsof -ti:8000 \| xargs kill` then restart the backend |

---

## 10. Data Sources

| Variable | Source | Frequency | Period |
|---|---|---|---|
| US Inflation (CPI YoY) | FRED — CPIAUCSL | Monthly | Jan 2017 – Dec 2024 |
| US Unemployment Rate | FRED — UNRATE | Monthly | Jan 2017 – Dec 2024 |
| US GDP Growth | FRED — GDP (annualised, converted to monthly) | Monthly | Jan 2017 – Dec 2024 |
| US Federal Funds Rate | FRED — FEDFUNDS | Monthly | Jan 2017 – Dec 2024 |
| Canada Inflation | Statistics Canada | Monthly | Jan 2017 – Dec 2024 |
| Canada Unemployment | Statistics Canada | Monthly | Jan 2017 – Dec 2024 |
| Canada Overnight Rate | Bank of Canada | Monthly | Jan 2017 – Dec 2024 |

The period was chosen to capture three major episodes: COVID shock (2020), post-COVID inflation surge (2021–2022), and the aggressive tightening cycle (2022–2024).

---

## 11. Dependencies

### Python (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | 0.29.1 | RL environment interface |
| `stable-baselines3` | 2.3.2 | PPO and DDPG implementations |
| `torch` | ≥2.2 | Neural network backend |
| `numpy` | ≥2.0 | Numerical computing |
| `pandas` | ≥2.0 | Data manipulation |
| `tensorboard` | ≥2.14 | Training curve visualisation |

### Simulator extras (`simulator/requirements_sim.txt`)

| Package | Purpose |
|---|---|
| `fastapi` | REST API backend |
| `uvicorn` | ASGI server |

### Frontend (`simulator/frontend/package.json`)

| Package | Purpose |
|---|---|
| `react` 18 | UI framework |
| `recharts` | Real-time charts |
| `vite` | Build tool / dev server |

---

## 12. Team

**Group 8 — DTU Business Analytics, Term 3, 42578 Advanced Business Analytics**

| Name | Contribution |
|---|---|
| Chinmay Dongarkar | RL environment design, PPO/DDPG training, simulator backend |
| [Team Member 2] | Data collection, economic model calibration |
| [Team Member 3] | SHAP explainability, multi-country extension |
| [Team Member 4] | Evaluation framework, real-data backtest |
| [Team Member 5] | Notebook report, visualisations |

---

## References

1. Taylor, J. B. (1993). *Discretion versus policy rules in practice.* Carnegie-Rochester Conference Series on Public Policy, 39, 195–214.
2. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Lillicrap, T. P. et al. (2016). *Continuous control with deep reinforcement learning (DDPG).* ICLR 2016.
4. Laubach, T. & Williams, J. (2003). *Measuring the natural rate of interest.* Review of Economics and Statistics, 85(4).
5. Okun, A. M. (1962). *Potential GNP: Its measurement and significance.* Proceedings of the Business and Economics Section of the American Statistical Association.
6. Brockman, G. et al. (2016). *OpenAI Gym.* arXiv:1606.01540.

---

*Built for educational and research purposes. The simulator does not represent real-world monetary policy advice.*

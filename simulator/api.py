"""
simulator/api.py — FastAPI backend for the Monetary Policy Simulator.

Wraps the existing EconomicEnv and pre-trained models (PPO, DDPG) to expose
an HTTP API consumed by the React dashboard.

Start:
    cd simulator
    uvicorn api:app --reload --port 8000

Or:
    python api.py
"""

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — api.py lives in simulator/, project root is one level up
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.economic_env import EconomicEnv  # noqa: E402

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_PATH    = PROJECT_ROOT / "data" / "us_macro_data_real.csv"
PPO_PATH     = PROJECT_ROOT / "models" / "ppo_economic_policy.zip"
DDPG_PATH    = PROJECT_ROOT / "models" / "ddpg_economic_policy.zip"
PPO_VN_PATH  = PROJECT_ROOT / "models" / "vec_normalize.pkl"
DDPG_VN_PATH = PROJECT_ROOT / "models" / "ddpg_vec_normalize.pkl"

# ---------------------------------------------------------------------------
# Global objects (single-user demo — no concurrency needed)
# ---------------------------------------------------------------------------
ppo_model   = None
ddpg_model  = None
ppo_vn      = None   # VecNormalize stats dict for PPO
ddpg_vn     = None   # VecNormalize stats dict for DDPG

df_global: Optional[pd.DataFrame] = None   # numeric columns only
df_raw:    Optional[pd.DataFrame] = None   # with Month column

sim_env:   Optional[EconomicEnv] = None
sim_state: dict = {}


# ---------------------------------------------------------------------------
# VecNormalize helpers
# ---------------------------------------------------------------------------

def _load_vn_stats(path: Path) -> dict:
    """Extract obs normalisation stats from a saved VecNormalize pickle."""
    with open(path, "rb") as f:
        vn = pickle.load(f)
    return {
        "mean": np.array(vn.obs_rms.mean, dtype=np.float32),
        "var":  np.array(vn.obs_rms.var,  dtype=np.float32),
        "clip": float(vn.clip_obs),
        "eps":  float(getattr(vn, "epsilon", 1e-8)),
    }


def _apply_norm(obs: np.ndarray, stats: dict) -> np.ndarray:
    normed = (obs - stats["mean"]) / np.sqrt(stats["var"] + stats["eps"])
    return np.clip(normed, -stats["clip"], stats["clip"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _load_data() -> None:
    global df_global, df_raw
    raw = pd.read_csv(DATA_PATH)
    df_raw = raw.copy()

    df = raw.rename(columns={
        "GDP Growth":         "gdp_growth",
        "Inflation":          "inflation",
        "Unemployment Rate ": "unemployment",
        "Unemployment Rate":  "unemployment",
        "Interest rate":      "interest_rate",
    })
    df_global = (
        df[["inflation", "unemployment", "gdp_growth", "interest_rate"]]
        .dropna()
        .reset_index(drop=True)
    )
    print(f"[data]  Loaded {len(df_global)} monthly rows from {DATA_PATH.name}")


def _load_models() -> None:
    global ppo_model, ddpg_model, ppo_vn, ddpg_vn

    # PPO
    try:
        from stable_baselines3 import PPO  # type: ignore
        if PPO_PATH.exists():
            ppo_model = PPO.load(str(PPO_PATH))
            print("[model] PPO loaded")
        if PPO_VN_PATH.exists():
            ppo_vn = _load_vn_stats(PPO_VN_PATH)
            print("[model] PPO VecNormalize stats loaded")
    except Exception as e:
        print(f"[model] PPO load failed: {e}")

    # DDPG
    try:
        from stable_baselines3 import DDPG  # type: ignore
        if DDPG_PATH.exists():
            ddpg_model = DDPG.load(str(DDPG_PATH))
            print("[model] DDPG loaded")
        if DDPG_VN_PATH.exists():
            ddpg_vn = _load_vn_stats(DDPG_VN_PATH)
            print("[model] DDPG VecNormalize stats loaded")
    except Exception as e:
        print(f"[model] DDPG load failed: {e}")


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _get_action(policy: str, obs: np.ndarray) -> float:
    """Return the scalar action for the given policy."""
    if policy == "random":
        return float(np.random.uniform(-1.0, 1.0))

    if policy == "taylor":
        # Augmented Taylor Rule: inflation gap + GDP gap
        rc = 0.5 * (float(obs[0]) - 2.0) + 0.5 * (float(obs[2]) - 0.25)
        return float(np.clip(rc, -1.0, 1.0))

    if policy == "ppo" and ppo_model is not None:
        o = _apply_norm(obs, ppo_vn) if ppo_vn else obs
        action, _ = ppo_model.predict(o.reshape(1, -1), deterministic=True)
        return float(action[0][0])

    if policy == "ddpg" and ddpg_model is not None:
        o = _apply_norm(obs, ddpg_vn) if ddpg_vn else obs
        action, _ = ddpg_model.predict(o.reshape(1, -1), deterministic=True)
        return float(action[0][0])

    return 0.0   # fallback: no change


def _obs_to_point(obs: np.ndarray, action: float, reward: float, step: int) -> dict:
    return {
        "step":          step,
        "inflation":     round(float(obs[0]), 4),
        "unemployment":  round(float(obs[1]), 4),
        "gdp_growth":    round(float(obs[2]), 4),
        "interest_rate": round(float(obs[3]), 4),
        "action":        round(float(action), 4),
        "reward":        round(float(reward), 4),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    _load_models()
    yield


app = FastAPI(title="Monetary Policy Simulator API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    policy:    str = "manual"
    start_idx: int = 0


class StepRequest(BaseModel):
    action: float = 0.0   # ignored for non-manual policies


class RunRequest(BaseModel):
    policy: str
    seed:   int = 42


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/policies")
def get_policies():
    """Return which policies are available (models loaded)."""
    return {
        "available": {
            "manual": True,
            "random": True,
            "taylor": True,
            "ppo":    ppo_model is not None,
            "ddpg":   ddpg_model is not None,
        },
        "data_length": len(df_global) if df_global is not None else 0,
    }


@app.get("/api/dates")
def get_dates():
    """Return month labels for the start-period selector."""
    if df_raw is not None and "Month" in df_raw.columns:
        months = df_raw["Month"].dropna().tolist()
    else:
        months = [f"Month {i + 1}" for i in range(len(df_global))]
    return {"dates": months}


@app.post("/api/reset")
def reset_simulation(req: ResetRequest):
    """Reset the interactive simulation."""
    global sim_env, sim_state

    if df_global is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    idx = int(np.clip(req.start_idx, 0, len(df_global) - 10))
    data_slice = df_global.iloc[idx:].reset_index(drop=True)

    sim_env = EconomicEnv(data_slice)
    obs, _ = sim_env.reset(seed=42)

    initial = _obs_to_point(obs, 0.0, 0.0, 0)
    sim_state = {
        "policy":     req.policy,
        "obs":        obs,
        "step":       0,
        "done":       False,
        "trajectory": [initial],
    }

    return {
        "state":   initial,
        "policy":  req.policy,
        "targets": {
            "inflation":    EconomicEnv.INFLATION_TARGET,
            "unemployment": EconomicEnv.UNEMPLOYMENT_TARGET,
            "gdp_growth":   EconomicEnv.GDP_TARGET,
            "neutral_rate": EconomicEnv.NEUTRAL_RATE,
        },
    }


@app.post("/api/step")
def step_simulation(req: StepRequest):
    """Advance the interactive simulation by one step."""
    global sim_state

    if sim_env is None:
        raise HTTPException(status_code=400, detail="Call /api/reset first")
    if sim_state.get("done"):
        raise HTTPException(status_code=400, detail="Episode is complete — call /api/reset")

    obs    = sim_state["obs"]
    policy = sim_state["policy"]

    # Determine action
    if policy == "manual":
        action = float(np.clip(req.action, -1.0, 1.0))
    else:
        action = _get_action(policy, obs)

    new_obs, rew, terminated, truncated, _ = sim_env.step(
        np.array([action], dtype=np.float32)
    )
    done = bool(terminated or truncated)

    sim_state["step"]  += 1
    sim_state["obs"]    = new_obs
    sim_state["done"]   = done

    pt = _obs_to_point(new_obs, action, float(rew), sim_state["step"])
    sim_state["trajectory"].append(pt)

    total_reward = sum(p["reward"] for p in sim_state["trajectory"])
    return {**pt, "done": done, "total_reward": round(total_reward, 2)}


@app.get("/api/trajectory")
def get_trajectory():
    """Return the full trajectory of the current interactive session."""
    if sim_env is None:
        raise HTTPException(status_code=400, detail="Call /api/reset first")
    traj = sim_state["trajectory"]
    return {
        "trajectory":   traj,
        "done":         sim_state["done"],
        "total_reward": round(sum(p["reward"] for p in traj), 2),
    }


@app.post("/api/run-episode")
def run_full_episode(req: RunRequest):
    """Run a complete episode from the beginning with the given policy."""
    if df_global is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    if req.policy not in ("ppo", "ddpg", "taylor", "random"):
        raise HTTPException(status_code=400, detail=f"Unknown policy: {req.policy}")

    env = EconomicEnv(df_global)
    obs, _ = env.reset(seed=req.seed)

    traj  = [_obs_to_point(obs, 0.0, 0.0, 0)]
    total = 0.0
    done  = False
    step  = 0

    while not done:
        action = _get_action(req.policy, obs)
        obs, rew, terminated, truncated, _ = env.step(
            np.array([action], dtype=np.float32)
        )
        done  = bool(terminated or truncated)
        step += 1
        total += float(rew)
        traj.append(_obs_to_point(obs, action, float(rew), step))

    return {
        "policy":       req.policy,
        "trajectory":   traj,
        "total_reward": round(total, 2),
        "steps":        step,
    }


@app.post("/api/compare")
def compare_all_policies():
    """Run one full episode per policy and return all results for comparison."""
    if df_global is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    policies = []
    if ppo_model:   policies.append("ppo")
    if ddpg_model:  policies.append("ddpg")
    policies += ["taylor", "random"]

    results = {}
    for pol in policies:
        env = EconomicEnv(df_global)
        obs, _ = env.reset(seed=42)

        traj  = [_obs_to_point(obs, 0.0, 0.0, 0)]
        total = 0.0
        done  = False
        step  = 0

        while not done:
            action = _get_action(pol, obs)
            obs, rew, terminated, truncated, _ = env.step(
                np.array([action], dtype=np.float32)
            )
            done  = bool(terminated or truncated)
            step += 1
            total += float(rew)
            traj.append(_obs_to_point(obs, action, float(rew), step))

        results[pol] = {
            "trajectory":   traj,
            "total_reward": round(total, 2),
            "steps":        step,
        }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

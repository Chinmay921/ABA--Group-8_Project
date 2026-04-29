"""
train_ddpg.py — Train a DDPG agent on the EconomicEnv.

Usage (from project root):
    python train_ddpg.py

The script will:
1. Load macroeconomic data from data/us_macro_data_real.csv
2. Build EconomicEnv (with the full named-shock catalogue)
3. Validate it against the Gymnasium API
4. Train a DDPG agent for TOTAL_TIMESTEPS
5. Save the trained model  → models/ddpg_economic_policy.zip
6. Save VecNormalize stats → models/ddpg_vec_normalize.pkl
7. Evaluate 20 episodes and print mean ± std vs PPO / Taylor Rule / Random

Re-run this script whenever the environment changes (e.g. after adding shock
events) so the saved weights reflect the current transition dynamics.
"""

import os

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.economic_env import EconomicEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH      = "data/us_macro_data_real.csv"
MODEL_DIR      = "models"
LOG_DIR        = "logs"
TOTAL_TIMESTEPS = 500_000

DDPG_CONFIG = {
    "policy":        "MlpPolicy",
    "verbose":       1,
    "learning_rate": 1e-3,
    "batch_size":    100,
    "gamma":         0.99,
    "tau":           0.005,     # soft-update coefficient (Polyak averaging)
    "buffer_size":   50_000,    # replay-buffer capacity
    "tensorboard_log": LOG_DIR,
}


# ---------------------------------------------------------------------------
# Helpers (shared with train.py)
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Make sure data/us_macro_data_real.csv is present."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={
        "GDP Growth":         "gdp_growth",
        "Inflation":          "inflation",
        "Unemployment Rate ": "unemployment",   # note trailing space in source
        "Unemployment Rate":  "unemployment",
        "Interest rate":      "interest_rate",
    })
    required = {"inflation", "unemployment", "gdp_growth", "interest_rate"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[["inflation", "unemployment", "gdp_growth", "interest_rate"]].dropna()
    print(f"[data]  Loaded {len(df)} timesteps from '{path}'")
    print(df.describe().round(2))
    return df


def make_env(df):
    def _init():
        return EconomicEnv(df)
    return _init


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_policy(policy_fn, df: pd.DataFrame,
                     n_episodes: int = 20, seed_offset: int = 0) -> np.ndarray:
    """Evaluate a callable policy on the raw (un-normalised) env."""
    env = EconomicEnv(df)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_rew, done = 0.0, False
        while not done:
            action = policy_fn(obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_rew += rew
            done = terminated or truncated
        rewards.append(ep_rew)
    return np.array(rewards)


def _evaluate_sb3(model, vec_norm_path: str, df: pd.DataFrame,
                  n_episodes: int = 20, seed_offset: int = 0) -> np.ndarray:
    """Evaluate an SB3 model with frozen VecNormalize statistics."""
    eval_raw = DummyVecEnv([make_env(df)])
    eval_env = VecNormalize.load(vec_norm_path, eval_raw)
    eval_env.training   = False   # do NOT update running stats during eval
    eval_env.norm_reward = False  # keep reward in original scale

    rewards = []
    for ep in range(n_episodes):
        try:
            eval_env.seed(seed_offset + ep)
        except Exception:
            pass
        obs  = eval_env.reset()
        ep_rew, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, dones, _ = eval_env.step(action)
            ep_rew += float(rew[0])
            done    = bool(dones[0])
        rewards.append(ep_rew)
    return np.array(rewards)


def _summary(label: str, arr: np.ndarray) -> float:
    print(f"  {label:<20}  mean={arr.mean():+.1f}  std={arr.std():.1f}"
          f"  best={arr.max():+.1f}  worst={arr.min():+.1f}")
    return float(arr.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Build vectorised + normalised training environment
    #    Separate VecNormalize instance from PPO — their running stats must
    #    not mix (different exploration patterns → different obs distributions).
    train_env = DummyVecEnv([make_env(df)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 3. Validate raw env
    print("\n[env]   Running Gymnasium compliance check …")
    check_env(EconomicEnv(df), warn=True)
    print("[env]   Environment OK\n")

    # 4. Initialise DDPG
    model = DDPG(env=train_env, **DDPG_CONFIG)

    # 5. Train
    print(f"[train] Starting DDPG training for {TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 6. Save
    save_path     = os.path.join(MODEL_DIR, "ddpg_economic_policy")
    vec_norm_path = os.path.join(MODEL_DIR, "ddpg_vec_normalize.pkl")
    model.save(save_path)
    train_env.save(vec_norm_path)
    print(f"\n[saved] Model        → {save_path}.zip")
    print(f"[saved] VecNormalize → {vec_norm_path}")

    # 7. Evaluate — 20 episodes per policy
    N_EVAL = 20
    EVAL_SEED = 7777
    print(f"\n[eval]  Evaluating {N_EVAL} episodes per policy …\n")

    def taylor(obs):
        rc = (
            0.5 * (float(obs[0]) - 2.0)
            + 0.5 * (float(obs[2]) - 0.25)
            - 0.5 * (float(obs[1]) - 4.5)
        )
        return np.array([np.clip(rc, -1.0, 1.0)], dtype=np.float32)

    def random_pol(obs):
        return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    ddpg_mean   = _summary("DDPG",       _evaluate_sb3(model, vec_norm_path, df,
                                                       n_episodes=N_EVAL,
                                                       seed_offset=EVAL_SEED))
    taylor_mean = _summary("Taylor Rule", _evaluate_policy(taylor, df,
                                                            n_episodes=N_EVAL,
                                                            seed_offset=EVAL_SEED + 1000))
    rand_mean   = _summary("Random",      _evaluate_policy(random_pol, df,
                                                            n_episodes=N_EVAL,
                                                            seed_offset=EVAL_SEED + 2000))

    print(f"\n  DDPG vs Taylor Rule : {ddpg_mean - taylor_mean:+.1f}")
    print(f"  DDPG vs Random      : {ddpg_mean - rand_mean:+.1f}")


if __name__ == "__main__":
    main()

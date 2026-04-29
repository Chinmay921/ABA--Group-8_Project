"""
train.py — Train a PPO agent on the EconomicEnv.

Usage:
    python train.py

The script will:
1. Load macroeconomic data from data/us_macro_data.csv
2. Build the EconomicEnv
3. Validate it against the Gymnasium API
4. Train a PPO agent
5. Save the trained model to models/ppo_economic_policy
"""

import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.economic_env import EconomicEnv


# ---------------------------------------------------------------------------
# Configuration — change these values without touching the rest of the script
# ---------------------------------------------------------------------------

DATA_PATH = "data/us_macro_data_real.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 500_000

# PPO hyperparameters (sensible defaults for a small dataset)
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "verbose": 1,
    "learning_rate": 3e-4,
    "n_steps": 512,     # collect ~5 full episodes before each update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,      # discount factor
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load and validate the macro dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run  python data/generate_sample_data.py  to create a synthetic dataset."
        )

    df = pd.read_csv(path)

    # Rename real-data columns to the standard names the environment expects
    df = df.rename(columns={
        "GDP Growth":         "gdp_growth",
        "Inflation":          "inflation",
        "Unemployment Rate ": "unemployment",    # note trailing space in source file
        "Unemployment Rate":  "unemployment",
        "Interest rate":      "interest_rate",
    })

    required_cols = {"inflation", "unemployment", "gdp_growth", "interest_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[["inflation", "unemployment", "gdp_growth", "interest_rate"]].dropna()
    print(f"[data]  Loaded {len(df)} timesteps from '{path}'")
    print(df.describe().round(2))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_env(df):
    """Factory — avoids lambda closure issues inside DummyVecEnv."""
    def _init():
        return EconomicEnv(df)
    return _init


def random_policy(obs: np.ndarray) -> np.ndarray:
    return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)


def taylor_rule_policy(obs: np.ndarray) -> np.ndarray:
    """
    Deterministic Taylor Rule baseline:
        Δr = 0.5 × (π − 2.0) + 0.5 × (g − 0.25)
    obs[0]=inflation, obs[2]=gdp_growth (stable in 6D obs space).
    Clipped to [-1, +1] to match the action space.
    """
    rate_change = 0.5 * (float(obs[0]) - 2.0) + 0.5 * (float(obs[2]) - 0.25)
    return np.array([np.clip(rate_change, -1.0, 1.0)], dtype=np.float32)


def evaluate_episodes(policy_fn, df: pd.DataFrame,
                      n_episodes: int = 20, seed_offset: int = 0) -> list:
    """
    Evaluate a policy on a raw (un-normalised) EconomicEnv.
    Used for the random and Taylor Rule baselines.
    Returns a list of total episode rewards.
    """
    env = EconomicEnv(df)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_rew = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_rew += rew
            done = terminated or truncated
        rewards.append(ep_rew)
    return rewards


def evaluate_ppo(model, vec_norm_path: str, df: pd.DataFrame,
                 n_episodes: int = 20) -> list:
    """
    Evaluate trained PPO using frozen VecNormalize statistics.

    Critical settings:
      eval_env.training = False   — do NOT update running stats during eval
      eval_env.norm_reward = False — keep reward in original scale so it is
                                     directly comparable to baseline rewards
    """
    eval_raw = DummyVecEnv([make_env(df)])
    eval_env = VecNormalize.load(vec_norm_path, eval_raw)
    eval_env.training = False
    eval_env.norm_reward = False

    rewards = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        ep_rew = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, dones, _ = eval_env.step(action)
            ep_rew += float(rew[0])
            done = bool(dones[0])
        rewards.append(ep_rew)
    return rewards


def print_eval_summary(label: str, rewards: list) -> float:
    arr = np.array(rewards)
    print(f"\n  {label}")
    print(f"    Mean  : {arr.mean():+.2f}")
    print(f"    Std   :  {arr.std():.2f}")
    print(f"    Best  : {arr.max():+.2f}")
    print(f"    Worst : {arr.min():+.2f}")
    return float(arr.mean())


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Build vectorised + normalised training environment
    #    norm_obs=True    — standardise observations to zero-mean unit-variance
    #                       so all 6 state variables contribute equal gradient
    #    norm_reward=False — keep reward in original scale for interpretability
    #                        and fair comparison against baselines
    #    clip_obs=10.0    — clip normalised obs to [-10, 10] to handle outliers
    train_env = DummyVecEnv([make_env(df)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 3. Validate raw env (Gymnasium API check on the unwrapped version)
    print("\n[env]   Running Gymnasium compliance check ...")
    check_env(EconomicEnv(df), warn=True)
    print("[env]   Environment OK\n")

    # 4. Initialise PPO agent
    model = PPO(env=train_env, tensorboard_log=LOG_DIR, **PPO_CONFIG)

    # 5. Train
    print(f"[train] Starting training for {TOTAL_TIMESTEPS:,} timesteps ...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 6. Save model + normalisation statistics
    #    Both files are required to reload the agent correctly.
    save_path     = os.path.join(MODEL_DIR, "ppo_economic_policy")
    vec_norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    model.save(save_path)
    train_env.save(vec_norm_path)
    print(f"\n[saved] Model        → {save_path}.zip")
    print(f"[saved] VecNormalize → {vec_norm_path}")

    # 7. Evaluate — 20 episodes per policy
    N_EVAL = 20
    print(f"\n[eval]  Evaluating {N_EVAL} episodes per policy ...")

    ppo_rewards    = evaluate_ppo(model, vec_norm_path, df, n_episodes=N_EVAL)
    random_rewards = evaluate_episodes(random_policy,      df, n_episodes=N_EVAL, seed_offset=100)
    taylor_rewards = evaluate_episodes(taylor_rule_policy, df, n_episodes=N_EVAL, seed_offset=200)

    print(f"\n{'='*55}")
    print("  EVALUATION RESULTS  (mean ± std over 20 episodes)")
    print(f"{'='*55}")
    ppo_mean    = print_eval_summary("PPO (trained)",  ppo_rewards)
    rand_mean   = print_eval_summary("Random policy",  random_rewards)
    taylor_mean = print_eval_summary("Taylor Rule",    taylor_rewards)

    diff_rand   = ppo_mean - rand_mean
    diff_taylor = ppo_mean - taylor_mean

    print(f"\n{'='*55}")
    print("  COMPARISON")
    print(f"{'='*55}")
    print(f"  PPO vs Random     : {diff_rand:+.2f}  ({'BETTER' if diff_rand > 0 else 'WORSE'} than random)")
    print(f"  PPO vs Taylor Rule: {diff_taylor:+.2f}  ({'BETTER' if diff_taylor > 0 else 'WORSE'} than Taylor Rule)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()


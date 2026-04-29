"""
train_multi_country.py — Train a PPO agent on the multi-country GNN environment.

The agent observes BOTH the US and Canadian economies simultaneously and
outputs two actions (one interest rate change per country). This implements
the Graph Neural Network concept: countries are nodes, trade links are edges.

Usage:
    python train_multi_country.py

Expects:
    data/us_macro_data_real.csv    — US monthly data (2017-01 to 2024-12)
    data/ca_macro_data_real.csv    — Canada monthly data (same period, same format)

Output:
    models/ppo_multi_country.zip
"""

import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment.multi_country_env import MultiCountryEconomicEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

US_DATA_PATH = "data/us_macro_data_real.csv"
CA_DATA_PATH = "data/ca_macro_data_real.csv"
MODEL_DIR    = "models"
LOG_DIR      = "logs"

TOTAL_TIMESTEPS = 500_000

PPO_CONFIG = {
    "policy":        "MlpPolicy",
    "verbose":       1,
    "learning_rate": 3e-4,
    "n_steps":       512,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLUMN_RENAMES = {
    "GDP Growth":         "gdp_growth",
    "GDP growth":         "gdp_growth",    # Canada uses lowercase
    "Inflation":          "inflation",
    "Unemployment Rate ": "unemployment",
    "Unemployment Rate":  "unemployment",
    "Interest rate":      "interest_rate",
    "Interest Rate":      "interest_rate", # Canada uses title case
}

REQUIRED_COLS = ["inflation", "unemployment", "gdp_growth", "interest_rate"]


def load_country_data(path: str, country: str) -> pd.DataFrame:
    """Load and validate a country's macro CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{country}] Data file not found: {path}\n"
            f"Please place the {country} CSV at that path and re-run."
        )

    df = pd.read_csv(path).rename(columns=COLUMN_RENAMES)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[{country}] Missing columns: {missing}")

    df = df[REQUIRED_COLS].dropna().reset_index(drop=True)
    print(f"[{country}]  Loaded {len(df)} timesteps from '{path}'")
    print(df.describe().round(2))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Load data
    us_df = load_country_data(US_DATA_PATH, "US")
    ca_df = load_country_data(CA_DATA_PATH, "CA")

    # Align lengths — use the shorter of the two
    min_len = min(len(us_df), len(ca_df))
    us_df   = us_df.iloc[:min_len].reset_index(drop=True)
    ca_df   = ca_df.iloc[:min_len].reset_index(drop=True)
    print(f"\n[data]  Using {min_len} aligned monthly timesteps for both countries.\n")

    # 2. Build environment
    env = MultiCountryEconomicEnv(us_df, ca_df)

    # 3. Validate
    print("[env]   Running Gymnasium compliance check ...")
    check_env(env, warn=True)
    print("[env]   Environment OK\n")

    # 4. Train
    model = PPO(env=env, tensorboard_log=LOG_DIR, **PPO_CONFIG)
    print(f"[train] Starting multi-country training for {TOTAL_TIMESTEPS:,} timesteps ...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. Save
    save_path = os.path.join(MODEL_DIR, "ppo_multi_country")
    model.save(save_path)
    print(f"\n[done]  Model saved to '{save_path}.zip'")

    # 6. Evaluation rollout
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    print(f"\n{'='*70}")
    print("  EVALUATION — Multi-Country PPO (US + Canada)")
    print(f"{'='*70}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        print(f"\nStep {step:>3}  (Month {step})")
        print(f"  US  — inflation={info['us_inflation']:+.2f}%  "
              f"unemployment={info['us_unemployment']:+.2f}%  "
              f"gdp={info['us_gdp_growth']:+.3f}%  "
              f"rate={info['us_interest_rate']:+.2f}%  "
              f"(change={info['us_rate_change']:+.3f})  "
              f"reward={info['us_reward']:+.3f}")
        print(f"  CA  — inflation={info['ca_inflation']:+.2f}%  "
              f"unemployment={info['ca_unemployment']:+.2f}%  "
              f"gdp={info['ca_gdp_growth']:+.3f}%  "
              f"rate={info['ca_interest_rate']:+.2f}%  "
              f"(change={info['ca_rate_change']:+.3f})  "
              f"reward={info['ca_reward']:+.3f}")

    print(f"\n{'='*70}")
    print(f"  Total combined reward : {total_reward:+.4f}")
    print(f"  Average reward/step   : {total_reward / step:+.4f}")
    print(f"  Steps (months)        : {step}")
    print(f"{'='*70}")

    # 7. Random policy comparison
    obs, _ = env.reset()
    random_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        random_reward += reward
        done = terminated or truncated

    print(f"\n{'='*70}")
    print("  MODEL vs RANDOM COMPARISON")
    print(f"  PPO reward   : {total_reward:+.4f}")
    print(f"  Random reward: {random_reward:+.4f}")
    diff = total_reward - random_reward
    verdict = "BETTER" if diff > 0 else "WORSE"
    print(f"  Difference   : {diff:+.4f}  ({verdict} than random)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

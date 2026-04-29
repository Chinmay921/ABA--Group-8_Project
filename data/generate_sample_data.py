"""
generate_sample_data.py — Creates a synthetic US-like macroeconomic dataset.

This is ONLY for testing the pipeline before real FRED/World Bank data is
available. The values are statistically plausible but not historically exact.

Run:
    python data/generate_sample_data.py

Output:
    data/us_macro_data.csv  (quarterly, 2000 Q1 – 2023 Q4, 96 rows)
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Time axis: quarterly 2000-Q1 → 2023-Q4
# ---------------------------------------------------------------------------
dates = pd.date_range(start="2000-01-01", end="2023-12-31", freq="QS")
n = len(dates)

# ---------------------------------------------------------------------------
# Base series (mean-reverting random walks, loosely calibrated to US history)
# ---------------------------------------------------------------------------

def mean_reverting_series(n, mean, std, theta=0.15, seed_val=None):
    """Ornstein-Uhlenbeck process — stays near `mean` over time."""
    x = np.zeros(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mean - x[t - 1]) + rng.normal(0, std)
    return x


inflation = mean_reverting_series(n, mean=2.5, std=0.6)
unemployment = mean_reverting_series(n, mean=5.5, std=0.5)
gdp_growth = mean_reverting_series(n, mean=2.2, std=0.9)

# ---------------------------------------------------------------------------
# Inject the 2008 financial crisis (approx Q3 2008 – Q2 2009)
# ---------------------------------------------------------------------------
crisis_08_idx = pd.Index(dates).get_indexer(
    pd.date_range("2008-07-01", "2009-06-30", freq="QS"), method="nearest"
)
for i in crisis_08_idx:
    gdp_growth[i] -= rng.uniform(3.0, 5.0)
    unemployment[i] += rng.uniform(1.5, 3.0)
    inflation[i] -= rng.uniform(0.5, 1.5)

# ---------------------------------------------------------------------------
# Inject the COVID shock (approx Q2 2020 – Q3 2020) and recovery
# ---------------------------------------------------------------------------
covid_shock_idx = pd.Index(dates).get_indexer(
    pd.date_range("2020-04-01", "2020-09-30", freq="QS"), method="nearest"
)
for i in covid_shock_idx:
    gdp_growth[i] -= rng.uniform(5.0, 9.0)
    unemployment[i] += rng.uniform(5.0, 9.0)

covid_recovery_idx = pd.Index(dates).get_indexer(
    pd.date_range("2020-10-01", "2021-12-31", freq="QS"), method="nearest"
)
for i in covid_recovery_idx:
    gdp_growth[i] += rng.uniform(2.0, 5.0)
    unemployment[i] -= rng.uniform(1.0, 3.0)
    inflation[i] += rng.uniform(0.5, 2.0)   # post-COVID inflation surge

# ---------------------------------------------------------------------------
# Clip to realistic bounds
# ---------------------------------------------------------------------------
inflation = np.clip(inflation, -2.0, 12.0)
unemployment = np.clip(unemployment, 2.0, 15.0)
gdp_growth = np.clip(gdp_growth, -12.0, 10.0)

# ---------------------------------------------------------------------------
# Build and save dataframe
# ---------------------------------------------------------------------------
df = pd.DataFrame(
    {
        "date": dates,
        "inflation": np.round(inflation, 2),
        "unemployment": np.round(unemployment, 2),
        "gdp_growth": np.round(gdp_growth, 2),
    }
)

out_path = os.path.join(os.path.dirname(__file__), "us_macro_data.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to '{out_path}'")
print(df.describe().round(2))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MultiCountryEconomicEnv(gym.Env):
    """
    Multi-country RL environment implementing the Graph Neural Network concept.

    Two countries (US + Canada) are modelled as nodes in a graph.
    Each step, a single agent observes BOTH countries' states (message passing)
    and outputs TWO actions — one interest rate change per country.

    This mirrors the GNN idea: each "node" (country) receives information from
    its neighbour before the policy decision is made.

    Observation (8D):
        [us_inflation, us_unemployment, us_gdp_growth, us_interest_rate,
         ca_inflation, ca_unemployment, ca_gdp_growth, ca_interest_rate]

    Action (2D):
        [us_rate_change, ca_rate_change]  — each clipped to [-1, +1]

    Transition model (per country):
        next_state = baseline_data[t+1] + policy_effect + cross_country_spillover + noise

        Cross-country spillover (GNN edge):
            A rate hike in one country slightly dampens inflation in the neighbour
            (trade and capital flow channel).
    """

    metadata = {"render_modes": []}

    # Monetary policy targets
    INFLATION_TARGET    = 2.0
    UNEMPLOYMENT_TARGET = 4.5
    GDP_TARGET          = 0.25
    RATE_MIN            = -0.5
    RATE_MAX            = 20.0

    # Transition coefficients
    ALPHA     = 0.5   # own rate hike → own inflation reduction
    BETA      = 0.3   # own rate hike → own GDP reduction
    GAMMA     = 0.2   # weaker GDP → higher unemployment (Okun)
    SPILLOVER = 0.1   # neighbour rate hike → own inflation reduction (cross-country)
    NOISE_STD = 0.1

    def __init__(self, us_data: pd.DataFrame, ca_data: pd.DataFrame):
        super().__init__()

        assert len(us_data) == len(ca_data), \
            "US and Canada datasets must have the same number of rows (same time period)."

        self.us_data = us_data.reset_index(drop=True)
        self.ca_data = ca_data.reset_index(drop=True)
        self.n_steps = len(self.us_data)
        self.current_step = 0

        # ------------------------------------------------------------------
        # Observation space: 8 variables (4 per country)
        # Order: [us_inf, us_unemp, us_gdp, us_rate, ca_inf, ca_unemp, ca_gdp, ca_rate]
        # ------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array([-10., 0., -20., -2.,
                          -10., 0., -20., -2.], dtype=np.float32),
            high=np.array([30., 30., 20., 25.,
                           30., 30., 20., 25.], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space: 2 values — one rate change per country
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1., -1.], dtype=np.float32),
            high=np.array([1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        us = self._us_state
        ca = self._ca_state
        return np.array([
            us["inflation"], us["unemployment"], us["gdp_growth"], us["interest_rate"],
            ca["inflation"], ca["unemployment"], ca["gdp_growth"], ca["interest_rate"],
        ], dtype=np.float32)

    def _compute_reward(self, state: dict, current_rate: float, previous_rate: float) -> float:
        """Quadratic loss per country — same formula as the single-country env."""
        return float(-(
            1.0 * (state["inflation"]   - self.INFLATION_TARGET)    ** 2
            + 0.5 * (state["unemployment"] - self.UNEMPLOYMENT_TARGET) ** 2
            + 0.5 * (state["gdp_growth"]   - self.GDP_TARGET)          ** 2
            + 0.1 * (current_rate          - previous_rate)             ** 2
        ))

    def _step_country(self, data: pd.DataFrame, current_state: dict,
                      own_action: float, neighbour_action: float) -> dict:
        """
        Advance one country by one step.
        own_action      : rate change this country's agent applied
        neighbour_action: rate change the other country applied (spillover channel)
        """
        next_row = data.iloc[self.current_step]   # current_step already incremented

        baseline_inflation    = float(next_row["inflation"])
        baseline_gdp          = float(next_row["gdp_growth"])
        baseline_unemployment = float(next_row["unemployment"])

        # Own policy effect
        inflation_effect    = -self.ALPHA * own_action
        gdp_effect          = -self.BETA  * own_action
        gdp_next            = baseline_gdp + gdp_effect
        unemployment_effect =  self.GAMMA * (baseline_gdp - gdp_next)

        # Cross-country spillover (GNN edge weight)
        spillover_inflation = -self.SPILLOVER * neighbour_action

        noise = self.np_random.normal(0.0, self.NOISE_STD, size=3)

        inflation_next    = baseline_inflation    + inflation_effect + spillover_inflation + noise[0]
        gdp_next_final    = gdp_next                                                       + noise[1]
        unemployment_next = baseline_unemployment + unemployment_effect                    + noise[2]

        previous_rate = current_state["interest_rate"]
        current_rate  = float(np.clip(previous_rate + own_action, self.RATE_MIN, self.RATE_MAX))

        new_state = {
            "inflation":    inflation_next,
            "unemployment": unemployment_next,
            "gdp_growth":   gdp_next_final,
            "interest_rate": current_rate,
        }

        reward = self._compute_reward(new_state, current_rate, previous_rate)

        baseline_info = {
            "baseline_inflation":    baseline_inflation,
            "baseline_gdp":          baseline_gdp,
            "baseline_unemployment": baseline_unemployment,
        }

        return new_state, reward, baseline_info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        us_row = self.us_data.iloc[0]
        ca_row = self.ca_data.iloc[0]

        self._us_state = {
            "inflation":    float(us_row["inflation"]),
            "unemployment": float(us_row["unemployment"]),
            "gdp_growth":   float(us_row["gdp_growth"]),
            "interest_rate": float(us_row["interest_rate"]),
        }
        self._ca_state = {
            "inflation":    float(ca_row["inflation"]),
            "unemployment": float(ca_row["unemployment"]),
            "gdp_growth":   float(ca_row["gdp_growth"]),
            "interest_rate": float(ca_row["interest_rate"]),
        }

        return self._get_obs(), {}

    def step(self, action):
        us_action = float(action[0])
        ca_action = float(action[1])

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        # Each country steps forward, seeing the neighbour's action (GNN message pass)
        new_us_state, us_reward, us_baseline = self._step_country(
            self.us_data, self._us_state, us_action, ca_action
        )
        new_ca_state, ca_reward, ca_baseline = self._step_country(
            self.ca_data, self._ca_state, ca_action, us_action
        )

        self._us_state = new_us_state
        self._ca_state = new_ca_state

        obs    = self._get_obs()
        reward = us_reward + ca_reward   # combined reward across both countries

        info = {
            "step": self.current_step,
            # US
            "us_inflation":    new_us_state["inflation"],
            "us_unemployment": new_us_state["unemployment"],
            "us_gdp_growth":   new_us_state["gdp_growth"],
            "us_interest_rate": new_us_state["interest_rate"],
            "us_rate_change":  us_action,
            "us_reward":       us_reward,
            **{f"us_{k}": v for k, v in us_baseline.items()},
            # Canada
            "ca_inflation":    new_ca_state["inflation"],
            "ca_unemployment": new_ca_state["unemployment"],
            "ca_gdp_growth":   new_ca_state["gdp_growth"],
            "ca_interest_rate": new_ca_state["interest_rate"],
            "ca_rate_change":  ca_action,
            "ca_reward":       ca_reward,
            **{f"ca_{k}": v for k, v in ca_baseline.items()},
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

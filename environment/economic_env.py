import collections

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class EconomicEnv(gym.Env):
    """
    Custom Gymnasium environment for monetary policy decisions.

    State  : [inflation, unemployment, gdp_growth, interest_rate]
    Action : single continuous value representing the change in interest rate,
             clipped to [-1, +1] (percentage points per step)
    Reward : quadratic loss penalising deviations from macro targets
             -(1.0*(π-2)² + 0.5*(u-4.5)² + 0.5*(g-0.25)² + 0.1*(Δr)²)

    Transition model (fully self-contained AR1 dynamics):
    -------------------------------------------------------
    Historical data seeds the starting state only. From there the simulation
    runs on its own dynamics:

        g_{t+1}  = ρ_g × g_t   + (1-ρ_g) × G_SS   - f_g(actions)  + ε_g
        π_{t+1}  = ρ_π × π_t   + (1-ρ_π) × π_SS   - f_π(actions)  + ε_π
        u_{t+1}  = ρ_u × u_t   + (1-ρ_u) × U_SS   - OKUN×(g-G_SS) + ε_u
        r_{t+1}  = clip(r_t + action, RATE_MIN, RATE_MAX)

    where f(actions) is a 3-lag weighted + nonlinear (signed quadratic) policy
    transmission and ε are correlated supply shocks with rare recession events.

    Key design decisions
    --------------------
    Persistence  : AR1 coefficients calibrated to empirical US monthly data
                   (ρ_π=0.75, ρ_u=0.92, ρ_g=0.40)
    Lag effects  : inflation transmission peaks at lag 1–2 (slow);
                   GDP transmission front-loaded at lag 0 (fast)
    Nonlinearity : signed quadratic captures financial accelerator effects
    Shocks       : correlated supply shocks (ε_π, ε_g covariance=-0.5σσ);
                   1.5% monthly recession probability
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Macro targets
    # ------------------------------------------------------------------
    INFLATION_TARGET    = 2.0
    UNEMPLOYMENT_TARGET = 4.5
    GDP_TARGET          = 0.25
    RATE_MIN            = -0.5
    RATE_MAX            = 20.0

    # ------------------------------------------------------------------
    # Persistence (AR1 coefficients)
    # Inflation  0.75 : moderate stickiness — price-setting via wage contracts
    # Unemploy   0.92 : very sticky — Beveridge curve, slow hiring/firing
    # GDP        0.40 : lower persistence — more driven by demand shocks
    # ------------------------------------------------------------------
    RHO_PI = 0.75
    RHO_U  = 0.92
    RHO_G  = 0.40

    # Long-run steady states (attractors for mean reversion)
    PI_SS = 2.5    # slightly above target: typical central-bank inflation bias
    U_SS  = 4.5    # NAIRU
    G_SS  = 0.25   # (~3% annualised)

    # ------------------------------------------------------------------
    # Lagged policy transmission weights
    # Index: [lag0=current period, lag1=1 month ago, lag2=2 months ago]
    #
    # Inflation (SLOW): minimal same-period effect, peaks at lag 1.
    #   Reflects sticky prices (menu costs, long-term contracts).
    # GDP (FAST): largest effect same period, decays quickly.
    #   Reflects immediate credit/investment channel (housing, capex).
    # ------------------------------------------------------------------
    W_PI = np.array([0.10, 0.20, 0.15])   # total: 0.45 pp per unit action
    W_G  = np.array([0.25, 0.10, 0.05])   # total: 0.40 pp per unit action

    # Okun's law coefficient (monthly scale).
    # 1% below-trend GDP → +0.15% rise in unemployment per month.
    OKUN = 0.15

    # ------------------------------------------------------------------
    # IS curve: neutral (natural) real interest rate
    # When r_t > NEUTRAL_RATE, GDP is dragged down every period rates
    # stay elevated — not just when they move.  Laubach-Williams (2024)
    # estimate for the US is ~2.5%.  Coefficient IS_COEFF=0.25 means a
    # 1pp rate-above-neutral gap reduces monthly GDP by 0.25pp.
    # ------------------------------------------------------------------
    NEUTRAL_RATE = 2.5
    IS_COEFF     = 0.25

    # ------------------------------------------------------------------
    # Phillips curve: output gap feeds inflation
    # KAPPA=0.12: 1pp above-trend GDP growth adds 0.12pp to next-month
    # inflation.  Conservative end of empirical range (0.10–0.30).
    # ------------------------------------------------------------------
    KAPPA = 0.12

    # ------------------------------------------------------------------
    # Nonlinear policy effects (signed quadratic add-on)
    # total_effect = linear_term + NL × action × |action|
    # Captures: financial accelerator (large hikes cause disproportionate
    # credit crunch) and convex Phillips curve disinflationary effects.
    # NL=0.08: at full hike (+1%), adds 0.08 pp extra on top of linear.
    # ------------------------------------------------------------------
    NL_PI = 0.08
    NL_G  = 0.08

    # ------------------------------------------------------------------
    # Shock structure
    # Supply shocks: negative correlation between inflation and GDP shocks.
    #   SUPPLY_CORR=-0.50 → a cost-push shock raises π and lowers g.
    # Monthly std calibrated as residual variation after AR1 persistence.
    # Recession shock: 1.5% monthly probability (~1 per 5.5 years).
    #   Magnitude: historically plausible moderate recession, not COVID.
    # ------------------------------------------------------------------
    SHOCK_STD_PI = 0.12
    SHOCK_STD_G  = 0.22
    SHOCK_STD_U  = 0.07
    SUPPLY_CORR  = -0.50
    P_RECESSION  = 0.015

    # ------------------------------------------------------------------
    # Hard state bounds applied after every step (RL stability)
    # ------------------------------------------------------------------
    PI_MIN, PI_MAX =  -5.0, 25.0
    U_MIN,  U_MAX  =   0.0, 20.0
    G_MIN,  G_MAX  =  -5.0,  8.0

    def __init__(self, data: pd.DataFrame):
        super().__init__()

        self.data    = data.reset_index(drop=True)
        self.n_steps = len(self.data)

        # ------------------------------------------------------------------
        # Observation space: [inflation, unemployment, gdp_growth, interest_rate]
        # Bounds are generous to cover extreme historical episodes.
        # ------------------------------------------------------------------
        # 6-dimensional: [inflation, unemployment, gdp_growth, interest_rate,
        #                   action_{t-1}, action_{t-2}]
        # The last two slots expose the two most-recent rate changes so the
        # agent can observe the delayed policy cycle it already initiated.
        self.observation_space = spaces.Box(
            low=np.array( [-10.0,  0.0, -20.0, -2.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 30.0, 30.0,  20.0, 25.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space: change in interest rate, continuous in [-1, +1]
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array( [-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Mutable state (initialised in reset)
        self._sim_state      = {}
        self.current_step    = 0
        self._action_history = collections.deque([0.0, 0.0, 0.0], maxlen=3)

        # Precompute shock covariance matrix (avoids repeated allocation in step)
        self._shock_cov = np.array([
            [self.SHOCK_STD_PI ** 2,
             self.SUPPLY_CORR * self.SHOCK_STD_PI * self.SHOCK_STD_G],
            [self.SUPPLY_CORR * self.SHOCK_STD_PI * self.SHOCK_STD_G,
             self.SHOCK_STD_G ** 2],
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        s = self._sim_state
        # _action_history[0] = most recent action (a_t after step, 0.0 at reset)
        # _action_history[1] = one step prior (a_{t-1})
        return np.array(
            [
                s["inflation"],
                s["unemployment"],
                s["gdp_growth"],
                s["interest_rate"],
                self._action_history[0],
                self._action_history[1],
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, inflation: float, unemployment: float,
                        gdp_growth: float,
                        current_rate: float, previous_rate: float) -> float:
        """
        Quadratic loss reward — penalises deviations from macro targets.

          inflation term  : (π - 2.0)²       — primary central bank mandate
          unemployment    : (u - 4.5)²       — deviation from natural rate
          gdp growth      : (g - 0.25)²      — stable growth target
          rate smoothing  : (r_t - r_{t-1})² — penalise abrupt rate changes
        """
        return float(
            -(
                1.0 * (inflation    - self.INFLATION_TARGET)    ** 2
              + 0.5 * (unemployment - self.UNEMPLOYMENT_TARGET) ** 2
              + 0.5 * (gdp_growth   - self.GDP_TARGET)          ** 2
              + 0.1 * (current_rate  - previous_rate)            ** 2
            )
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Seed from a random row in the first 80% of the dataset.
        # Gives PPO diverse starting conditions across episodes instead of
        # always starting from Jan 2017.
        max_start = max(1, int(0.8 * self.n_steps))
        start_idx = int(self.np_random.integers(0, max_start))
        row = self.data.iloc[start_idx]

        self._sim_state = {
            "inflation":     float(row["inflation"]),
            "unemployment":  float(row["unemployment"]),
            "gdp_growth":    float(row["gdp_growth"]),
            "interest_rate": float(row["interest_rate"]),
        }

        # Assume no prior policy moves at the start of each episode.
        self._action_history = collections.deque([0.0, 0.0, 0.0], maxlen=3)

        return self._get_obs(), {}

    def step(self, action):
        rate_change   = float(action[0])
        previous_rate = self._sim_state["interest_rate"]
        current_rate  = float(np.clip(previous_rate + rate_change,
                                      self.RATE_MIN, self.RATE_MAX))

        # --- 1. Update lag buffer (most recent action at index 0) -------
        self._action_history.appendleft(rate_change)
        hist = np.array(self._action_history)  # [a_t, a_{t-1}, a_{t-2}]

        # --- 2. Lagged + nonlinear policy effects -----------------------
        # pi_policy: slow — weighted sum peaks at lag 1 + signed quadratic
        # g_policy : fast — front-loaded at current period + signed quadratic
        pi_policy = float(self.W_PI @ hist) + self.NL_PI * rate_change * abs(rate_change)
        g_policy  = float(self.W_G  @ hist) + self.NL_G  * rate_change * abs(rate_change)

        # --- 3. Correlated supply shocks --------------------------------
        eps_pi, eps_g = self.np_random.multivariate_normal([0.0, 0.0], self._shock_cov)
        eps_u         = self.np_random.normal(0.0, self.SHOCK_STD_U)

        # Rare recession shock: GDP contraction + unemployment spike
        if self.np_random.random() < self.P_RECESSION:
            eps_g -= float(self.np_random.uniform(1.0, 2.5))
            eps_u += float(self.np_random.uniform(0.5, 1.5))

        # --- 4. AR1 transitions with policy and shocks ------------------
        # IS curve level drag: continuous penalty while rates above neutral.
        # A 5% rate held steady for 6 months keeps dragging GDP every month.
        is_drag = self.IS_COEFF * (current_rate - self.NEUTRAL_RATE)

        # GDP computed first — feeds into Okun's law for unemployment.
        gdp_next = (
            self.RHO_G * self._sim_state["gdp_growth"]
            + (1.0 - self.RHO_G) * self.G_SS
            - g_policy
            - is_drag
            + eps_g
        )

        # Unemployment: Okun's law — below-trend GDP raises unemployment.
        unemployment_next = (
            self.RHO_U  * self._sim_state["unemployment"]
            + (1.0 - self.RHO_U) * self.U_SS
            - self.OKUN * (gdp_next - self.G_SS)
            + eps_u
        )

        # Inflation: slow transmission peaks at lag 1 + Phillips output gap.
        # Hot growth (gdp_t > 0.25%) pushes inflation up next period.
        inflation_next = (
            self.RHO_PI * self._sim_state["inflation"]
            + (1.0 - self.RHO_PI) * self.PI_SS
            + self.KAPPA * (self._sim_state["gdp_growth"] - self.G_SS)
            - pi_policy
            + eps_pi
        )

        # --- 5. Clip to hard bounds (RL stability) ----------------------
        inflation_next    = float(np.clip(inflation_next,    self.PI_MIN, self.PI_MAX))
        unemployment_next = float(np.clip(unemployment_next, self.U_MIN,  self.U_MAX))
        gdp_next          = float(np.clip(gdp_next,          self.G_MIN,  self.G_MAX))

        # --- 6. Update state --------------------------------------------
        self._sim_state = {
            "inflation":     inflation_next,
            "unemployment":  unemployment_next,
            "gdp_growth":    gdp_next,
            "interest_rate": current_rate,
        }

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        reward = self._compute_reward(
            inflation_next, unemployment_next, gdp_next,
            current_rate, previous_rate,
        )

        info = {
            "step":                self.current_step,
            "inflation":           inflation_next,
            "unemployment":        unemployment_next,
            "gdp_growth":          gdp_next,
            "interest_rate":       current_rate,
            "interest_rate_change": rate_change,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # No rendering needed for the baseline version
        pass

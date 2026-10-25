"""1-day VaR / ES models with a common interface.

A model implements  forecast(history, weights, alphas) -> {alpha: (VaR, ES)}  where
  history : DataFrame of asset simple returns, ending the day BEFORE the forecast day
  weights : Series of portfolio weights indexed by asset name (daily rebalanced)
  alphas  : tail probabilities, e.g. (0.01, 0.05) for 99% and 95%
VaR and ES are positive losses (0.02 = a 2% loss). ES is the expected loss given that
the loss is larger than the VaR.

The backtest calls forecast() once per day, in date order. Models may keep state between
calls (GarchVaR does, to refit only every N days), so use a fresh instance per backtest run.

A VAE/GAN model plugs in the same way: train on `history`, generate asset return scenarios,
compute portfolio scenarios with `portfolio_returns(scenarios, weights)` and return
{a: var_es_from_scenarios(portfolio_scenarios, a) for a in alphas}.
"""
import numpy as np
from arch import arch_model
from scipy.stats import norm
from scipy.stats import t as student_t

from src.portfolio import portfolio_returns


def var_es_from_scenarios(returns, alpha):
    """Empirical VaR and ES from a sample of portfolio returns."""
    returns = np.asarray(returns)
    q = np.quantile(returns, alpha)
    return -q, -returns[returns <= q].mean()


def var_es_normal(mu, sigma, alpha):
    """VaR and ES when the return is Normal(mu, sigma^2)."""
    z = norm.ppf(alpha)
    return -(mu + sigma * z), -mu + sigma * norm.pdf(z) / alpha


def var_es_student_t(mu, sigma, nu, alpha):
    """VaR and ES when the return is mu + sigma * T, with T a Student-t(nu) scaled to
    variance 1 (the convention used by the arch package), so sigma is the std dev."""
    scale = sigma * np.sqrt((nu - 2) / nu)
    q = student_t.ppf(alpha, nu)
    es_std = student_t.pdf(q, nu) / alpha * (nu + q**2) / (nu - 1)  # -E[T | T <= q], standard t
    return -(mu + scale * q), -mu + scale * es_std


class VaRModel:
    name = "base"

    def forecast(self, history, weights, alphas):
        raise NotImplementedError


class HistoricalSimulation(VaRModel):
    """VaR/ES from the empirical distribution of the last `window` portfolio returns."""

    def __init__(self, window=500):
        self.window = window
        self.name = "Historical sim."

    def forecast(self, history, weights, alphas):
        r = portfolio_returns(history, weights).iloc[-self.window:]
        return {a: var_es_from_scenarios(r, a) for a in alphas}


class NormalVaR(VaRModel):
    """Normal distribution with rolling mean and std of the last `window` portfolio returns."""

    def __init__(self, window=500):
        self.window = window
        self.name = "Normal"

    def forecast(self, history, weights, alphas):
        r = portfolio_returns(history, weights).iloc[-self.window:]
        return {a: var_es_normal(r.mean(), r.std(), a) for a in alphas}


class GarchVaR(VaRModel):
    """GARCH(1,1) with constant mean, fitted to the last `window` portfolio returns.

    dist="normal" or "t" (Student-t innovations). Parameters are re-estimated every
    `refit_every` calls; on the days in between they are kept fixed, but the volatility
    forecast is still updated with the newest returns.
    """

    def __init__(self, dist="normal", window=1000, refit_every=20):
        self.dist = dist
        self.window = window
        self.refit_every = refit_every
        self.name = f"GARCH-{dist}"
        self._calls = 0
        self._params = None

    def forecast(self, history, weights, alphas):
        r = 100 * portfolio_returns(history, weights).iloc[-self.window:]  # percent: better numerics
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=self.dist)
        if self._params is None or self._calls % self.refit_every == 0:
            res = am.fit(disp="off")
            self._params = res.params
        else:
            res = am.fix(self._params)
        self._calls += 1

        f = res.forecast(horizon=1)
        mu, sigma = f.mean.iloc[-1, 0] / 100, np.sqrt(f.variance.iloc[-1, 0]) / 100
        if self.dist == "t":
            return {a: var_es_student_t(mu, sigma, self._params["nu"], a) for a in alphas}
        return {a: var_es_normal(mu, sigma, a) for a in alphas}

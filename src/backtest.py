"""Backtest 1-day VaR/ES models: violations, Kupiec and Christoffersen tests, table, plot."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from src.portfolio import portfolio_returns


def level(alpha):
    """0.01 -> '99' (confidence level as text, used in column and file names)."""
    return f"{round(100 * (1 - alpha))}"


def run_backtest(model, asset_returns, weights, start, alphas=(0.01, 0.05)):
    """One forecast per day from `start` to the end, each using only data before that day.

    Returns a DataFrame indexed by date with the realised portfolio return `ret` and, per
    level (e.g. 99): `var_99`, `es_99` (positive losses) and `viol_99` (loss > VaR).
    """
    port = portfolio_returns(asset_returns, weights)
    first = asset_returns.index.searchsorted(pd.Timestamp(start))
    forecasts = [model.forecast(asset_returns.iloc[:pos], weights, alphas)  # rows before day pos
                 for pos in range(first, len(asset_returns))]

    out = pd.DataFrame({"ret": port.iloc[first:]})
    for a in alphas:
        L = level(a)
        out[f"var_{L}"] = [f[a][0] for f in forecasts]
        out[f"es_{L}"] = [f[a][1] for f in forecasts]
        out[f"viol_{L}"] = -out["ret"] > out[f"var_{L}"]
    return out


def kupiec_test(violations, alpha):
    """Kupiec proportion-of-failures test. H0: P(violation) = alpha. Returns (LR, p-value)."""
    n, x = len(violations), int(np.sum(violations))
    phat = x / n
    ll_h0 = xlogy(x, alpha) + xlogy(n - x, 1 - alpha)
    ll_h1 = xlogy(x, phat) + xlogy(n - x, 1 - phat)
    lr = -2 * (ll_h0 - ll_h1)
    return lr, chi2.sf(lr, 1)


def christoffersen_test(violations):
    """Christoffersen independence test. H0: a violation today does not change the chance
    of a violation tomorrow (no clustering). Returns (LR, p-value)."""
    v = np.asarray(violations, dtype=int)
    prev, curr = v[:-1], v[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0  # P(violation | none yesterday)
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0  # P(violation | violation yesterday)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_h1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll_h0 - ll_h1)
    return lr, chi2.sf(lr, 1)


def summary_table(results, alphas, test_level=0.05):
    """results: dict model name -> run_backtest output. One row per (level, model)."""
    rows = []
    for a in alphas:
        L = level(a)
        for name, df in results.items():
            v = df[f"viol_{L}"]
            n, x = len(v), int(v.sum())
            lr_uc, p_uc = kupiec_test(v, a)
            lr_ind, p_ind = christoffersen_test(v)
            rows.append({
                "Level": f"{L}%", "Model": name, "Days": n,
                "Expected": n * a, "Violations": x, "Rate": x / n,
                "Avg VaR": df[f"var_{L}"].mean(), "Avg ES": df[f"es_{L}"].mean(),
                "Avg loss when violated": -df.loc[v, "ret"].mean() if x else np.nan,
                "Kupiec LR": lr_uc, "Kupiec p": p_uc,
                "Kupiec": "reject" if p_uc < test_level else "pass",
                "Christoffersen LR": lr_ind, "Christoffersen p": p_ind,
                "Christoffersen": "reject" if p_ind < test_level else "pass",
            })
    return pd.DataFrame(rows).set_index(["Level", "Model"])


def plot_backtest(results, alpha, path, title=""):
    """One panel per model: realised returns, -VaR line and violation dots."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # palette order
    if len(results) > len(colors):
        raise ValueError(f"plot_backtest has {len(colors)} colours but got {len(results)} models")
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
    L = level(alpha)
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.6 * len(results)),
                             sharex=True, sharey=True, facecolor=surface)
    for ax, (name, df), color in zip(np.atleast_1d(axes), results.items(), colors):
        viol = df[df[f"viol_{L}"]]
        ax.set_facecolor(surface)
        ax.plot(df.index, df["ret"], color="#b9b8b2", lw=0.6)
        ax.plot(df.index, -df[f"var_{L}"], color=color, lw=1.4)
        ax.plot(viol.index, viol["ret"], "o", color=ink, ms=3.5, mec=surface, mew=0.8)
        ax.axhline(0, color=muted, lw=0.8)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, length=0)
        ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))  # no 2.5% ticks
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.set_title(f"{name}: {len(viol)} violations (expected {len(df) * alpha:.0f})",
                     loc="left", color=ink, fontsize=10)
    fig.suptitle(f"{title}\nGrey: daily portfolio return. Line: -VaR {L}%. Dots: violations (loss > VaR).",
                 x=0.01, ha="left", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

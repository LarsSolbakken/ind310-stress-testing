"""Check that VAE scenarios look like real returns, using the training period (2004-2018) only.

  python -m src.vae_realism --experiment E6_name   # results go to results/E6_name/ (must not exist)

(The results of experiment E3 are the files results/vae_realism*.)

1. Free-running simulation: N_PATHS synthetic histories as long as the training period, each new
   day generated from the model's own previous WINDOW days. Their stylized facts (fat tails, volatility
   clustering, stock-bond correlation) are compared with the real 2004-2018 returns.
2. Conditional check: scenario clouds on a calm and a stormy day, and the VAE's 99% VaR over
   2004-2018 against the real returns.
The pass criteria in check() were fixed before looking at any results.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data import load_data
from src.experiment import experiment_dir
from src.portfolio import make_weights, portfolio_returns
from src.var_models import var_es_from_scenarios
from src.vae import SEED, WINDOW, ScenarioGenerator, VAEVaR, make_deterministic

N_PATHS = 20
CALM_DAY, STORMY_DAY = "2017-06-01", "2008-10-10"
WEIGHTS = make_weights(SP500=0.5, BONDS=0.5)
VAE, REAL = "#e87ba4", "#52514e"  # VAE keeps its colour from the backtest plots; real data is neutral
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def stylized_facts(r):
    """r: DataFrame of daily returns with two assets (stocks, bonds) -> Series of statistics."""
    facts = {}
    for a in r.columns:
        x = r[a]
        facts[f"{a} std %"] = 100 * x.std()
        facts[f"{a} skew"] = x.skew()
        facts[f"{a} excess kurtosis"] = x.kurt()
        facts[f"{a} 1% quantile %"] = 100 * x.quantile(0.01)
        facts[f"{a} 99% quantile %"] = 100 * x.quantile(0.99)
        facts[f"{a} ACF r lag 1"] = x.autocorr(1)
        for lag in (1, 10, 50):
            facts[f"{a} ACF |r| lag {lag}"] = x.abs().autocorr(lag)
    stocks, bonds = r.columns[:2]
    roll = r[stocks].rolling(60).corr(r[bonds]).dropna()
    facts["stock-bond corr"] = r[stocks].corr(r[bonds])
    facts["rolling 60d corr median"] = roll.median()
    facts["rolling 60d corr spread"] = roll.quantile(0.95) - roll.quantile(0.05)
    return pd.Series(facts)


def check(table, assets, paths):
    """The pass criteria from the plan. Returns {criterion: passed}."""
    sim = table["Sim median"]
    inside = table["Real in band"]
    stocks = assets[0]
    return {
        "Fat tails: simulated excess kurtosis > 3 for both assets":
            all(sim[f"{a} excess kurtosis"] > 3 for a in assets),
        "Fat tails: real 1% and 99% quantiles inside the simulated 5-95% band":
            all(inside[f"{a} {q} quantile %"] for a in assets for q in ("1%", "99%")),
        f"Clustering: simulated ACF of |r| at lags 1 and 10 > 0.1 ({stocks})":
            sim[f"{stocks} ACF |r| lag 1"] > 0.1 and sim[f"{stocks} ACF |r| lag 10"] > 0.1,
        "Correlation: simulated stock-bond correlation < 0 and real value inside the band":
            sim["stock-bond corr"] < 0 and inside["stock-bond corr"],
        "Correlation varies: simulated rolling-correlation spread >= half the real spread":
            sim["rolling 60d corr spread"] >= 0.5 * table.loc["rolling 60d corr spread", "Real"],
        "Stable: no NaN or exploding paths (all |r| < 50%)":
            bool(np.isfinite(paths).all() and np.abs(paths).max() < 0.5),
    }


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=10)


def plot_stylized(train, paths, path):
    assets = list(train.columns)
    stocks, bonds = assets[:2]
    pooled = paths.reshape(-1, len(assets))
    probs, lags = np.linspace(0.0005, 0.9995, 500), np.arange(1, 101)
    fig, axes = plt.subplots(4, 2, figsize=(11, 15), facecolor=SURFACE)

    for j, a in enumerate(assets):
        ax = axes[0, j]
        style(ax, f"{a}: quantiles, simulated vs real (on the line = same distribution)")
        real_q, sim_q = 100 * np.quantile(train[a], probs), 100 * np.quantile(pooled[:, j], probs)
        lim = [min(real_q.min(), sim_q.min()), max(real_q.max(), sim_q.max())]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        ax.plot(real_q, sim_q, "o", color=VAE, ms=3, mec=SURFACE, mew=0.5)
        ax.set_xlabel("real quantile (%)", color=MUTED)
        ax.set_ylabel("simulated quantile (%)", color=MUTED)

        ax = axes[1, j]
        style(ax, f"{a}: autocorrelation of |returns| (volatility clustering)")
        sim_acf = np.array([[pd.Series(np.abs(p[:, j])).autocorr(k) for k in lags] for p in paths])
        ax.fill_between(lags, *np.quantile(sim_acf, [0.05, 0.95], axis=0), color=VAE, alpha=0.25, lw=0)
        ax.plot(lags, np.median(sim_acf, axis=0), color=VAE, lw=1.5, label="VAE simulated (median, 5-95% band)")
        ax.plot(lags, [train[a].abs().autocorr(k) for k in lags], color=REAL, lw=1.5, label="Real 2004-2018")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("lag (days)", color=MUTED)
        ax.legend(frameon=False)

    real_roll = train[stocks].rolling(60).corr(train[bonds]).dropna()
    sim_roll = [pd.Series(p[:, 0]).rolling(60).corr(pd.Series(p[:, 1])).dropna() for p in paths]
    ax = axes[2, 0]
    style(ax, "60-day rolling stock-bond correlation: distribution")
    bins = np.linspace(-1, 1, 41)
    ax.hist(np.concatenate(sim_roll), bins=bins, density=True, color=VAE, alpha=0.45, label="VAE simulated (all paths)")
    ax.hist(real_roll, bins=bins, density=True, histtype="step", color=REAL, lw=1.5, label="Real 2004-2018")
    ax.legend(frameon=False)
    ax = axes[2, 1]
    style(ax, "60-day rolling stock-bond correlation over time")
    ax.plot(np.arange(len(real_roll)), real_roll.to_numpy(), color=REAL, lw=1, label="Real 2004-2018")
    ax.plot(np.arange(len(sim_roll[0])), sim_roll[0].to_numpy(), color=VAE, lw=1, label="VAE simulated path 1")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("trading day", color=MUTED)
    ax.legend(frameon=False)

    ax = axes[3, 0]
    style(ax, f"{stocks} daily returns: real 2004-2018")
    ax.plot(train.index, train[stocks], color=REAL, lw=0.6)
    ax2 = axes[3, 1]
    style(ax2, f"{stocks} daily returns: VAE simulated path 1")
    ax2.plot(np.arange(len(paths[0])), paths[0][:, 0], color=VAE, lw=0.6)
    ax2.set_xlabel("trading day", color=MUTED)
    lim = max(np.abs(train[stocks]).max(), np.abs(paths[0][:, 0]).max()) * 1.05
    for a in (ax, ax2):
        a.set_ylim(-lim, lim)
        a.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

    fig.suptitle(f"VAE realism check: {N_PATHS} free-running simulated paths vs real returns (training period)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_conditional(train, gen, insample, path):
    stocks, bonds = list(train.columns)[:2]
    fig = plt.figure(figsize=(11, 9), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    clouds = []
    for label, day in [("Calm", CALM_DAY), ("Stormy", STORMY_DAY)]:
        pos = train.index.searchsorted(pd.Timestamp(day))
        history = train.iloc[:pos]
        scenarios = gen.sample(history, 10_000, seed=SEED)
        var99, _ = var_es_from_scenarios(portfolio_returns(scenarios, WEIGHTS), 0.01)
        vol = 100 * history[stocks].iloc[-20:].std()
        clouds.append((label, train.index[pos].date(), scenarios, var99, vol))
    lim = 1.05 * max(np.abs(np.quantile(s, [0.001, 0.999], axis=0)).max() for *_, s, _, _ in clouds)
    for j, (label, day, scenarios, var99, vol) in enumerate(clouds):
        ax = fig.add_subplot(grid[0, j])
        style(ax, f"{label}: 10,000 scenarios for {day}\n{stocks} vol last 20 days {vol:.1f}%/day, "
                  f"50/50 99% VaR {var99:.1%}")
        ax.plot(scenarios[stocks], scenarios[bonds], "o", color=VAE, ms=1.5, alpha=0.3, mew=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim / 3, lim / 3)
        ax.set_xlabel(f"{stocks} return", color=MUTED)
        ax.set_ylabel(f"{bonds} return", color=MUTED)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(lambda v, _: f"{v:.0%}")

    ax = fig.add_subplot(grid[1, :])
    viol = insample[insample["viol_99"]]
    style(ax, f"In-sample: VAE 99% VaR for the 50/50 portfolio, 2004-2018 (the model was trained on these days)\n"
              f"{len(viol)} violations, {len(insample) * 0.01:.0f} expected")
    ax.plot(insample.index, insample["ret"], color="#b9b8b2", lw=0.6)
    ax.plot(insample.index, -insample["var_99"], color=VAE, lw=1.2)
    ax.plot(viol.index, viol["ret"], "o", color=INK, ms=3, mec=SURFACE, mew=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="name of the new results folder, e.g. E6_realism")
    out = experiment_dir(parser.parse_args().experiment)  # stops if it already exists

    make_deterministic()
    train, _ = load_data()  # the test period is not used
    gen = ScenarioGenerator.load()

    paths = gen.simulate_paths(train.iloc[:WINDOW], n_steps=len(train), n_paths=N_PATHS, seed=SEED)
    sims = pd.DataFrame([stylized_facts(pd.DataFrame(p, columns=train.columns)) for p in paths])
    table = pd.DataFrame({"Real": stylized_facts(train), "Sim median": sims.median(),
                          "Sim 5%": sims.quantile(0.05), "Sim 95%": sims.quantile(0.95)})
    table["Real in band"] = table["Real"].between(table["Sim 5%"], table["Sim 95%"])
    print(table.round(3).to_string(), "\n")

    results = check(table, list(train.columns), paths)
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{sum(results.values())} of {len(results)} criteria passed")
    insample = run_backtest(VAEVaR(), train, WEIGHTS, train.index[WINDOW], alphas=(0.01,))

    out.mkdir(parents=True)  # only now, so a crashed run leaves nothing behind
    table.to_csv(out / "vae_realism.csv")
    plot_stylized(train, paths, out / "vae_realism_stylized.png")
    plot_conditional(train, gen, insample, out / "vae_realism_conditional.png")
    print(f"Saved in {out}")


if __name__ == "__main__":
    main()

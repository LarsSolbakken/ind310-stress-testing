"""Backtest the baseline VaR/ES models on the test period and save results in results/.

  python -m src.run_baselines                  # 50/50 portfolio
  python -m src.run_baselines --portfolio 20-80
  python -m src.run_baselines --portfolio all  # all portfolios in PORTFOLIOS
"""
import argparse

import pandas as pd

from src.backtest import level, plot_backtest, run_backtest, summary_table
from src.data import load_data
from src.portfolio import make_weights
from src.var_models import GarchVaR, HistoricalSimulation, NormalVaR

PORTFOLIOS = {
    "50-50": make_weights(SP500=0.5, BONDS=0.5),
    "20-80": make_weights(SP500=0.2, BONDS=0.8),
    "80-20": make_weights(SP500=0.8, BONDS=0.2),
}
ALPHAS = (0.01, 0.05)  # 99% and 95%


def baseline_models():
    """Fresh instances every run (GARCH keeps state between days)."""
    return [
        HistoricalSimulation(window=500),
        NormalVaR(window=500),
        GarchVaR(dist="normal", window=1000, refit_every=20),
        GarchVaR(dist="t", window=1000, refit_every=20),
    ]


def print_table(table):
    show = table[["Violations", "Expected", "Rate", "Avg VaR", "Avg ES", "Avg loss when violated",
                  "Kupiec p", "Kupiec", "Christoffersen p", "Christoffersen"]].copy()
    for col in ["Rate", "Avg VaR", "Avg ES", "Avg loss when violated"]:
        show[col] = show[col].map("{:.2%}".format)
    for col in ["Kupiec p", "Christoffersen p"]:
        show[col] = show[col].map("{:.3f}".format)
    show["Expected"] = show["Expected"].map("{:.1f}".format)
    print(show.to_string())


def run(name, weights, asset_returns, test_start):
    results = {}
    for model in baseline_models():
        print(f"  {model.name} ...", flush=True)
        results[model.name] = run_backtest(model, asset_returns, weights, test_start, ALPHAS)

    table = summary_table(results, ALPHAS)
    table.to_csv(f"results/var_backtest_{name}.csv")
    pd.concat(results, names=["model", "date"]).to_csv(f"results/var_forecasts_{name}.csv")
    weights_text = ", ".join(f"{w:.0%} {asset}" for asset, w in weights.items())
    for a in ALPHAS:
        plot_backtest(results, a, f"results/var_backtest_{name}_{level(a)}.png",
                      title=f"1-day VaR {level(a)}% backtest, {weights_text}")
    print_table(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", default="50-50", choices=[*PORTFOLIOS, "all"])
    args = parser.parse_args()

    train, test = load_data()
    asset_returns = pd.concat([train, test])
    names = list(PORTFOLIOS) if args.portfolio == "all" else [args.portfolio]
    for name in names:
        print(f"\nPortfolio {name} (test {test.index[0].date()} to {test.index[-1].date()}):")
        run(name, PORTFOLIOS[name], asset_returns, test.index[0])


if __name__ == "__main__":
    main()

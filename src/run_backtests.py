"""Backtest all VaR/ES models (baselines and the three VAE variants) on the test period.

  python -m src.run_backtests --experiment E6_name                  # 50/50 portfolio
  python -m src.run_backtests --experiment E6_name --portfolio all  # all portfolios in PORTFOLIOS

Results go to a new folder results/<experiment>/; the script stops if that folder already exists,
because results are never overwritten. Log every run in RESULTS.md.
The VAE variants need a trained model first: python -m src.train_vae
"""
import argparse

import pandas as pd

from src.backtest import level, plot_backtest, run_backtest, summary_table
from src.data import load_data
from src.experiment import experiment_dir
from src.portfolio import make_weights
from src.vae import RollingVAEVaR, VAEVaR
from src.var_models import GarchVaR, HistoricalSimulation, NormalVaR

PORTFOLIOS = {
    "50-50": make_weights(SP500=0.5, BONDS=0.5),
    "20-80": make_weights(SP500=0.2, BONDS=0.8),
    "80-20": make_weights(SP500=0.8, BONDS=0.2),
}
ALPHAS = (0.01, 0.05)  # 99% and 95%


def all_models():
    """Fresh instances every run (GARCH and the rolling VAEs keep state between days).
    The three VAE variants are the only ones tested on 2019-2024 (see README)."""
    return [
        HistoricalSimulation(window=500),
        NormalVaR(window=500),
        GarchVaR(dist="normal", window=1000, refit_every=20),
        GarchVaR(dist="t", window=1000, refit_every=20),
        VAEVaR(name="VAE-frozen"),
        RollingVAEVaR(refit_every=20, name="VAE-monthly"),
        RollingVAEVaR(refit_every=252, name="VAE-yearly"),
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


def run(weights, asset_returns, test_start):
    models, results = all_models(), {}
    for model in models:
        print(f"  {model.name} ...", flush=True)
        results[model.name] = run_backtest(model, asset_returns, weights, test_start, ALPHAS)
    return models, results


def save(out, name, weights, results):
    table = summary_table(results, ALPHAS)
    table.to_csv(out / f"var_backtest_{name}.csv")
    pd.concat(results, names=["model", "date"]).to_csv(out / f"var_forecasts_{name}.csv")
    weights_text = ", ".join(f"{w:.0%} {asset}" for asset, w in weights.items())
    for a in ALPHAS:
        plot_backtest(results, a, out / f"var_backtest_{name}_{level(a)}.png",
                      title=f"1-day VaR {level(a)}% backtest, {weights_text}")
    print(f"\nPortfolio {name}:")
    print_table(table)


def save_vae_logs(out, models):
    """When each rolling VAE was retrained, and which forecasts had clipped inputs
    (data_up_to = last day of data used; the forecast is for the next trading day)."""
    fits = [dict(model=m.name, fit=i + 1, **f) for m in models for i, f in enumerate(getattr(m, "fits", []))]
    clipped = [dict(model=m.name, data_up_to=d.date()) for m in models for d in getattr(m, "clipped_days", [])]
    pd.DataFrame(fits).to_csv(out / "vae_fits.csv", index=False)
    pd.DataFrame(clipped, columns=["model", "data_up_to"]).to_csv(out / "vae_clipped_days.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="name of the new results folder, e.g. E6_name")
    parser.add_argument("--portfolio", default="50-50", choices=[*PORTFOLIOS, "all"])
    args = parser.parse_args()
    out = experiment_dir(args.experiment)  # stops if it already exists

    train, test = load_data()
    asset_returns = pd.concat([train, test])
    names = list(PORTFOLIOS) if args.portfolio == "all" else [args.portfolio]
    runs = {}
    for name in names:
        print(f"\nPortfolio {name} (test {test.index[0].date()} to {test.index[-1].date()}):", flush=True)
        runs[name] = run(PORTFOLIOS[name], asset_returns, test.index[0])

    out.mkdir(parents=True)  # only now, so a crashed run leaves nothing behind
    for name, (_, results) in runs.items():
        save(out, name, PORTFOLIOS[name], results)
    save_vae_logs(out, runs[names[-1]][0])  # the same for every portfolio
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()

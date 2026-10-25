# IND310 – Market Stress Scenarios with Generative Models

University course project (IND310). We generate market stress scenarios with a
Variational Autoencoder (VAE) and later a GAN, and use them for risk measurement
and portfolio optimization.

## Goals
1. Generate stress scenarios of daily returns with a VAE (later a GAN).
2. Compute Value-at-Risk (VaR) and Conditional VaR (CVaR) from the scenarios.
3. Compare against baselines: historical simulation, normal distribution, GARCH.
4. Backtest VaR with the Kupiec and Christoffersen tests.
5. Do CVaR portfolio optimization.

## Data
- Daily prices 2004–2024 from Yahoo Finance via `yfinance`.
- Train: 2004–2018. Test: 2019–2024.
- Assets: S&P 500 (`^GSPC`) and bond ETF (`AGG`), USD. Add more in `YF_TICKERS` in `src/data.py`.
- OSEBX is not on Yahoo before 2013; its CSV loader is commented out in `src/data.py` for later.
- Data is not committed (see `.gitignore`); `src/data.py` downloads it into `data/`.
- Overview plot: `python -m src.plot_data` -> `results/data_overview.png`.

## VAE scenario generator
A conditional VAE (`src/vae.py`) generates 10,000 joint S&P 500 / AGG return scenarios for the next
day, given a summary of the last 60 days (volatility over 5/20/60 days, recent returns, stock-bond
correlation). It is trained once on 2004-2018 and then frozen. Its decoder is a Student-t, so the
scenarios have fat tails. All design choices were made on training data only (fit 2004-2015,
validate 2016-2018); the reasons are listed at the top of `src/vae.py`.

```bash
python -m src.train_vae      # train (seconds), saves checkpoints/vae.pt (not in git; fixed seed 42)
python -m src.vae_realism    # realism check vs real 2004-2018 data -> results/vae_realism*
```

## VaR/ES backtest
Portfolio of S&P 500 and AGG, rebalanced daily (50/50, 20/80, 80/20). For every test day,
1-day VaR and ES at 99% and 95% are forecast from data up to the day before with historical
simulation (500 days), a normal distribution (500 days), GARCH(1,1)-normal and GARCH(1,1)-t
(1000 days, refit every 20 days), and the VAE (10,000 scenarios per day). Violations are
tested with Kupiec (right number) and Christoffersen (no clustering).

```bash
python -m src.run_backtests                   # 50/50 (train the VAE first)
python -m src.run_backtests --portfolio all   # all three portfolios
```
Outputs in `results/`: `var_backtest_<portfolio>.csv` (table), `var_forecasts_<portfolio>.csv`
(daily VaR/ES), `var_backtest_<portfolio>_<99|95>.png` (plots).

## Structure
```
data/        downloaded data (ignored by git)
notebooks/   exploration and experiments
src/         reusable code (data loading, models, risk measures, backtests)
results/     figures and tables for the report
```

## Setup
```bash
conda env create -f environment.yml
conda activate ind310
jupyter lab
```

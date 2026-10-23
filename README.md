# IND310 – Market Stress Scenarios with Generative Models

University course project (IND310). We generate market stress scenarios with a
Variational Autoencoder (VAE) and later a GAN, and use them for risk measurement
and portfolio optimization.

## Goals
1. Generate stress scenarios of daily returns with a VAE (later a GAN).
2. Compute Value-at-Risk (VaR) and Conditional VaR (CVaR) from the scenarios.
3. Compare against baselines: historical simulation, normal distribution, GARCH.
4. Backtest VaR with the Kupiec test (proportion-of-failures).
5. Do CVaR portfolio optimization.

## Data
- Daily prices 2004–2024 from Yahoo Finance via `yfinance`.
- Train: 2004–2018. Test: 2019–2024.
- Assets: OSEBX (Norway), S&P 500 (`^GSPC`), bond ETF (`AGG`), in local currency.
- Data is not committed (see `.gitignore`). S&P 500 and AGG are downloaded by `src/data.py`;
  OSEBX is not on Yahoo before 2013, so save its history as `data/osebx.csv` (date + close columns).

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

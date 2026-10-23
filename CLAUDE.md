# CLAUDE.md

University course project IND310. The user is a beginner: explain steps briefly
and say why they are done.

## Project
Generate market stress scenarios with a VAE (later a GAN), compute VaR/CVaR,
compare against historical simulation, normal distribution and GARCH, backtest
with the Kupiec test, and do CVaR portfolio optimization.

## Data
- Daily prices 2004–2024 from yfinance, stored in `data/` (git-ignored).
- Assets: OSEBX (from `data/osebx.csv`), S&P 500 and AGG via yfinance. `src/data.py:load_data()` returns (train, test) log returns.
- Strict split: train 2004–2018, test 2019–2024. Never fit anything on test data.

## Layout
- `src/` reusable code; `notebooks/` experiments; `results/` figures/tables.

## Conventions
- Environment: conda, `environment.yml` (env name `ind310`). Add new packages there.
- Use log returns unless stated otherwise; fix random seeds for reproducibility.
- Keep logic in `src/` and call it from notebooks; keep notebooks readable.
- Do not commit data, checkpoints or `.env`.

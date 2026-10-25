# CLAUDE.md

University course project IND310. The user is a beginner: explain steps briefly
and say why they are done.

## Project
Generate market stress scenarios with a VAE (later a GAN), compute VaR/CVaR,
compare against historical simulation, normal distribution and GARCH, backtest
with the Kupiec test, and do CVaR portfolio optimization.

## Data
- Daily prices 2004–2024 from yfinance, stored in `data/` (git-ignored).
- Assets: S&P 500 (^GSPC) and AGG via yfinance (OSEBX loader is commented out in `src/data.py`; add assets in `YF_TICKERS`). `src/data.py:load_data()` returns (train, test) simple returns.
- Strict split: train 2004–2018, test 2019–2024. Never fit anything on test data.
- 2025 onwards is the final holdout: never download or use it (`check_no_holdout` in `src/data.py`).

## Experiment rules (to avoid overfitting the test period)
- Only three VAE variants are ever evaluated on 2019–2024: VAE-frozen, VAE-monthly, VAE-yearly
  (README, "Pre-registered test-period experiments"). Don't add test-period variants or tune on
  test results without the user's explicit decision.
- Never overwrite or delete results. Each experiment writes to a new `results/<experiment>/`
  folder (`src/experiment.py`); E1-E4 files sit directly in `results/`.
- Log every experiment in RESULTS.md (date, what, key settings, commit, output files, main
  numbers), including failures. Commit the code before running, so the entry can cite the commit.

## Layout
- `src/` reusable code; `notebooks/` experiments; `results/` figures/tables; RESULTS.md experiment log.

## VaR/ES backtesting
- Every VaR model (baselines and future VAE/GAN) subclasses `VaRModel` in `src/var_models.py` and
  implements `forecast(history, weights, alphas) -> {alpha: (VaR, ES)}`: `history` = asset returns
  up to the day before, VaR/ES as positive losses. Scenario models use `var_es_from_scenarios`.
- `src/backtest.py:run_backtest` calls `forecast` once per test day in order (models may keep
  state, so use a fresh instance per run); Kupiec + Christoffersen tests, 99% and 95%.
- Portfolios (daily rebalanced) and the model list are in `src/run_backtests.py`.

## VAE
- `src/vae.py`: conditional VAE (Student-t decoder, free bits, 60-day summary features clipped to
  the training range), `ScenarioGenerator`, `fit_generator` (shared by all trainings), and the
  plug-ins `VAEVaR` ("VAE-frozen") and `RollingVAEVaR` ("VAE-monthly"/"VAE-yearly", EPOCHS = 77).
  Design rationale is in its docstring. Use these names in all tables and plots.
- Order: `python -m src.train_vae` -> `python -m src.vae_realism --experiment <name>` (gate: 6 pass
  criteria fixed in advance) -> `python -m src.run_backtests --experiment <name>`. Checkpoint
  `checkpoints/vae.pt` is git-ignored; training is deterministic (seed 42, 1 CPU thread; the frozen
  model's weights checksum is 2e7e6b527cd6).
- Make model choices on fit 2004-2015 / validation 2016-2018 only, never on test-period results.

## Conventions
- Environment: conda, `environment.yml` (env name `ind310`). Add new packages there.
- Use simple returns (not log returns); fix random seeds for reproducibility.
- Keep logic in `src/` and call it from notebooks; keep notebooks readable.
- Do not commit data, checkpoints or `.env`.

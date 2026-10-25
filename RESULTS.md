# Results log

One entry per experiment, oldest first. Each entry has the date, what was done, the key settings,
the commit of the code that produced it, the output files and the main numbers. All results are
kept, including the ones that did not work; files are never overwritten (from E5 on, each
experiment has its own folder in `results/`). The test period 2019-2024 is only used for the
pre-registered experiments in the README; 2025 onwards is a final holdout and has not been downloaded.

Common setup unless stated otherwise: S&P 500 (`^GSPC`) and AGG daily simple returns; portfolios
50/50, 20/80, 80/20 (stocks/bonds), rebalanced daily; 1-day VaR and ES at 99% and 95%; test period
2019-01-02 to 2024-12-31 = 1,510 days, so 15.1 violations are expected at 99% and 75.5 at 95%.
Notation: **K** = Kupiec test (right number of violations), **C** = Christoffersen test (no
clustering); ✓ = pass, ✗ = reject at the 5% level.

| ID | Date | Experiment | Data used | Code commit | Outcome |
|---|---|---|---|---|---|
| E1 | 2026-10-25 | Baseline VaR/ES models | test 2019-2024 | `733e06d` | no baseline passes both tests at 99% |
| E2 | 2026-10-25 | VAE design choices | train/validation only | exploratory, see entry | Student-t decoder + free bits chosen |
| E3 | 2026-10-25 | VAE realism check | train 2004-2018 | `a1e95e3` | 6/6 criteria after 60-day memory + clipping |
| E4 | 2026-10-25 | VAE-frozen backtest | test 2019-2024 | `a1e95e3` | behaves like GARCH-t; fails Kupiec at 99% |
| E5 | 2026-10-25 | VAE-monthly and VAE-yearly | test 2019-2024 | see entry | pre-registered, not yet run |

---

## E1 · Baseline VaR/ES models · 2026-10-25
- **What:** four standard models forecast 1-day VaR and ES for every test day, using only data up
  to the day before.
- **Settings:** historical simulation (last 500 days); normal (rolling mean and std, 500 days);
  GARCH(1,1) with normal and with Student-t errors, constant mean, fitted on the last 1,000 days,
  parameters refit every 20 days (volatility updated daily in between).
- **Commit:** `733e06d`
- **Outputs:** `results/var_backtest_<portfolio>.csv`, `results/var_forecasts_<portfolio>.csv`,
  `results/var_backtest_<portfolio>_<99|95>.png`. Note: E4 regenerated these files with VAE rows
  added; the baseline rows were verified identical. The baseline-only files are in git at `733e06d`.
- **Main numbers, 99% (violations, expected 15.1):**

  | Model | 50/50 | 20/80 | 80/20 |
  |---|---|---|---|
  | Historical sim. | 20 (K ✓, C ✗) | 20 (K ✓, C ✗) | 21 (K ✓, C ✗) |
  | Normal | 31 (K ✗, C ✗) | 29 (K ✗, C ✗) | 33 (K ✗, C ✗) |
  | GARCH-normal | 34 (K ✗, C ✓) | 39 (K ✗, C ✓) | 34 (K ✗, C ✓) |
  | GARCH-t | 29 (K ✗, C ✓) | 32 (K ✗, C ✓) | 25 (K ✗, C ✓) |

- **95% (expected 75.5):** historical sim. 71/84/65 and normal 63/65/60 pass Kupiec (both fail
  Christoffersen for 20/80); GARCH-normal 97/100/102 and GARCH-t 103/105/119 fail Kupiec.
- **Notes:** historical simulation's violations cluster in 2020 and 2022 (50/50: 11 and 7, none in
  2019, 2021, 2023); GARCH's are spread out but too many. Refitting GARCH every day instead of every
  20 days gave identical violation counts (50/50: 34 and 29).

## E2 · VAE design choices (training data only) · 2026-10-25
- **What:** choose the VAE design. The test period was not used.
- **Data:** fit 2004-2015, validation 2016-2018 (754 days).
- **Selection rule (fixed before comparing):** lowest validation negative log-likelihood per day,
  estimated with 200 importance samples (IWAE). Lower is better.
- **Common settings:** latent dim 4, two hidden layers of 64, Adam lr 1e-3, batch 256, weight decay
  1e-4, KL warm-up 20 epochs, early stopping (patience 50), gradient clipping 5, seed 42.
- **Code:** exploratory scripts run during development; they are not in the repository and ran
  against an earlier, uncommitted version of `src/vae.py`. The chosen design is in `src/vae.py`
  at `a1e95e3`. The numbers below are copied from the run output.
- **Step 1, first attempt** (raw 20-day window, normal decoder; = variant A): validation loss lowest
  at epoch 20, then rose steeply (overfitting); all 4 latent dimensions unused (posterior collapse).
- **Step 2, variants** (stormy day = 2008-10-10; validation violations for the 50/50 portfolio,
  expected 7.5 at 99% and 37.7 at 95%):

  | Variant | Val. NLL (IWAE) | Latent dims used | Stormy-day S&P kurtosis | Val. 99% viol. | Val. 95% viol. |
  |---|---|---|---|---|---|
  | A raw 20-day window, normal decoder | 1.718 | 0 | -0.05 | 15 | 43 |
  | B 9 summary features, normal decoder | 1.625 | 2 | -0.01 | 11 | 42 |
  | C B + free bits 0.1 | 1.619 | 4 | 0.03 | 12 | 47 |
  | D summary features, constant decoder noise | 1.690 | 2 | -0.33 | 11 | 41 |
  | E raw window, constant decoder noise | 1.908 | 1 | -0.04 | 18 | 41 |
  | F summary features, Student-t decoder | 1.610 | 2 | 2.80 | 8 | 45 |
  | **G summary features, Student-t + free bits 0.1 (chosen)** | **1.598** | 4 | 3.01 | 10 | 41 |

- **Step 3, memory:** 20-day features (G) vs 60-day features (volatility over 5/20/60 days,
  correlation over 20/60 days): same validation NLL (1.598 vs 1.598). Realism (E3) decided: 60-day.

## E3 · VAE realism check (training data only) · 2026-10-25
- **What:** check that generated scenarios look like real 2004-2018 returns before any backtest.
  20 free-running simulated paths of 3,774 days each (each day generated from the model's own
  previous days), seed 42. Six pass criteria were fixed in advance (`check()` in `src/vae_realism.py`).
- **Commit:** `a1e95e3`
- **Outputs:** `results/vae_realism.csv`, `results/vae_realism_stylized.png`,
  `results/vae_realism_conditional.png`, `results/vae_training.png`.
- **Attempts:**

  | Design | Criteria passed | Failed |
  |---|---|---|
  | G, 20-day memory | 4/6 | S&P 1% quantile (-3.42% real vs band -3.40 to -1.89); stock-bond corr (-0.135 real vs -0.31 sim) |
  | 60-day memory | 5/6 | stability: simulated paths spiralled to daily S&P moves of 31-79% |
  | **60-day memory + inputs clipped to the training range (final)** | **6/6** | none |

- **Final model:** best epoch 77, validation NLL 1.598, Student-t degrees of freedom 4.95, 4/4
  latent dimensions used (KL about 0.1 each, at the free-bits floor), weights checksum `2e7e6b527cd6`.
- **Main numbers (real vs simulated median [5-95% band]):** S&P excess kurtosis 12.3 vs 7.7
  [4.6, 113]; S&P 1% quantile -3.42% vs -2.68% [-5.41, -1.80]; S&P autocorrelation of |r| at lag
  50 0.18 vs 0.13; stock-bond correlation -0.135 vs -0.119 [-0.257, -0.065]. One-step 99% VaR
  (50/50): 0.6% on a calm day (2017-06-01), 7.7% on a stormy day (2008-10-10).
- **Known weaknesses:** simulated extremes overshoot (9 of 20 paths contain an S&P day beyond the
  real maximum of 11.6%, the worst 44%); bond skew and kurtosis (-2.6 and 90 in reality, from a
  few 2008 days) are not reproduced; volatility clustering somewhat weaker than real; in-sample
  99% VaR (50/50, 2004-2018) had 46 violations vs 37 expected.

## E4 · VAE-frozen backtest · 2026-10-25
- **What:** the final VAE from E3, trained once on 2004-2018 and never updated, through the same
  backtest as E1.
- **Settings:** 10,000 scenarios per day; seed per day = 42 + date (yyyymmdd).
- **Commit:** `a1e95e3`
- **Outputs:** the E1 files in `results/`, now with VAE rows. In these files the model is called
  "VAE"; it is VAE-frozen.
- **Main numbers:**

  | | 50/50 | 20/80 | 80/20 |
  |---|---|---|---|
  | 99% violations (exp. 15.1) | 30 (K ✗, C ✓) | 36 (K ✗, C ✓) | 32 (K ✗, C ✓) |
  | 95% violations (exp. 75.5) | 102 (K ✗, C ✓) | 124 (K ✗, C ✓) | 97 (K ✗, C ✓) |
  | 99% avg ES vs avg loss on violation days | 1.88% vs 1.67% | 1.10% vs 1.09% | 2.92% vs 2.48% |

- **Checks:** a rerun gave byte-identical daily VaR/ES; baseline rows identical to E1.
- **Diagnostics:** violations per year track GARCH-t (50/50, 2019-2024: VAE 3/8/7/5/2/5, GARCH-t
  3/7/7/4/2/6). On 119 test days an input was outside the training range and was clipped, mostly
  the 60-day stock-bond correlation (above the training maximum of 0.47 on 86 days in 2022-24, up
  to 0.60) and S&P volatility and returns in March-April 2020.
- **March 2020 check (asked for before E5; nothing changed):** clipping was active on 20 forecast
  days, 2020-03-13 to 2020-04-09: S&P 5-day and 20-day volatility above the 2008 maximum, and on
  some days the last-day or 20-day S&P return below the training minimum. On those days the model
  treated the market as "as bad as the worst of 2008", not worse.

## E5 · VAE-monthly and VAE-yearly (rolling retraining) · 2026-10-25
- **Status:** pre-registered, not yet run.
- **What:** retrain the VAE from scratch on all data from 2004 up to the day before, every 20
  trading days (VAE-monthly, like GARCH) or every 252 trading days (VAE-yearly). Compared with all
  E1 baselines and VAE-frozen in one table.
- **Settings:** exactly the frozen model's (architecture, features, Student-t decoder, free bits,
  77 epochs, seed 42, no validation or re-tuning). Standardization and clipping range are
  recomputed from the data available at each retraining. The first retraining sees the same data
  as VAE-frozen and must reproduce it exactly (checksum `2e7e6b527cd6`).
- **Outputs (planned):** `results/E5_vae_rolling/`

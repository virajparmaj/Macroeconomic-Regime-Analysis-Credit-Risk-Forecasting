# Analysis scripts

Focused analytical review behind `ANALYSIS_REPORT.md`. Read-only with respect to
`data/`; every output goes to `results/`.

Run with the project interpreter from the repository root, e.g.
`~/.venvs/global/bin/python analysis/<script>.py`.

## Exploration

| Script | What it establishes |
|---|---|
| `profile.py` | Panel shape, missingness, duplicates, percentiles, autocorrelation |
| `daily2.py` | Daily vs monthly aggregation; intra-month range; peak compression (Insight 6) |
| `target_def.py` | How much of the reported R² is an artefact of the month-mean target |
| `rollcorr.py` | Rolling correlations in levels; every predictor flips sign |
| `stationarity.py` | ADF/KPSS; repeats the rolling correlations on stationary differences |
| `conditional.py` | Calm-vs-stress correlations, split on the spread's own level |
| `macro_regime.py` | Same split using a **macro-only** index, removing the circularity (Insight 1) |
| `leadlag.py` | Cross-correlations, Granger tests, recession warning leads (Insight 2) |
| `granger_robust.py` | Lead-lag robustness: crisis-excluded, point-in-time, sample halves |
| `breaks.py` | Sup-Wald break test and rolling 60-month R² |
| `r2_timing.py` | Whether rolling R² leads or lags stress (Insight 3) |
| `honest_eval.py` | Walk-forward vs the random walk (Insight 4) |
| `fair_eval.py` | Same, predicting the *change* so the random walk is nested |
| `horizon_class.py` | Horizons 1/3/6/12, regression and classification (Insight 5) |

`horizon_class.py` takes several minutes; the rest run in seconds.

## Figures and outputs

`fig1.py` … `fig6.py` write `results/figures/0*.png`. `export.py` writes
`results/key_numbers.csv`. `style.py` holds the shared matplotlib style.

Requires numpy, pandas, scipy, scikit-learn, statsmodels and matplotlib, plus
`src/` for `metrics`, `splits` and `baselines`.

# Study design to freeze before implementation

> **Historical planning document.** Implementation details and the partial-month correction are recorded in [the executable study](../study/README.md) and its [decision log](../study/DECISIONS.md). Executed results belong in the [results register](08_RESULTS_REGISTER.md); proposed outcomes below are not evidence.

Status: proposed specification. Historical data have already been inspected, so this is not retrospective preregistration.

## Forecast origin and targets

Let `m_t` be the observed-day monthly mean and `z_t` the latest daily spread available at the origin. Define origin `t` after the chosen month-end availability cutoff. Predict `m_(t+h)` at h = 1 (primary) and 3 (secondary). Use direct horizon-specific models. Predicting the residual `m_(t+h) - z_t` and adding `z_t` back makes the endpoint benchmark explicit.

Freeze q at the 75th percentile of the chosen target through December 2008. Recompute q for each explicitly different target definition. The existing **7.245238 pp** threshold applies to the legacy target only; do not transplant it silently to the observed-day target.

State outcome: `1[m_(t+h) >= q]`. Entry outcome, restricted to `m_t < q`: `1[max(m_(t+1), ..., m_(t+h)) >= q]`. Unknown future observations remain missing; entry requires the entire future path. Current state is known at origin under the chosen availability assumption.

All-origin state discrimination and at-risk entry discrimination have different populations. Do not subtract their AUCs and call the difference a causal effect of persistence. Report three panels: state on all origins, state on the common at-risk origins, and entry on those same at-risk origins. At h=1 the last two labels must coincide; at h>1 an entry can reverse before the endpoint.

## Matched feature blocks

| Block | Definition |
|---|---|
| S_mean | Current monthly mean, monthly changes at 1/3 months, trailing mean 3, trailing spread-change volatility 6 |
| S_last | S_mean plus latest available daily spread and endpoint-minus-monthly-mean gap |
| M | Publication-lagged inflation, industrial-production growth, rate/unemployment levels and changes, sentiment change, correctly named Euro-area growth level |
| R | Two-state K-means on standardized M, fitted only on the outer training data; neutral state IDs |

All transformations, scaling and clustering must use permissible data at the training cutoff. For ridge, add a regime indicator and interactions with M to allow state-dependent slopes. For RF, add the regime indicator and use the same capacity settings with/without R. R is a learned transformation of M, not new external information; any improvement is a representation/modeling gain. Fix cluster count at two; report occupancy and stability without interpreting clusters as structural economic causes.

## Models and estimands

Primary: h=1 ridge residual forecast using `S_last+M+R` versus `S_last+M`, evaluated by mean squared error on common origins. Ridge uses train-only standardization and alpha in {0.1, 1, 10, 100}, selected by chronological inner folds with label-maturity checks. Fold-fitted regime preprocessing must also be repeated inside tuning folds.

Secondary: `S_last+M` versus `S_last`; RF repetition; h=3; and comparisons against monthly-mean persistence, daily-endpoint persistence, training mean and AR(1). RF starts with fixed 200 trees, minimum leaf 5, all features, seed 42; sensitivity seeds 0/1/2/3/4 are not independent replications. Run `S_mean`, `S_last`, `S_last+M`, `S_last+M+R` using identical origins. Do not select the best test-period specification and call it the primary result.

For classification use training prevalence, a smoothed two-state transition baseline, standardized one-spread logistic baseline, full-block logistic, and RF. The single-spread logistic supplies calibrated probabilities in the sense of fitted probabilistic predictions, but calibration must still be assessed empirically. Raw current spread remains a ranking baseline only. Use unweighted classifiers for primary probability scores; any weighting/calibration experiment must be separately trained and labeled.

## Splitting and inference

Start evaluation at January 2009. Primary refit monthly, expanding window; annual refits and a 120-month rolling window are sensitivities. At cutoff T, training origin u is eligible only if the **label availability timestamp is no later than T**. For idealized monthly labels this means u+h<=T. Shared historical feature windows do not by themselves require an embargo. Inner validation follows the same maturity rule.

Within each comparison use exactly the same target dates, origins and availability rules. Report coverage. A cross-horizon summary uses the common window through August 2021; full horizon-specific samples remain separate tables.

Define incremental gain `I = 1 - MSE(regime)/MSE(no_regime)`. Report RMSE/MAE in pp and bps, benchmark-relative R², and paired loss differences. Use 5,000 paired calendar-block bootstrap draws, main block length 12 months, sensitivities 6 and 24. Keep predictions/actuals/model pairs together; preserve the complete calendar and apply at-risk masks after resampling. These intervals describe uncertainty in the evaluated forecast-loss series, not all uncertainty from reselecting the research design. Show crisis influence by deleting event windows from scoring only, keeping the original fitted forecasts.

A proposed practical margin is 5% MSE reduction, an explicit study convention without a trading-profit interpretation. A positive point estimate alone is insufficient. If the interval is wide, conclude inconclusive. If its upper bound is below 5%, the study can rule out that magnitude under the stated procedure; equivalence within ±5% requires the whole interval inside those bounds. Use one primary contrast and Holm adjustment for the stated secondary family. Do not export the old h−1-only DM p-values as final inference.

For probabilities report Brier/log loss, AUC, AP and prevalence. Undefined one-class AUCs remain NA. For entries report all three known event dates, warning lead times, misses and false-alarm episodes per at-risk year. Define an alarm using a training-only threshold achieving at most one false-alarm episode per at-risk year where feasible; consecutive alarm months form one episode and an episode is matched once to its first subsequent eligible entry. If calibration cannot meet the budget, record that explicitly. Three entries cannot support a broad crisis-warning reliability claim.

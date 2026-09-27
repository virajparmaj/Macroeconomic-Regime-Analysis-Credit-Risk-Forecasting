# Ordered implementation plan

**Planned work, not an implemented pipeline.** The paths below are proposed additions. Preserve existing notebooks and legacy outputs as historical evidence. Build reusable code first; notebooks should read the prediction ledger rather than own a second implementation.

## Milestones and acceptance gates

| Milestone | Concrete work | Deliverable | Gate to proceed |
|---|---|---|---|
| M0: freeze specification | Encode file 05, source identities, target definitions, date boundaries, seed, primary contrast and practical margin | `research/study/protocol.json`, dated decision log | No field described merely as “choose best”; input hashes captured |
| M1: data contract | Reconstruct daily/monthly targets; rename EA growth; build assumed availability dates; validate observed versus legacy means | `research/study/data.py`, manifest and quality report | Source reconciliation and all timestamp assertions pass |
| M2: walk-forward engine | Implement label-maturity eligibility, inner chronological tuning, train-only transforms and fold-fitted regimes | `research/study/features.py`, `regimes.py`, `splits.py`, `models.py` | Future-perturbation, threshold-freeze and label tests pass |
| M3: matched forecast matrix | Run baselines and the four feature blocks at h=1/3; emit predictions, failures and coverage | `research/study/run.py`; versioned `results/research/<run_id>/predictions.csv` | Identical origins within each contrast; zero duplicate forecast keys; scores recompute |
| M4: uncertainty/events | Add paired blocks, risk-set panels, event ledger, crisis influence and prespecified sensitivities | `research/study/inference.py`, `events.py`, generated tables | Intervals show block sensitivity; no inflated event sample; missing metrics reported |
| M5: external check | Freeze five-macro variant; obtain later data and vintages if feasible; run once without retuning on holdout | Extension manifest and separate results table | No undocumented variable substitution or retrospective threshold tuning |
| M6: paper assembly | Generate every table/figure and replace pending placeholders in manuscript | `research/study/report.py`, manuscript and release manifest | Every numerical claim maps to a ledger/table; full clean reproduction succeeds |

M0–M4 are the minimum credible historical study. M5 materially strengthens the novelty and generalization claim. If M5 is unavailable, document the failure and frame the result as a single-series historical case study. M6 may prepare an exploratory manuscript but cannot manufacture a strong finding from inconclusive estimates.

## First implementation batch

1. Create `research/study/` with importable modules and a small JSON protocol; no CLI framework needed.
2. Build immutable target/availability tables from the existing CSVs. Fail on duplicate/missing calendar keys, ambiguous daily files, nonnumeric values, or changed hashes unless the run explicitly declares new inputs.
3. Generate future labels and eligibility timestamps independently from features. Keep labels missing until observed.
4. Implement endpoint and mean persistence, AR(1), and ridge residual forecasts before RF. Use a short historical prefix as a smoke run, then the full specified dates.
5. Save the prediction ledger atomically with run/config hashes. Recompute metrics from that saved ledger in a separate step.
6. Add the two-state regime transform and rerun the paired models. Include the same latest-spread information in all full-information models.

## Minimum forecast ledger

`run_id, config_hash, input_hash, origin, cutoff_timestamp, target_date, target_available_at, target_definition, horizon, task, risk_set, threshold_value, threshold_training_end, model, feature_block, regime_fit_end, train_start, train_end, last_training_label_available_at, tuning_end, seed, prediction, actual, eligible, exclusion_reason`.

Classification probabilities and regression units must be explicit metadata. Key uniqueness includes run, origin, target, horizon, task, model and feature block. Log single-class training failures and model failures rather than silently dropping difficult periods. The comparison builder must either require a complete paired set or report why coverage differs.

## Existing code to reuse or replace carefully

| Current path | Treatment |
|---|---|
| `src/data.py`, `src/config_v2.py` | Reuse identities/paths; correct language and verify timing assumptions before treating as PIT |
| `src/splits.py` | Retain legacy behavior for reproduction; new engine purges by label availability, not feature lookback |
| `analysis/horizon_class.py` | Reproduce old diagnostics only; replace full-sample threshold and false future labels in study engine |
| `analysis/fair_eval.py` | Retain old result; replace default ridge tuning with scaled chronological tuning |
| `src/metrics.py` | Review serial dependence treatment; do not inherit h−1 bandwidth as sole inference |
| `analysis/export.py` | Do not use for final paper; replace hard-coded results with ledger-derived exports |
| `notebooks/v2/01...06` | Fill as readers/explanations after pipeline works; presently empty code scaffolds |

## Scope control

Do not begin with LSTM, transformers, HMM model search, SHAP plots, more macro series, or a dashboard. None resolves the current scientific uncertainty. A random-walk simulation can illustrate the established aggregation mechanism, but is optional supporting material rather than a new contribution. If all robust paired estimates are inconclusive, prioritize additional untouched data over expanding the model menu.

Rough planning allowance: M0–M2 about 1–2 focused working days; M3–M4 about 2–3; M5 depends on data/vintage access; M6 about 1–2 after stable results. These are planning estimates, not measured runtime or a completion promise.

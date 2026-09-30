# Research implementation prompt

Prepared 30 September 2026. **Status: execution instructions for a future coding task. Adding this file does not implement or validate the research pipeline.**

Copy the prompt below into a coding task opened in this repository. It is intended to carry the existing research plan through implementation, experiments, documentation and a pull request.

---

You are implementing the research study in **Macroeconomic-Regime-Analysis-Credit-Risk**. Act as a careful Python research engineer. Make the changes in this repository, execute the experiments that the available data support, test their validity, and produce reproducible results suitable for writing a research paper. Do not stop at another plan or a set of empty notebooks.

## Objective and scientific scope

Answer: **Do macroeconomic regime features improve monthly U.S. high-yield spread forecasts once models and benchmarks share the latest available spread information, and does any improvement extend to new stress entries?**

The intended contribution is a controlled empirical assessment, not a new forecasting algorithm. Preserve negative and inconclusive results. Do not tune the test period to obtain a favorable finding or describe inspected historical data as an untouched holdout.

Implement the historical study M0–M4, attempt the feasible external/vintage checks in M5, and generate the evidence-based paper materials in M6. If an external dataset cannot be obtained, complete all independent work and document the exact limitation. Do not invent data, substitute variables silently, or mark blocked experiments complete.

## Read first and respect repository state

1. Read applicable `AGENTS.md` instructions, inspect Git status, repository structure, dependency configuration, and existing code before editing.
2. Read `research/paper_preparation/00_README.md`, `03_DATA_AND_VALIDITY.md`, `05_STUDY_DESIGN.md`, `06_IMPLEMENTATION_PLAN.md`, `07_TESTS_AND_VALIDATION.md`, `08_RESULTS_REGISTER.md`, and `11_REPRODUCIBILITY.md`. Read the remaining package for interpretation and paper structure.
3. Inspect `src/`, `analysis/`, `tests/`, `research/audit_checks.py`, `research/evidence/`, the original data, and the v2 notebooks. Verify actual file contents; do not assume the current state still matches the earlier audit.
4. Use the narrowed primary contrast in `05_STUDY_DESIGN.md`; the broader `research/EXPERIMENT_PROTOCOL.md` is supporting context. Record any necessary resolution of conflicting specifications before running the affected experiment.
5. Work on **`viraj/research`**. Reuse it if it already exists. If needed, fetch and track the remote branch; create it from the current upstream default branch only when absent and safe. Preserve unrelated work. Never reset, force-push, or overwrite someone else's changes.
6. Create/use a project-local virtual environment. Resolve the declared Python >=3.12 versus historical Python 3.11 mismatch explicitly, lock actual research dependencies, and document the tested interpreter. Do not install into the global environment.

Preserve original data, legacy notebooks and historical results. Implement reusable study modules under `research/study/` as specified by the user-approved plan; do not create duplicate model implementations in notebooks. Keep existing public interfaces stable where practical. Avoid unrelated frontend, deployment or branding changes. Do not introduce an NUS affiliation in new paper materials.

## M0 — Freeze the experiment specification

Create a versioned `research/study/protocol.json` and dated decision log before the full run. Include source identities and hashes, target definitions, availability assumptions, feature formulas, horizons, train/evaluation boundaries, refit schedule, tuning rules, seeds, model settings, contrasts, bootstrap settings, practical margin and event matching rules.

Specify implementation details left open by the plan, such as exact macro transformations, inner validation windows, minimum training sizes, transition smoothing, classification regularization, probability clipping, and event exclusion windows. Choose defensible defaults from training data and document them; never resolve these choices by inspecting the outer test scores. The protocol is frozen after exploratory historical inspection, not a retrospective preregistration.

## M1 — Reconstruct data and availability

Implement `research/study/data.py` with a source manifest and data-quality report.

- Keep the observed-day monthly spread mean as the main target and the existing source-grid forward-filled mean as a separate compatibility target. Reconcile both to the daily source. Do not fill dates absent from the source grid when reproducing the legacy target.
- Correct the misleading `GDP` identity to the Euro Area 19 OECD growth reference series. It is not U.S. GDP.
- Distinguish reference/observation dates from release availability and vintage dates. Use a transparent availability ledger. Fixed shifts on revised data are a publication-lag approximation, not verified point-in-time observations.
- Make the forecast cutoff precise. A same-month mean or latest daily value may enter a forecast only under the stated availability rule. If historical publication times cannot be verified, label the idealized assumption and implement a one-business-day availability sensitivity without admitting future observations.
- Validate ordered unique calendar keys, numeric values, missingness, source identity and units. Known source missing markers are explicit missing values; unexpected malformed values fail validation. Record exclusions rather than silently dropping rows.
- Generate labels separately from features. Unknown future targets stay missing. Entry labels require the complete future path.

## M2 — Implement causally ordered evaluation

Create importable `features.py`, `regimes.py`, `splits.py`, and `models.py` modules, keeping functions focused and typed where useful.

For target mean `m_t` and latest available spread `z_t`, forecast `m_(t+h)` at h=1 and h=3. Use residual learning `m_(t+h) - z_t` for the main learned models, adding `z_t` back at prediction. A restricted S_mean model may use its own available mean anchor; label that distinction explicitly and compare the reconstructed level predictions on common origins.

Use monthly expanding refits beginning January 2009. A training origin is eligible only when its entire label is available at the refit cutoff; in the idealized monthly case, `u+h <= T`. Apply the same rule to inner tuning folds. Do not impose a gap merely because training and validation features share legitimate past observations.

Implement these feature blocks:

| Block | Features |
|---|---|
| S_mean | Current monthly mean, changes over 1/3 months, trailing mean 3, trailing spread-change volatility 6 |
| S_last | S_mean plus latest available daily spread and endpoint-minus-mean gap |
| M | Documented publication-lagged macro transformations from the study design |
| R | Two-state K-means on standardized M, fitted within each training fold |

Evaluate S_mean, S_last, S_last+M, and S_last+M+R. Fit scaling, clustering, interactions and tuning strictly within each applicable outer/inner training fold. No centered smoothing, full-sample fitting or future-informed state naming. For ridge, add regime indicator and macro interactions; for RF, add the indicator with otherwise identical capacity. R is a representation of M, not additional external information.

## M3 — Run the matched forecasting matrix

Implement `run.py` with a lightweight command-line interface, smoke mode and full protocol mode. Start with baselines and ridge; add RF after data/split checks pass.

**Primary contrast:** h=1 ridge S_last+M+R versus ridge S_last+M, measured by paired MSE on identical origins. Tune ridge alpha over `{0.1, 1, 10, 100}` using chronological inner validation and train-only scaling.

**Secondary comparisons:** macro ablation S_last+M versus S_last; RF repetition; h=3; monthly-mean persistence, endpoint persistence, training-mean and AR(1) baselines. Start RF with 200 trees, minimum leaf 5, all features and seed 42. Prespecify annual-refit, rolling-120-month, legacy-target and RF-seed sensitivities. Seeds are not independent statistical replications.

For classification freeze q75 using the target-specific sample through December 2008; q70/q80 are sensitivities. Recompute thresholds for each target definition. The legacy threshold 7.245238 pp is a diagnostic reference, not the observed-day threshold.

- State: `1[m_(t+h) >= q]`.
- Entry: on origins with `m_t < q`, predict any crossing during `t+1,...,t+h`.
- Report all-origin state, at-risk state, and at-risk entry panels. At h=1 the latter two labels must agree. Their equivalence does not extend to longer horizons.
- Include training prevalence, a smoothed two-state transition model, one-spread logistic, full-block logistic and RF. Distinguish h-step terminal-state probability from first-entry probability in the transition baseline.
- Use unweighted probability models for primary scores. A logistic output is not proof of calibration; assess it. Raw spread is only a ranking score. Define and log single-class training fallbacks before execution.

Save predictions atomically to a new, non-overwriting `results/research/<run_id>/` directory. Never silently exclude failed model forecasts to improve scores. Require common-origin coverage for paired comparisons or clearly mark the comparison incomplete.

The ledger must contain at least:

```text
run_id, config_hash, input_hash, origin, cutoff_timestamp, target_date,
target_available_at, target_definition, horizon, task, risk_set,
threshold_value, threshold_training_end, model, feature_block,
regime_fit_end, train_start, train_end, last_training_label_available_at,
tuning_end, seed, prediction, actual, eligible, exclusion_reason
```

Record regression units, probability/score type and any fallback separately. Validate key uniqueness. Save enough configuration, input provenance and fit metadata to reconstruct each prediction.

## M4 — Inference and event evaluation

Implement `inference.py` and `events.py`. Recompute all metrics from saved predictions rather than model objects or copied constants.

- Report RMSE/MAE in pp and bps, benchmark-relative R², paired loss differences, and incremental MSE gain `1 - MSE(regime)/MSE(no_regime)`.
- Use 5,000 paired calendar-block bootstrap draws, main block length 12 months, sensitivities 6/24. Resample the full calendar with model pairs together, then apply at-risk masks. State the bootstrap construction, interval level and degenerate-draw handling.
- Treat 5% MSE reduction as the proposed practical margin, not a trading-return threshold. Distinguish evidence of gain, evidence ruling out that gain, and an inconclusive interval. Do not interpret failure to reject as equivalence.
- Specify the secondary test family and a valid dependence-aware test procedure before applying Holm adjustment. Do not infer adjusted significance from unadjusted intervals or reuse the legacy h−1-only DM inference uncritically.
- Report Brier score, log loss, AUC, average precision and prevalence; mark undefined statistics explicitly.
- Generate an event ledger with entry dates, matched alarm episodes, lead times, misses and false alarms per at-risk year. Choose alarm thresholds using training-only chronological out-of-fold forecasts, not in-sample fitted probabilities or test outcomes. Enforce the prespecified false-alarm budget where feasible; record no-alarm/infeasible cases.
- Consecutive alarm months form an episode. Match an episode to at most one entry within its forecast horizon and count each warned entry once. Use the actual complete risk-set calendar when calculating exposure.
- Report event-window deletion sensitivities by removing scoring windows from the original forecast ledger, not refitting on a hindsight-filtered sample.

The earlier three-entry count applies to the frozen legacy target. Recompute event counts for each actual target. Overlapping positive windows are not independent crises. Keep claims proportional to the event count.

## M5 — Attempt frozen external and vintage checks

Freeze a five-macro variant excluding the discontinued Euro-area series and rerun it historically before evaluating September 2022 onward. Obtain public data if accessible, save provenance, and evaluate without test-period retuning. Record prior exposure to this extension. If a second series is used instead, label it a different generalization test; correlated indices do not create independent crises.

Attempt a vintage-data subset on common origins. Never call shifted revised values genuine vintages. If data access or redistribution terms prevent a check, retain a precise pending/blocked status, explain the claim limitation, and finish the historical study. Do not publish restricted raw data or credentials.

## M6 — Reports, paper materials and integration

Implement `report.py` to generate tables, static scientific figures and Markdown from versioned ledger artifacts. Update the results register P01–P09 and the conclusions only when the corresponding evidence exists. Include run IDs, targets, dates, n, event counts, intervals and coverage. Keep failed/inconclusive results visible.

Populate the v2 notebooks as thin readers of the new modules and artifacts after the pipeline works. Preserve legacy notebooks and exploratory evidence. Add documented reproduction commands to the research README. Do not route manuscript numbers through the old hard-coded exporter.

Use the existing paper blueprint to produce a manuscript draft that distinguishes completed historical experiments from missing external checks. Every numerical claim must map to a generated table or source audit. Do not claim a novel benchmark principle, causation, borrower default prediction, profitable trading, or publication readiness unsupported by the evidence.

## Required tests and verification

Implement meaningful behavioral tests corresponding to T01–T15 in `07_TESTS_AND_VALIDATION.md`: label maturity; missing future paths; future perturbation invariance; frozen thresholds; inner/outer fit isolation; endpoint versus entry labels; already-stressed risk exclusion; release/vintage availability; target/units reconciliation; paired forecast uniqueness; constant-series baselines; single-class probability handling; calendar-block pairing; event accounting; and deterministic reproduction.

Keep existing tests passing. Correct misleading test descriptions when touched; do not weaken assertions to accommodate leakage. Convert the old boolean-return checks to real assertions if they are included in the claimed validation gate. Run focused tests, a smoke experiment, the full required matrix and reports. Verify that a second execution under the same locked environment reproduces keys and scores within documented tolerances. Record commands, duration, warnings and failures. Check resource use before full execution; use bounded parallelism on the local machine.

## Git delivery and definition of done

Commit only relevant changes in the repository's required commit-message format. Push `viraj/research` without force. Reuse/update its existing open PR; create one against the verified default branch only if absent. Rewrite the PR title/body around the final delivered implementation, with validation and remaining limitations. Keep it draft if required study work is incomplete. Do not merge the PR.

Inspect ignore rules before delivery: this repository ignores most generated results and data. Deliberately track the small, nonsecret derived tables, manifests and documentation needed for review, using narrow exceptions where appropriate. Keep bulk artifacts out of Git with reproducible acquisition/generation instructions. Never broadly unignore raw data, local prompts or credentials.

The task is complete when the required historical pipeline runs end to end, the validity tests pass, ledger-derived results and uncertainty exist, notebooks/docs reflect actual execution, and a reviewable PR is open. M5 access limitations must be explicit. A scaffold, passing legacy tests, or copied audit numbers do not satisfy this definition.

In the final report provide the branch and PR link, implemented milestones, exact tests/experiments executed, primary effect and uncertainty, artifact locations, and any remaining limitations. Report what was established, not what the study hoped to find.

# Frozen implementation decisions — 30 September 2026

This historical design was chosen after exploratory inspection, not preregistered.

- Exclude December 1996 and August 2022, each represented by one daily observation. The main target has 307 complete calendar months. Recompute thresholds on January 1997–December 2008. Earlier 309-row diagnostics are preserved, not promoted to final evidence.
- Keep the observed-day target primary. Legacy forward-fill reproduces the existing source grid only. Missing calendar months remain missing in either definition.
- Python 3.13 in a local environment; NumPy/pandas/sklearn versions match the earlier audit. Package versions are locked before the full run.
- Monthly mean availability sensitivity is deliberately conservative: next weekday after month-end. Thus current-month stress state is unavailable at month-end in that sensitivity; evaluate regression only, not fabricated at-risk entry classifications.
- Inner ridge validation uses three latest 12-month calendar blocks with label maturity; 60 training rows minimum, pooled MSE, stronger-penalty tie break. Outer regression/state needs 96 rows; entry requires 60 at-risk rows because exclusion from risk is part of its estimand, not a missingness failure.
- Classification has fixed C=1 or fixed forest capacity. Historical out-of-fold calibration begins in 2005. Its labels use the initial-sample frozen threshold as a fixed retrospective measurement definition; these pre-2009 scores are calibration data, not claimed real-time performance. Outer evaluation starts in 2009.
- Two-state transition estimates include only adjacent observed calendar months and add one to every cell. Constant-series AR uses least-squares predictions without unstable polynomial fitting.
- Sensitivities are one-at-a-time. Annual/window/target/delay/seed checks run the ridge/RF regime pair plus baselines. Threshold checks run all classification blocks; the five-macro historical check precedes external validation.
- At most four workers, estimator/BLAS threads bounded to one, four-hour batches. Completed jobs are immutable atomic checkpoints; an interrupted active job is rerun rather than salvaged partially. In-flight jobs finish when the batch budget expires; no further jobs start.
- Public artifacts contain aggregates and manifests. Detailed spread observations, prediction ledgers, alarms and source downloads remain local. No new affiliation is asserted.

## Implementation organization and completion notes — 1 October 2026

The planned regime component is the fold-local `Representation` class in `models.py`; keeping it with preprocessing avoids a second implementation. `runner.py` supplies orchestration and `python -m research.study` supplies the CLI planned as `run.py`. These are file-organization changes, not changes to the experimental contrasts.

The free later-period source has missing macro observations that exclude November and December 2025, leaving 26/24 origins at horizons 1/3. These exclusions follow the frozen missingness rule. No later-period tuning or gap imputation was introduced. After the runs, completed-run resumption was made a no-op and covered by an immutability test; no forecasting calculation changed.

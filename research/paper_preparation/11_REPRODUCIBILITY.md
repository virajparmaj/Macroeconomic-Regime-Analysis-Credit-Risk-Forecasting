# Reproducibility and handoff

The implemented study runs with **Python 3.13.15** in the project `.venv`. [requirements.lock](../study/requirements.lock) pins the actual environment. `pip check` reports no broken requirements. This resolves the earlier global Python 3.11 versus declared Python >=3.12 mismatch. Package versions and operating-system metadata are recorded in every run manifest.

## Reproduction commands

See [the study README](../study/README.md) for installation, acquisition and the full CLI. From the repository root:

```sh
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests test_logging.py -q
.venv/bin/python -m research.study validate
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.study run --profile core --workers 4 --hours 4
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.study run --profile sensitivity --workers 4 --hours 4
.venv/bin/python -m research.study.external
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.study run --profile external --workers 4 --hours 4
MPLCONFIGDIR=/tmp/research-mpl XDG_CACHE_HOME=/tmp/research-cache .venv/bin/python -m research.study report --run RUN_ID
.venv/bin/python research/verify_study.py RUN_ID
.venv/bin/python research/verify_study.py FIRST_CORE_ID --compare REPEATED_CORE_ID
```

Use a new run ID for each execution. Resume an incomplete run with its original `--profile` and `--resume RUN_ID`. A changed manifest fails; completed jobs and completed forecast/status files remain unchanged. At most four workers run, with estimator/BLAS threads constrained to one. At four hours, no new jobs start and active jobs finish. No full experiment approached the batch limit.

To rebuild the paper artifacts, set `research/study_results/latest_runs.json` to four verified run IDs, refresh verification records, then run:

```sh
.venv/bin/python research/assemble_study.py
MPLCONFIGDIR=/tmp/research-mpl XDG_CACHE_HOME=/tmp/research-cache .venv/bin/python research/plot_study.py
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python research/execute_notebooks.py
```

Jupyter uses the project interpreter and local kernel ports. A sandbox may require local port permission. Readers 01–06 are executed; legacy audit notebook 00 is preserved without rerunning it.

## Delivered runs

| Profile | Run ID | Jobs | Forecast seconds |
|---|---|---|---|
| core | `20261001T021109Z-947a55` | 72 | 267.01 |
| sensitivity | `20261001T022614Z-d43c7b` | 236 | 713.59 |
| external | `20261001T024151Z-f6c6a9` | 16 | 4.57 |
| repeat | `20261001T023825Z-9610b2` | 72 | 204.22 |

Durations measure forecasting batches, excluding acquisition/report rendering. A 72-job smoke run completed first. The final suite took 25.36 seconds. Repeated-core predictions match all 14,040 rows exactly. The repeat uses the same frozen settings and environment after equivalent provenance/orchestration refactoring; differing code hashes are disclosed. The subsequent completed-resume guard changes no prediction calculation.

## Provenance and access

Each local run includes protocol/environment/input hashes, source snapshots and a dirty-tree patch where applicable. Public manifests identify the Git revision at launch. The first core predates the implementation commit; its source snapshot was preserved locally. The run code hash identifies executed source, which can differ from subsequent reporting or orchestration changes. Detailed observations and per-origin prediction/availability/alarm ledgers remain ignored; aggregate CSVs, figures and manifests are tracked.

Original licensed data are expected at the paths in `research/study/input_manifest.json`. Their hashes must match. They are not added to this PR. New FRED exports can change with revisions and the rolling spread access window: reacquisition need not reproduce the frozen snapshot, and exact external reproduction needs the retained matching files. [Acquisition metadata](../study_results/external_acquisition.json) records URLs, retrieval time and hashes. No credential is stored. Public review artifacts do not remove raw-data access requirements for independent reproduction.

The external target ends August 2026, with features beginning April 2024 and missing macro origins in November–December 2025. It has 26/24 origins at h=1/3. Historical vintage requests returned HTTP errors without a configured credential; P09 is unavailable. Free access cannot reconstruct the missing continuous post-2022 spread interval.

## Failures and warnings

No delivered job has a model failure. An accidentally duplicated sensitivity process was interrupted early and excluded from the registry. Notebook execution initially failed because the sandbox blocked a local Jupyter socket; permission was requested for normal kernel execution. A matplotlib font-cache warmup delayed initial reporting. Pip emitted a nonwritable-cache warning and used no cache; dependency consistency passed. Historical exploratory logs and the original audit validation report remain untouched.

The manuscript is an exploratory empirical draft. Closest-work full-text comparison and stronger real-time or independent crisis validation remain future work. No institutional affiliation is added.

# Reproducible credit-spread study

The historical study uses complete months from January 1997 through July 2022. December 1996 and August 2022 in the original daily file each contain only one observation and are excluded. Macro inputs are revised snapshots with assumed publication lags, not verified historical vintages.

## Setup

Use Python 3.13 and a project-local virtual environment:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r research/study/requirements.lock
```

The lock records the environment used for the experiments. Original inputs remain under `data/original/`; `input_manifest.json` pins their SHA-256 hashes. A changed source requires an explicit new input/protocol snapshot, not an unnoticed rerun. Original licensed ICE observations are not redistributed by this implementation.

## Commands

Run from the repository root:

```sh
.venv/bin/python -m research.study validate
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests test_logging.py -q
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 .venv/bin/python -m research.study run --profile smoke
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 .venv/bin/python -m research.study run --profile core
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 .venv/bin/python -m research.study run --profile sensitivity
.venv/bin/python -m research.study.external
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 .venv/bin/python -m research.study run --profile external
MPLCONFIGDIR=/tmp/research-mpl XDG_CACHE_HOME=/tmp/research-cache .venv/bin/python -m research.study report --run RUN_ID
```

Every run prints its ID. `--workers` accepts 1–4 and `--hours` accepts a positive value up to 4. Completed jobs are atomic checkpoints. To resume, specify both the original profile and `--resume RUN_ID`; a changed code/input/configuration/environment manifest is rejected. A completed job is never overwritten. An interrupted in-flight job is recomputed. At the time budget, no new jobs start and existing jobs finish.

Detailed ledgers, source snapshots and alarms are stored in ignored `results/research/<run_id>/`. Each run includes its source-code snapshot and dirty-tree patch. Generated aggregate reports live in `research/study_results/<run_id>/`; `latest_runs.json` identifies the delivered study. Reports do not publish per-origin actual spread levels.

## Interpretation and limits

The primary comparison is the one-month ridge model with macro-regime features against the identical information set without regime features. Statistical preprocessing is fitted inside each outer and inner training fold. Reported gains are relative MSE reductions, not returns or default-risk improvements.

There are three separate classification panels: all-origin state, state among at-risk origins, and first entry among at-risk origins. Raw spread is a ranking score only. An unavailable future is never treated as a negative. Calibration uses past out-of-fold predictions; insufficient calibration evidence produces no alarms with an explicit status.

The core, sensitivity and later-period external profiles are separate. FRED's public spread export starts October 2023, leaving a gap after the historical file. External models freeze July 2022 parameters and require a contiguous seven-month feature history; they do not interpolate that gap. Revised macro vintages remain a limitation. The vintage API access attempt requires a configured credential and currently has no usable historical vintage panel.

The historical period informed the design. Results are exploratory and must not be described as preregistered confirmation. See the generated manuscript and result register for measured findings, including negative and inconclusive outcomes.

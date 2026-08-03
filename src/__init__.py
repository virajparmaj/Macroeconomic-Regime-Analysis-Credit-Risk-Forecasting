"""Corrected v2 library for macroeconomic regime analysis and credit-spread forecasting.

This package is a deliberate parallel implementation of the logic in ``utilities/``.
The duplication is intentional: ``utilities/`` is frozen as the historical record of
what was originally reported, and ``notebooks/v2/00_audit_of_v1.ipynb`` imports both
packages side by side in order to diff their behaviour.

Nothing in this package may be imported by ``utilities/``, and nothing here writes to
``data/*.csv``. The single permitted writer of ``results/metrics.csv`` is
:mod:`src.evaluate`.

Modules:
    config_v2: Series metadata (including publication lags), paths and seeds.
    data: Loading and point-in-time alignment of the macro/credit panel.
    features: Leakage-safe feature engineering (mirrors the v1 public API).
    splits: Walk-forward origins with purge and embargo.
    baselines: Random walk, AR(1) and train-mean constant.
    metrics: Error metrics, out-of-sample R2, Diebold-Mariano, regime-conditional error.
    evaluate: The only module permitted to write results/metrics.csv.
"""

__version__ = "2.0.0"

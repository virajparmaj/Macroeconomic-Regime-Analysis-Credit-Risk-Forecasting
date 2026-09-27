# Reproducibility and handoff

## Evidence already available

The diagnostic scripts are `research/audit_checks.py` and `research/plot_diagnostics.py`. Inputs are located relative to the repository by the audit script. It writes to `research/evidence/`, so reruns replace the earlier diagnostic outputs; snapshot them before changing the analysis. Legacy analysis scripts contain user-specific input paths.

From the repository root, the interpreter used for the audit can reproduce current diagnostics:

```sh
/Users/veerr_89/.venvs/global/bin/python research/audit_checks.py
/Users/veerr_89/.venvs/global/bin/python research/plot_diagnostics.py
/Users/veerr_89/.venvs/global/bin/python -m pytest -q
```

The existing environment reports Python 3.11.14, pandas 3.0.0, NumPy 2.4.1 and scikit-learn 1.8.0. The project declares Python >=3.12 and currently does not declare all research dependencies. These observations are an environment mismatch to resolve, not a portable installation recipe. Before the final study, choose a supported Python version, lock the actual dependencies and rerun the diagnostics/tests in that environment. Save OS, BLAS/thread configuration and seeds where they affect reproducibility.

## Final run requirements — planned

Write each run under `results/research/<run_id>/` with its frozen protocol, code revision plus dirty-tree patch/hash where necessary, input manifest, availability assumptions, dependency lock, forecast ledger, metrics, inference outputs, event ledger and execution log. Never overwrite a completed result set. Use relative paths in persisted manifests so another machine can reproduce the run.

Generate tables and figures from saved predictions. Do not copy hard-coded old export values into the paper. A second clean execution should reproduce forecast keys and scores within explicitly stated numerical tolerances.

## Publication checklist

- All required P01–P07 results are completed or explicitly excluded with consequences for claims.
- External/vintage checks P08–P09 have results or transparent limitations.
- Methods describe actual data availability rather than the stronger interpretation suggested by old function names.
- Main/secondary/exploratory contrasts and protocol amendments are visible.
- Three event entries are not described as dozens of independent warning successes.
- Raw-data redistribution terms are verified; otherwise provide permitted acquisition instructions and hashes.
- Literature versions and access limitations are recorded; no unsupported first-of-kind claim remains.
- Tables, figures, abstract and conclusion agree with the same versioned run.

The planning package and verified exploratory evidence are ready for implementation handoff. The final empirical pipeline, confirmatory extension and submission-ready manuscript are still pending.

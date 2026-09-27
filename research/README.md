# Research advisory deliverables

**Start with [paper_preparation/00_README.md](paper_preparation/00_README.md)** for the narrowed research contribution, additional literature, verified findings, ordered implementation plan, test requirements, results register and manuscript blueprint. This is a planning/evidence package; the final paper experiments remain pending.

- `RESEARCH_ADVISORY.md`: assessment, evidence corrections, eight research questions, recommended story, study design, paper outline and provisional abstract.
- `LITERATURE_REVIEW.md`: 13 related studies, overlap/gaps, primary-source links and access limitations.
- `EXPERIMENT_PROTOCOL.md`: essential/recommended/optional experiments, inference, falsification criteria and reproducibility requirements.
- `evidence/`: numerical diagnostics, source reconciliation, predictions and rerun logs. These are exploratory checks, not completed paper experiments.

From the repository root, reproduce with the interpreter used in this review:

```sh
/Users/veerr_89/.venvs/global/bin/python research/audit_checks.py
/Users/veerr_89/.venvs/global/bin/python research/plot_diagnostics.py
```

To rerun existing scripts without writing their output CSVs into the repository root, run them from `research/evidence/`:

```sh
cd research/evidence
/Users/veerr_89/.venvs/global/bin/python ../../analysis/honest_eval.py > honest_eval.log
/Users/veerr_89/.venvs/global/bin/python ../../analysis/fair_eval.py > fair_eval.log
/Users/veerr_89/.venvs/global/bin/python ../../analysis/macro_regime.py > macro_regime.log
```

The legacy scripts have user-specific absolute input paths. The new diagnostic locates the repository relative to its own file. Dependency versions and input hashes are in `evidence/audit_summary.json`; statistical caveats are in the advisory. Do not rerun `analysis/export.py` to generate manuscript results: it contains hard-coded numbers.

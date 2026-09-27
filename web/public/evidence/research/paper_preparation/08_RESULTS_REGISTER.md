# Results register and table plan

## Completed exploratory evidence

| ID | Result | Artifact | Status |
|---|---|---|---|
| A01 | Panel coverage, source hashes, target reconstruction | `../evidence/audit_summary.json` | Verified exploratory |
| A02 | Observed versus legacy monthly target reconciliation | `../evidence/target_provenance.csv` | Verified exploratory |
| A03 | Old RF and delta-model forecast reruns | `../evidence/honest_eval.log`, `fair_eval.log` | Reproduced; methodological limitations remain |
| A04 | Correct unknown-label handling and raw spread ranking baseline | `../evidence/classification_label_diagnostic.csv` | Verified exploratory; full-sample threshold retained |
| A05 | Last daily versus mean benchmark | `../evidence/aggregation_benchmarks.csv` | Verified exploratory; availability assumptions remain |
| A06 | Frozen-threshold entry counts | `../evidence/onset_counts.csv` | Verified counts; no onset forecasting result |
| A07 | Existing software tests | `validation_report.txt` | See recorded rerun; not scientific validity certification |

## Required paper results still pending

| ID / paper table | Question | Exact required output | Current result |
|---|---|---|---|
| P01 / Table 1 | What could be known at each origin? | Series dictionary, lag/vintage rules, target reconciliation, usable counts | Partial A01/A02; availability reconstruction pending |
| P02 / Table 2 | Does current information change baseline/model comparisons? | S_mean versus S_last; mean/endpoint/AR baselines; identical targets and origins | Baseline pilot A05 only; matched models pending |
| P03 / Table 3 | Do regimes add prediction after S_last+M? | Primary h=1 ridge paired MSE gain and interval; RF/h=3 secondary | **Not run** |
| P04 / Table 4 | Does M add value after S_last? | Paired macro ablation and uncertainty | **Not run under final design** |
| P05 / Table 5 | Does state discrimination survive risk-set controls? | All-origin state, at-risk state, at-risk entry; probability/ranking metrics | **Not run** |
| P06 / Table 6 | Were entries warned with tolerable false alarms? | Event dates, matched alarms, lead time, misses, alarm episodes/year | Counts only; **forecasts not run** |
| P07 / Appendix A | Are conclusions robust to dependence and choices? | Block lengths, q70/q80, event influence, rolling window, legacy target | **Not run** |
| P08 / Table 7 | Does the frozen result extend? | Later-date or clearly labeled second-series evaluation | **Not acquired/run** |
| P09 / Appendix B | Does revision timing alter conclusions? | Vintage subset versus revised lagged macro on common origins | **Not acquired/run** |

Never fill a pending cell with an expected direction, a number from a different target, or an earlier method's statistic. Publishable summary tables should include target, origin window, n, event count where relevant, benchmark, loss, paired effect, interval, and whether the result was primary or exploratory.

The main table must retain failed competitors and report coverage. A negative result can support a useful paper only with adequate precision and clear scope; an uninformative interval belongs in the conclusions as uncertainty.

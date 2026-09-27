# Macroeconomic Regime Analysis & Credit Spread Forecasting

An interactive research portfolio within the existing React 18, TypeScript, Vite, Tailwind and D3 application. The site separates historical experiments from corrected exploratory evaluations and work that remains proposed.

## Run the application

```sh
cd web
npm ci
npm run dev -- --port 5177
npm run typecheck
npm run build
```

The checked-in JSON and evidence snapshots are sufficient to build. Python and raw input data are not required to view or build the site. `dist/` is the static build; relative Vite asset paths support deployment below a directory prefix.

## Evidence before interface

Read [EVIDENCE_MAP.md](EVIDENCE_MAP.md) for the inventory prepared before implementation, artifact lineage, corrected claims and limits. Its source hierarchy favors executable logic and stored records over old narrative summaries.

- Core panel: 309 monthly rows, six macro variables and one aggregate OAS target, December 1996–August 2022.
- Primary forecast evaluation: 163 origins, January 2009–July 2022; targets in the following month.
- Prediction paths: only the two RF future-level specifications with persisted records; persistence comes from those same records.
- Change-model scores: rounded stored log summaries, with missing prediction paths and metrics explicitly unavailable.
- Classification: corrected known-future rows, raw current-spread comparator and full-sample threshold caveat.
- Regime ablation, event warning, dependent-data uncertainty, vintages and external validation: pending.

Original notebooks are frozen. Neither the website nor its exporter executes them, trains models or generates live predictions.

## Regenerate the presentation data (optional)

The input files under `data/` and `research/evidence/` must exist locally. In particular, the historical rerun `.log` files are required; root ignore rules may omit these from other checkouts. The exporter fails rather than replacing missing evidence with synthetic values.

Use an isolated Python environment with the export dependencies:

```sh
python3 -m venv .venv-web-evidence
.venv-web-evidence/bin/python -m pip install -r web/scripts/requirements-export.txt
.venv-web-evidence/bin/python web/scripts/export_research.py
```

The implementation was verified using the pre-existing audit interpreter (Python 3.11.14, NumPy 2.4.1, pandas 3.0.0). The broader repository declares a different Python requirement; this small export workflow is separate from reproducing the research models.

The exporter:

1. Verifies audited input SHA-256 hashes and monthly alignment.
2. Recomputes summary errors and ranking metrics from stored predictions.
3. Checks aggregation comparisons, unknown-future exclusions, regime probability sums and event counts.
4. Exports raw/first-difference monthly views and only the March 2020 daily observations displayed in the detail chart.
5. Copies source artifacts verbatim into `public/evidence/` and writes a hash manifest. It writes only under `web/`.

Do **not** use `analysis/export.py` or `analysis/export_web.py` as authoritative presentation pipelines. Their hard-coded numbers and stale diagnostics are documented in the evidence map. Original JSON exports remain archived in the source tree and are not imported by the rebuilt page.

## Interface and structure

| Location | Purpose |
|---|---|
| `src/App.tsx` | Eight-part narrative, hiring-manager brief, summary/technical views |
| `src/components/FeatureExplorer.tsx` | Searchable feature formulas, exact names, stages and timing |
| `src/components/ExperimentExplorer.tsx` | Model families, historical metrics and SARIMAX case study |
| `src/components/ForecastWorkbench.tsx` | Saved predictions, benchmark comparison and corrected classification |
| `src/charts/ResearchTimeline.tsx` | Separate macro/spread panels with distinct regime/reference overlays |
| `src/charts/ResearchPlot.tsx` | Straight-segment D3 chart with pointer and keyboard inspector support |
| `src/charts/primitives.tsx` | Reused responsive sizing, axes and chart infrastructure |
| `src/lib/research.ts` | Typed evidence schema and source resolution |
| `src/lib/researchContent.ts` | Feature and model records traced to source cells |
| `scripts/export_research.py` | Verified artifact export; no model fitting |
| `public/evidence/manifest.json` | Input/output hashes and source snapshot inventory |

Native selects, search inputs, disclosure controls and sliders support keyboard use. All series keep their own units; target definitions and date axes are explicitly selectable. Responsive layouts preserve scrollable tables, and reduced-motion settings disable smooth movement.

See [VALIDATION.md](VALIDATION.md) for the completed build, evidence-download and browser checks, including their scope and limitations.

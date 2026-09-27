# Implementation validation — 26 September 2026

## Build and evidence

- `npm run typecheck`: passed.
- `npm run build`: passed. Largest JavaScript chunk: 311.08 kB (72.93 kB gzip); application chunk: 266.52 kB (85.79 kB gzip).
- `git diff --check`: passed.
- The presentation exporter verified source hashes, monthly alignment, stored-prediction errors, corrected classification metrics, regime probabilities and stress-entry counts. It did not fit models or execute notebooks.
- All 49 bundled evidence snapshots returned HTTP 200 from the production preview and matched their original SHA-256 hashes. Exporter and generated JSON hashes also matched the manifest.
- The production check caught and corrected Vite's handling of `%2B` in one prediction CSV URL. Source links retain the valid literal `+` in path segments.
- Active components use `research.json`; they do not import the superseded analytical JSON exports.

## Browser checks

The local application was inspected at desktop (1280 × 720) and mobile (390 × 844) sizes. Temporary viewport overrides were reset after testing. No document-level horizontal overflow was observed; wide tables and navigation scroll within their containers.

Verified interactions:

- Macro selection, raw/first-difference views and separate regime/stress overlays; unavailable warmup values remain unavailable.
- Keyboard month and forecast inspectors, including forecast-origin versus realized-target dates.
- Feature search, empty results and formula/source details.
- Model search, research-purpose filters, sequence experiments and SARIMAX disclosures.
- Proposed regime-ablation details correctly reference the newer, unexecuted study design.
- Forecast-window and date-axis selection; displayed evidence dates follow the selected window.
- Legacy versus observed-day benchmark targets, with separately computed RMSE reductions.
- Corrected classification horizons, including the 12-month sample and scores.
- Summary/technical views and the 60-second hiring-manager brief.

The rebuilt production page rendered successfully. No production console warning or error was observed; the tab retained an earlier development-server HMR connection message from port 5177. This was a targeted browser pass, not a claim of exhaustive cross-browser or assistive-technology certification. The embedded browser blocked direct raw-JSON navigation, so evidence delivery was verified independently over HTTP with file hashes.

## Research boundaries

The original research notebooks and evidence remain unchanged. No model fitting, additional validation study, synthetic prediction path or confidence band was added. Matched regime ablation, frozen-threshold event forecasting, dependent-data uncertainty, vintage reconstruction and external/later-period validation remain explicitly pending. See [EVIDENCE_MAP.md](EVIDENCE_MAP.md) for the analytical corrections and artifact lineage.

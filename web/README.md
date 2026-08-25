# Analysis site

Interactive presentation of the findings in [`../ANALYSIS_REPORT.md`](../ANALYSIS_REPORT.md).

```bash
npm install
npm run dev      # http://localhost:5177
npm run build    # -> dist/
```

## How data reaches the page

Nothing is computed in the browser. Every figure is produced by the Python
scripts in [`../analysis`](../analysis) and exported to `src/data/*.json` by:

```bash
~/.venvs/global/bin/python analysis/export_web.py
```

Re-run that after changing any analysis script, then rebuild. Numbers quoted in
the page copy are cross-checked against `../results/key_numbers.csv`.

## Structure

| Path | Contents |
|---|---|
| `src/App.tsx` | Page composition and all narrative copy |
| `src/charts/RegimeTimeline.tsx` | The centrepiece: spread, regime bands and a switchable diagnostic panel |
| `src/charts/*.tsx` | One chart per finding |
| `src/charts/primitives.tsx` | Axes, grid, tooltip, pointer layer, shared palette |
| `src/components/` | Insight block, KPI, disclosure, navigation |
| `src/lib/data.ts` | Typed access to the exported JSON |
| `src/lib/format.ts` | Number and date formatting used by both prose and charts |

## Conventions

- Charts are hand-built on `d3-scale` / `d3-shape` rather than a chart library,
  so annotation and styling stay under direct control.
- Every chart title states the finding, not the variable.
- Interactions exist to answer an analytical question — regime definition,
  forecast horizon, crisis window — not for decoration.
- `Reveal` fails open: if `IntersectionObserver` never fires, a timer shows the
  content anyway, so an entrance animation can never hide the page.

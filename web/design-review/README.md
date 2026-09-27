# Graphite visual redesign

## Product and scope

This is a macroeconomic regime and aggregate high-yield credit-spread research portfolio for credit-risk, banking, fintech, financial data-science and portfolio-analytics hiring managers. It presents stored empirical work and its validity limits. It is not an account-management or wealth product.

The existing single-page structure remains: introduction and hiring-manager brief, followed by Question, Data, Conditions, Experiments, Forecasts, Findings, Validity and Technical Details. The eight navigation anchors, `#top`, `#brief` and `#information` remain intact. No route, section or analytical content was added, removed or reordered.

## Visual changes

- Existing introduction: decorative monochrome alpine lake, masked toward text and faded into the page; responsive local image sources.
- Typography: system Georgia serif for major headings and headline figures, system sans-serif for controls and prose, tabular monospace for analytical tables and inspectors. No remote font dependency.
- Palette: obsidian and graphite backgrounds, silver borders and active states, soft off-white foregrounds; earlier gold and olive styling removed from the UI.
- Existing navigation, reading-mode buttons, source disclosures, feature dictionary, experiment details, benchmark comparison and inspector surfaces restyled with restrained highlights and shadows.
- Plotting areas: plain dark surfaces, subdued grids and muted identifiable series. Existing paths, axis calculations and labels unchanged. Dashed series now also have dashed legend keys. Ten categorical regime colors remain distinct.
- Existing responsive breakpoints retained. Reduced-motion rules disable transitions and smooth scrolling; no parallax or continuous animation was added.

## Preservation and checks

A pre-edit file baseline was compared with the result. All data JSON, research-content records, source resolution and bundled evidence files are byte-for-byte unchanged. Application-source differences are decorative imagery, a chart surface class, colors and legend rendering; calculation and event-handler logic is unchanged. Original notebooks were not executed or edited.

`npm run typecheck`, `npm run build` and `git diff --check` passed. Production preview inspected at 1280 × 720, 768 × 1024 and 390 × 844 with no document overflow. Macro/transformation/overlay selection, keyboard inspectors, forecast date/window selection, target aggregation, classification horizons, feature search, model filtering and summary/technical disclosure controls were checked. Representative text tokens have contrast ratios of at least 5.22:1 against the tested dark surfaces. This is targeted verification, not exhaustive accessibility certification.

Artwork generation prompt and paths: [asset notes](../public/images/README.md). Screenshots in this directory document selected desktop and mobile views.

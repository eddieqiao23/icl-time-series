# MLR results report

This directory contains the single-page HTML report presenting the accepted
results and methodology for the MLR experiment suite. The source is in
`site/`; run `pnpm install && pnpm test` there to build and verify it.

The report includes:

- the packed-task setup and an explanatory methodology figure;
- a short behavioral result section;
- attention and representation evidence;
- causal intervention results;
- OOD algorithm-identification results;
- caveats, model/checkpoint coverage, and links to experiment documentation.

The page must consume curated summary files and accepted figures rather than
reading large raw arrays directly. Every displayed claim should link back to
the experiment README and result manifest that supports it.

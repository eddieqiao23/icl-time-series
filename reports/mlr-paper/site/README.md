# MLR mechanistic report site

Single-page presentation of the accepted mixtures-of-linear-regressions
experiment suite. The page combines the packed-task methodology, behavioral
benchmark, attention and probe evidence, causal interventions, OOD tests, and
explicit coverage limitations.

## Local development

```bash
pnpm install
pnpm dev
```

Open the local URL printed by the development server.

## Validation

```bash
pnpm build
node --test tests/rendered-html.test.mjs
pnpm lint
```

The test checks server-rendered report content and verifies that all seven
curated result figures are present. The scientific claims are sourced from the
CSV summaries and `RESULTS.md` files in `src/experiments/mlr/`.

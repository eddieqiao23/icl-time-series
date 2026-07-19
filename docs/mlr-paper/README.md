# Mixture-of-linear-regressions paper experiments

This directory is the narrative and reproducibility index for experiments on
in-context learning over mixtures of linear regressions (MLR). The project
revisits the behavioral and mechanistic questions previously studied with
autoregressive mixtures using the packed MLR task.

## Canonical setup

- `K`: number of regression components in the prompt-level pool.
- `T`: labeled support pairs inside each task.
- `d`: regression coefficient dimension; initially fixed at 4.
- `N`: tasks per prompt; initially fixed at 50.
- Each task input packs `(x_1, y_1, ..., x_T, y_T, x_query)` into one token.
- The following output token contains the scalar query target.

The initial trained-model grid is `T in {2, 3, 4, 5}`, `K in {2, 3}`, and
`noise_std in {0, 0.2}`. All headline comparisons must record the exact model
run, checkpoint, coefficient-pool seed, sample seed, and evaluation size.

## Research story

The central question is how a trained transformer balances evidence from the
current task with evidence from earlier tasks in the prompt. The experiments
progress from behavior to mechanism:

1. Establish prediction quality and compare with statistically meaningful
   baselines.
2. Measure where attention flows and whether it is component-selective.
3. Test what component and coefficient information is represented.
4. Intervene causally on heads, task tokens, and packed support examples.
5. Use controlled distribution shifts to identify which candidate algorithm,
   if any, best explains transformer behavior.

## Repository layout

Experiment code lives under `src/experiments/mlr/`:

- `common/`: shared model loading, deterministic sampling, indexing,
  statistics, result schemas, and plotting helpers.
- `01_behavior/`: MSE versus task index and baseline comparisons.
- `02_attention/`: layer/head attention and component selectivity.
- `03_probing/`: grouped-CV representation probes and controls.
- `04_causal/`: head ablation, attention masking, and activation patching.
- `05_ood/`: pool geometry, imbalance, hierarchy, and component-count shifts.

The final one-page results report lives under `reports/mlr-paper/` and reads
curated summaries and publication figures from the experiment directories.
The setup comparison is in `AR_VS_MLR_COMPARISON.md`; the independent
leakage/control review and its dispositions are in `INDEPENDENT_AUDIT.md`.

## Artifact policy

Every experiment uses the following output contract:

- `artifacts/raw/`: large reproducible arrays and hidden-state/attention
  tensors. Ignored by Git.
- `artifacts/cache/`: disposable recomputation caches. Ignored by Git.
- `artifacts/summaries/`: compact CSV or JSON results with enough metadata to
  trace the run. Tracked.
- `figures/`: selected, reproducible figures used in review or the report.
  Tracked once the result is accepted.

Scripts must never silently overwrite accepted summaries. A result manifest
must include model IDs, checkpoint steps, configuration, seeds, sample counts,
software version or commit, and the command needed to reproduce it.

## Quality gates

An experiment is ready for the report only when:

1. Its README states the hypothesis, intervention, metrics, and expected
   failure modes.
2. A deterministic smoke run succeeds.
3. Statistical units and error bars are explicit.
4. At least one negative or sanity control is included where applicable.
5. Raw results produce the tracked summary and figures from a clean command.
6. The conclusion distinguishes direct evidence from interpretation.

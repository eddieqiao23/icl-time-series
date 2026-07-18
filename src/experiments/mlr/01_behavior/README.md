# 01 - Behavioral benchmark

## Question

How does prediction error change with task position, support size `T`,
component count `K`, and noise?

## Initial comparisons

- trained transformer;
- known-pool Bayesian predictor;
- causal multi-initialization EM ridge;
- ridge over all previous tasks;
- ridge using only the current task's support examples.

The first deliverable reproduces the existing 500k-checkpoint evaluation with
fixed pools and explicit uncertainty across independent pools and prompts.

## Files

- `inventory_checkpoints.py` discovers local MLR runs, validates their configs
  and weights, and writes a reviewable readiness inventory. `readiness.csv`
  explicitly marks every target condition as complete, partial, or missing.
- `evaluate.py` evaluates the transformer, known-pool Bayesian predictor,
  causal multi-start EM ridge, ridge on the current task, and ridge on task
  history using identical deterministic pools. Results use a long-form CSV
  plus a JSON reproducibility manifest.
- `plot.py` creates the first paper-facing transformer comparison figures.
- `baselines.py` contains experiment-local baseline implementations.

Raw checkpoints remain outside Git. Curated CSV/JSON summaries and accepted
figures are tracked.

## Run

From the repository root:

```bash
python src/experiments/mlr/01_behavior/inventory_checkpoints.py \
  --models-root /path/to/models/mlr

python src/experiments/mlr/01_behavior/evaluate.py \
  --models-root /path/to/models/mlr

python src/experiments/mlr/01_behavior/plot.py
```

For a quick CPU validation, add `--T 2 --K 2 --noise 0 --num-pools 1
--batch-size 8 --methods transformer known_pool` to the evaluation command.

## Checkpoint readiness (2026-07-18)

All 34 discovered MLR run directories have valid MLR configs and readable
weights. Of the 16 conditions in the balanced `T in {2,3,4,5}` by `K in
{2,3}` by `noise in {0,0.2}` grid, 11 have reached 500,000 steps. Five runs
need to be resumed before the final comparison:

| T | K | Noise | Current step | Steps remaining |
|---:|---:|---:|---:|---:|
| 2 | 2 | 0.0 | 417,798 | 82,202 |
| 5 | 2 | 0.0 | 421,086 | 78,914 |
| 2 | 3 | 0.0 | 462,707 | 37,293 |
| 5 | 3 | 0.0 | 381,611 | 118,389 |
| 5 | 3 | 0.2 | 361,887 | 138,113 |

No additional model shapes are needed for this experiment. The partial
checkpoints can support development runs, but the paper-facing grid should use
the common 500,000-step threshold. The machine-readable source of truth is
`artifacts/summaries/readiness.csv`.

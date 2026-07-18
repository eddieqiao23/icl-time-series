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
- `summarize.py` converts per-pool results into per-position and late-context
  means, standard errors, and 95% confidence intervals.
- `RESULTS.md` records the first paper-facing interpretation of the completed
  behavioral grid.
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

Expensive methods can be run separately without losing existing results. For
example, `--methods em_ridge --append` replaces only the EM rows and writes a
separate append-run manifest.

### Runtime and recovery

On the local CPU runtime, the 10-pool, 200-prompt transformer/Bayes/ridge pass
takes roughly 1--2 minutes. Causal five-initialization EM takes about 17
minutes for all 16 conditions (`K=2`: 48--59 seconds per condition; `K=3`:
67--83 seconds). The implementation vectorizes EM over prompts and components,
prints a live estimate, atomically checkpoints every condition, and resumes
completed condition-method cells with `--append`.

For a quick CPU validation, add `--T 2 --K 2 --noise 0 --num-pools 1
--batch-size 8 --methods transformer known_pool` to the evaluation command.

## Checkpoint readiness (2026-07-18)

All 34 discovered MLR run directories have valid MLR configs and readable
weights. All 16 conditions in the balanced `T in {2,3,4,5}` by `K in {2,3}`
by `noise in {0,0.2}` grid are ready. Eleven reached 500,000 steps and five
completed through early stopping:

| T | K | Noise | Final step | Completion |
|---:|---:|---:|---:|---|
| 2 | 2 | 0.0 | 417,798 | Early stopping |
| 5 | 2 | 0.0 | 421,086 | Early stopping |
| 2 | 3 | 0.0 | 462,707 | Early stopping |
| 5 | 3 | 0.0 | 381,611 | Early stopping |
| 5 | 3 | 0.2 | 361,887 | Early stopping |

No additional training or model shapes are needed for this experiment. The
explicit early-stopping declarations live in `config/completion_overrides.csv`;
the machine-readable resolved status is `artifacts/summaries/readiness.csv`.

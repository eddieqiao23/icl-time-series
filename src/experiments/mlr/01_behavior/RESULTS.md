# Behavioral results

## Evaluation

Each condition uses 10 deterministic coefficient pools and 200 prompts per
pool. Curves report MSE at each of 50 task positions; uncertainty is computed
across pools. The headline statistic below averages positions 40--49 within
each pool and then reports the across-pool mean. Checkpoints are the final
training states, including the five declared early-stopping completions.

## Late-context MSE

| K | Noise | T | Transformer | Known pool | EM ridge |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0 | 2 | 0.0107 | 0.0000 | 0.0000 |
| 2 | 0.0 | 3 | 0.0044 | 0.0000 | 0.0000 |
| 2 | 0.0 | 4 | 0.0027 | 0.0000 | 0.0000 |
| 2 | 0.0 | 5 | 0.0012 | 0.0000 | 0.0000 |
| 2 | 0.2 | 2 | 0.0879 | 0.0640 | 0.0709 |
| 2 | 0.2 | 3 | 0.0570 | 0.0469 | 0.0498 |
| 2 | 0.2 | 4 | 0.0525 | 0.0440 | 0.0459 |
| 2 | 0.2 | 5 | 0.0452 | 0.0415 | 0.0433 |
| 3 | 0.0 | 2 | 0.0498 | 0.0000 | 0.0002 |
| 3 | 0.0 | 3 | 0.0059 | 0.0000 | 0.0001 |
| 3 | 0.0 | 4 | 0.0058 | 0.0000 | 0.0000 |
| 3 | 0.0 | 5 | 0.0024 | 0.0000 | 0.0000 |
| 3 | 0.2 | 2 | 0.1764 | 0.0913 | 0.1071 |
| 3 | 0.2 | 3 | 0.0949 | 0.0523 | 0.0575 |
| 3 | 0.2 | 4 | 0.0572 | 0.0454 | 0.0494 |
| 3 | 0.2 | 5 | 0.0528 | 0.0427 | 0.0452 |

Exact standard errors and 95% confidence intervals are in
`artifacts/summaries/headline_summary.csv`.

## Story

1. The transformer uses cross-task context: its error falls with task position,
   unlike current-task ridge and the known-pool predictor, whose information is
   local to the current task.
2. The benefit of larger support is strong. Increasing `T` moves the
   transformer toward the known-pool and learned-pool EM limits, especially for
   noisy `K=3`, where late-context MSE falls from 0.1764 at `T=2` to 0.0528 at
   `T=5`.
3. Mixture complexity matters most when the current task is weakly identified.
   The largest transformer-oracle gap is at `K=3, T=2`; it narrows sharply as
   `T` increases.
4. History ridge is not a suitable mixture learner. Pooling prior tasks into a
   single regressor stays far above the transformer because it averages
   incompatible components. EM ridge instead learns the component pool and
   converges close to the known-pool predictor.

## Figures

- `figures/transformer_mse_K{2,3}_{noiseless,noisy}.png` isolates scaling with
  support size.
- `figures/method_comparison_K{2,3}_{noiseless,noisy}.png` compares all five
  predictors at every task position.

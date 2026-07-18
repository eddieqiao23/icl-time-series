# Causal intervention results

## Design

Interventions use the canonical noisy `T=3, K=2` checkpoint, 10 independent
coefficient pools, and 64 prompts per pool. Task/support replacements use
batch-mean activations or values; head ablation uses the conventional zero head
mask. Error changes are measured against the same clean prompts.

## Context necessity

| Task position | Replace same-component history | Replace different-component history | Replace all history |
|---:|---:|---:|---:|
| 10 | +0.559 | -0.034 | +0.405 |
| 20 | +0.675 | -0.019 | +0.460 |
| 30 | +0.782 | -0.003 | +0.469 |
| 40 | +0.777 | -0.003 | +0.479 |
| 49 | +0.819 | -0.012 | +0.374 |

Values are mean changes in MSE across pools. Removing matched history is highly
damaging; removing mismatched history has no cost and slightly improves mean
error. The surprising fact that replacing all history is less damaging than
replacing only matched history is consistent with mismatched context being a
distractor and with mean replacement changing the normalization context. It
should not be interpreted as an additive decomposition.

## Current-task support

Replacing one of the three support blocks adds `0.038--0.042` MSE. Replacing
all support blocks adds `0.640` MSE. The model therefore uses both current-task
evidence and matched cross-task evidence.

## Heads and residual stages

The largest single-head MSE increases are layer/head `1/3` (`+0.030`), `4/3`
(`+0.024`), `4/0` (`+0.020`), and `2/0` (`+0.018`). No single head accounts for
the full matched-context effect, indicating a distributed circuit.

Patching the clean prediction-token residual into the matched-context-corrupt
run recovers `0.00, 0.02, 0.30, 0.30, 0.34, 0.90, 1.00` of the clean--corrupt
gap at stages 0--6. Most recoverable task information becomes consolidated
between stages 4 and 5, aligning with the late-layer attention selectivity and
high probe scores.

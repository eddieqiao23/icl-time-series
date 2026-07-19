# Causal intervention results

## Design

Interventions use the canonical noisy `T=3, K=2` checkpoint, 10 independent
coefficient pools, and 64 recipient prompts per pool. For each pool, a second
independently seeded donor batch supplies on-distribution task-token pairs and
support blocks. Selected earlier input/output token pairs are replaced only by
donor pairs from the same task position; no replacement value is derived from
the recipient sequence. A random-task condition replaces the same number of
tasks as the same-component condition.

## Historical-context contribution

| Task position | Same component | Different component | Random, count matched | All history |
|---:|---:|---:|---:|---:|
| 10 | +0.061 | -0.030 | +0.017 | -0.008 |
| 20 | +0.025 | -0.016 | -0.000 | -0.008 |
| 30 | +0.032 | -0.001 | +0.003 | +0.005 |
| 40 | +0.017 | +0.000 | -0.003 | -0.004 |
| 49 | +0.012 | +0.000 | -0.006 | -0.004 |

Values are mean changes in MSE across pools. Same-component donor replacement
is more damaging than both different-component and count-matched random
replacement at every tested position. Paired pool-level t tests give
`p=0.0004--0.021` versus different-component replacement and
`p=0.0009--0.015` versus the count-matched random control.

The clean result is much smaller than the earlier recipient-wide mean
replacement effect. That earlier `+0.819` final-position result used means that
included future recipient positions and is rejected. The accepted conclusion
is modest but selective causal use of matched history, not wholesale
dependence on it.

## Current-task support

Replacing one of the three current-task support blocks with an independent
donor block adds `0.253--0.274` MSE. Replacing all support blocks adds `0.906`
MSE. Current-task evidence is therefore the dominant causal input in this
checkpoint; matched history provides a smaller refinement.

## Heads and residual stages

The largest single-head MSE increases remain layer/head `1/3` (`+0.030`),
`4/3` (`+0.024`), `4/0` (`+0.020`), and `2/0` (`+0.018`). No individual head
accounts for the full behavior.

Patching the clean query residual into the donor-corrupted run gives recovery
fractions `0.00, 0.43, 1.24, 1.27, -0.23, 0.32, 1.00` at stages 0--6. Because
the accepted corruption gap is small, intermediate recovery is unstable and
can overshoot; it does not cleanly localize the computation. Stage 6 is a
by-construction positive control because it overwrites the final normalized
query representation. These patching results are retained as a diagnostic,
not paper-level localization evidence.

# OOD algorithm-identification results

## Design

The noisy `T=3, K=2` transformer is held fixed. Every condition uses 10
independent pools and 128 prompts per pool; methods are compared on the same
forced final task. EM is fit only to completed history and evaluated with the
specified candidate component count.

## Coefficient geometry

Across cosine similarities from `-1` to identical components, transformer MSE
stays between `0.048` and `0.075`; known-pool and correctly specified EM remain
between `0.042` and `0.061`. Geometry alone causes a modest gap relative to the
more severe frequency and component-count shifts below.

## Forced minority task

| Majority probability | Transformer | Known pool | EM K=2 | Current-task ridge |
|---:|---:|---:|---:|---:|
| 0.50 | 0.065 | 0.051 | 0.053 | 0.490 |
| 0.80 | 0.086 | 0.048 | 0.059 | 0.432 |
| 0.95 | 0.324 | 0.041 | 0.506 | 0.426 |
| 0.99 | 0.966 | 0.047 | 1.975 | 0.461 |

The transformer and EM both over-weight the dominant historical component,
but the transformer is substantially more robust than EM at extreme skew. Its
error crosses toward the target-only ridge regime rather than following a
single fixed algorithm.

## Component-count mismatch

| Evaluation K | Transformer | Known pool | Correct-K EM | EM fixed at K=2 |
|---:|---:|---:|---:|---:|
| 1 | 0.047 | 0.041 | 0.042 | 0.043 |
| 2 | 0.060 | 0.048 | 0.051 | 0.051 |
| 3 | 0.157 | 0.052 | 0.054 | 0.263 |
| 4 | 0.251 | 0.053 | 0.067 | 0.525 |

Correctly specified EM adapts to unseen component counts. The transformer
degrades for `K>2` but outperforms EM constrained to its training count,
suggesting partial capacity to represent extra components rather than a hard
two-component algorithm.

## Hierarchical pools

For three-component pools, transformer MSE rises from `0.047` at dispersion
`0.05` to `0.164` at dispersion `1.0`; correct-K EM remains near `0.044--0.075`.
Near-identical components are easy because component confusion has little
prediction cost. As components separate, the unseen `K=3` structure dominates.

## Conclusion

The transformer resembles learned-pool EM in-distribution, but no single fixed
candidate explains all shifts. Under extreme imbalance it is more robust than
EM, while under component-count expansion it lies between fixed-`K` and
correctly specified EM. The evidence favors an adaptive combination of
historical mixture inference and current-task regression.

# OOD algorithm-identification results

## Design

The noisy `T=3, K=2` transformer is held fixed. Every condition uses 10
independent pools and 128 prompts per pool; sweep points reuse base pool
orientations, assignments, and prompt noise for paired comparisons. Methods
are scored on the same forced final task.

The assignment oracle knows the final component ID and is the protocol-aware
noise-floor reference. Known-pool Bayes knows the coefficient vectors but uses
a fixed uniform component prior. EM-ridge estimates both component coefficients
and mixture weights from completed history, then uses current-task supports to
infer the final component. Current-task ridge uses the prior-matched penalty
`noise² × d`.

## Coefficient geometry

Across cosine similarities from `-1` to identical components, transformer MSE
stays between `0.043` and `0.058`; the assignment oracle is `0.039`, and
known-pool/EM references stay between `0.039` and `0.052`. Geometry alone causes
a modest gap relative to the frequency and component-count shifts below.

## Forced minority task

| Majority probability | Transformer | Assignment oracle | Known pool, uniform prior | EM K=2 | Current-task ridge |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.050 | 0.039 | 0.047 | 0.050 | 0.397 |
| 0.80 | 0.087 | 0.039 | 0.047 | 0.073 | 0.397 |
| 0.95 | 0.372 | 0.039 | 0.047 | 0.429 | 0.397 |
| 0.99 | 0.966 | 0.039 | 0.047 | 1.678 | 0.397 |

At extreme skew the transformer is more robust than history-fitted EM, but it
remains far above both the protocol-aware assignment oracle and the
known-coefficient-pool reference. Its error crosses the underdetermined
current-task ridge baseline near 95% majority probability, consistent with a
shift away from historical inference when matched history becomes scarce.

## Component-count mismatch

| Evaluation K | Transformer | Assignment oracle | Known pool | Correct-K EM | EM fixed at K=2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.046 | 0.041 | 0.041 | 0.042 | 0.042 |
| 2 | 0.065 | 0.041 | 0.048 | 0.049 | 0.049 |
| 3 | 0.151 | 0.041 | 0.057 | 0.059 | 0.260 |
| 4 | 0.321 | 0.041 | 0.055 | 0.065 | 0.590 |

Correctly specified EM adapts to unseen component counts. The transformer
degrades for `K>2` but outperforms EM constrained to its training count,
suggesting partial capacity for extra components rather than a hard
two-component algorithm.

## Hierarchical pools

For three-component pools, transformer MSE rises from `0.050` at dispersion
`0.05` to `0.158` at dispersion `1.0`; correct-K EM ranges from `0.048` to
`0.053` at the endpoints. Near-identical components are easy because component
confusion has little prediction cost. As components separate, the unseen
`K=3` structure dominates.

## Conclusion

No single fixed candidate explains all shifts. The transformer is competitive
with mixture inference in distribution, abandons historical evidence more
gracefully than fitted EM under extreme imbalance, and lies between fixed-`K`
and correctly specified EM under component-count expansion. This supports an
adaptive, capacity-limited combination of historical mixture inference and
current-task regression—not exact EM.

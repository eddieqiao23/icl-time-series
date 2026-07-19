# 03 - Representation probing

## Question

What information about the active regression component and coefficient vector
is decodable from the residual stream?

## Required methodology

- splits grouped by prompt;
- shuffled-label and untrained-model controls;
- fixed evaluation pools for component-ID classification;
- direct regression onto `beta` or permutation-invariant same/different targets
  when pools vary;
- accuracy or regression quality reported by layer and task position.

Probe accuracy alone does not establish that the representation is used by the
model. Experiment 04 tests causal relevance.

## Implementation

`evaluate.py` extracts the embedding and every transformer-block output at
prediction-bearing task tokens. Multi-output ridge probes decode the active
coefficient and logistic probes decode component identity. Four-fold splits are
grouped by prompt, and the same pipeline is repeated for shuffled labels,
untrained weights, and the raw packed input. Scores remain pool-level before
aggregation.

`evaluate_t_sweep.py` runs the actual-label coefficient probe for the trained
`T = 2, 3, 4, 5` checkpoints under the same pools, positions, and prompt-grouped
cross-validation. It intentionally omits untrained, shuffled-label, and raw
input series so `plot_t_sweep.py` can compare the trained models directly.

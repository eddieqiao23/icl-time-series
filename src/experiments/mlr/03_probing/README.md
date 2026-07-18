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

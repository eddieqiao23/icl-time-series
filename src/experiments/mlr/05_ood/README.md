# 05 - OOD algorithm identification

## Question

Does transformer behavior track Bayesian component inference, online EM,
pooled regression, target-only regression, or an adaptive combination across
controlled distribution shifts?

## Initial shifts

- coefficient cosine similarity;
- identical, near-identical, orthogonal, and hierarchical pools;
- balanced and highly unbalanced mixture frequencies;
- forced minority-component query tasks;
- component-count mismatch at evaluation time;
- EM fitted with the wrong number of components.

All comparisons hold the trained transformer fixed and evaluate candidate
algorithms on identical prompts.

## Implemented sweeps

- exact coefficient cosine similarity from `-1` through identical components;
- majority probabilities from `0.5` to `0.99` with a forced minority final task;
- evaluation pools with `K=1,2,3,4`, including EM fitted with every candidate K;
- three-component hierarchical pools from near-identical to dispersed.

Each condition uses 10 independent pools and 128 prompts per pool. Candidate
algorithms are scored on the same final task, and EM is fitted only once to the
completed history, avoiding the unnecessary full-position refits used by the
behavioral curve experiment.

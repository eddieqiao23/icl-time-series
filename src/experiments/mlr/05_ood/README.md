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

# 04 - Causal interventions

## Question

Which heads, task tokens, and packed support examples are necessary for the
model's predictions and component information?

## Initial interventions

- mean head ablation with repeated evaluation seeds;
- attention-logit masking of matched and mismatched context tasks;
- clean-to-corrupted activation patching;
- matched-component and mismatched-component replacement;
- masking or patching individual packed `(x_t, y_t)` support blocks.

Zero ablation is retained only as a labeled stress test because it can create
strong out-of-distribution artifacts.

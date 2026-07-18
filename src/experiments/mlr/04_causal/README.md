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

## Implemented interventions

- per-head masking at every layer, repeated over coefficient pools;
- mean replacement of matched, mismatched, or all prior task tokens;
- mean replacement of each packed support block or every support block;
- clean-to-corrupted residual patching at the prediction token by layer.

Mean replacement is used for task and support tokens to avoid interpreting a
large zero-vector distribution shift as evidence of necessity. Head masking is
the conventional zero-mask intervention and is labeled separately.

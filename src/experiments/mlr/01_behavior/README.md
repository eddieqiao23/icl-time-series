# 01 - Behavioral benchmark

## Question

How does prediction error change with task position, support size `T`,
component count `K`, and noise?

## Initial comparisons

- trained transformer;
- known-pool Bayesian predictor;
- causal multi-initialization EM ridge;
- ridge over all previous tasks;
- ridge using only the current task's support examples.

The first deliverable reproduces the existing 500k-checkpoint evaluation with
fixed pools and explicit uncertainty across independent pools and prompts.

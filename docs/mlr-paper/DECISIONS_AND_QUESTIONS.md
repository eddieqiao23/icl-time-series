# Decisions and questions log

This file records choices made while running the experiment suite autonomously.
None of these items blocked execution.

## Decisions made

### Canonical mechanistic checkpoint

Attention, probing, and causal experiments use the final
`T=3, K=2, N=50, noise_std=0.2` checkpoint. The full behavioral grid already
tests all 16 combinations; using one checkpoint for mechanistic analyses avoids
mixing mechanisms across input widths and provides a single architecture-
matched untrained control. OOD experiments hold this checkpoint fixed while
changing pool geometry, frequencies, and component count.

### Statistical unit

Independent coefficient pools are the primary uncertainty unit. Prompts are
sampled independently within pools; summaries retain pool-level results so
confidence intervals do not treat all task positions as independent.

### Runtime budget

Runs are designed to complete in minutes to low tens of minutes on the local
CPU. Large tensors are reduced batch-by-batch, and long-running scripts write
atomic per-condition checkpoints with timing estimates.

## Questions for later review

1. Should the final paper include a second mechanistic checkpoint (`K=3`) as a
   replication, or is the full behavioral/OOD coverage sufficient?
2. For attention figures, should head identities be discussed individually or
   only as layerwise distributions unless causal ablation corroborates them?
3. Should the report emphasize the noisy setting as the main result and move
   noiseless results to supplementary material?

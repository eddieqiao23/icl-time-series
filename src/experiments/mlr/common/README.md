# Shared experiment infrastructure

This folder will provide:

- validated model/checkpoint discovery;
- deterministic MLR pool and prompt generation;
- task-to-token indexing helpers;
- common baselines and prediction extraction;
- grouped statistical resampling and confidence intervals;
- versioned result manifests;
- shared plotting defaults and figure metadata.

The common layer must not contain experiment-specific conclusions or plotting
layouts.

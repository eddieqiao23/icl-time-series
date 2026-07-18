# MLR experiment suite

This package contains the paper-facing experiment suite for the packed
mixture-of-linear-regressions task. Experiments are numbered in narrative order
and share infrastructure from `common/`.

Each experiment folder owns its runner, analysis code, README, compact
summaries, and accepted figures. Large raw tensors remain local under
`artifacts/raw/` and are reproducible from the recorded manifest.

Experiments must use the shared task/token terminology:

- task index `i` maps to transformer input-token position `2*i`;
- the observed output for task `i` maps to position `2*i + 1`;
- `K` is the component count, `T` is support examples per task, `N` is tasks
  per prompt, and `d` is regression dimension.

Do not copy model-loading, sampler-construction, or bootstrap/error-bar logic
between experiments. Add it to `common/` with a focused test instead.

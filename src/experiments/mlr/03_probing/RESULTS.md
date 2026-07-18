# Probing results

## Design

The canonical noisy `T=3, K=2` model was evaluated on 10 coefficient pools,
128 prompts per pool, and six task positions. Probes use four-fold splits
grouped by prompt. Representation stage 0 is the input embedding; stages 1--6
are successive transformer-block outputs. The statistical unit is the pool.

## Late-position results

Late positions are `30, 40, 49`, averaged after pool-level scoring.

| Stage | Trained coefficient R² | Untrained R² | Trained component accuracy | Untrained accuracy |
|---:|---:|---:|---:|---:|
| 0 | -0.073 | -0.074 | 0.489 | 0.484 |
| 1 | 0.549 | -0.050 | 0.874 | 0.493 |
| 2 | 0.602 | -0.026 | 0.891 | 0.500 |
| 3 | 0.753 | -0.010 | 0.958 | 0.507 |
| 4 | 0.848 | 0.006 | 0.967 | 0.518 |
| 5 | 0.861 | 0.029 | 0.968 | 0.525 |
| 6 | 0.840 | 0.043 | 0.967 | 0.533 |

The packed raw input has coefficient R² `-0.063` and component accuracy
`0.486` under the same linear probes. Shuffled-label probes remain at or below
their null levels.

## Interpretation

Training creates a linearly accessible representation of the active
coefficient immediately after the first transformer block. Decodability grows
through stage 5 and remains high at the final stage. The raw input and
untrained controls show that this is not a trivial linear consequence of token
packing. Probe performance establishes information availability; the causal
experiments test whether the model uses it.

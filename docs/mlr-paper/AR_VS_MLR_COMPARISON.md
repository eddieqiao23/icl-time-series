# Previous AR setup versus packed MLR setup

## Comparison boundary

The two suites test the same high-level question—whether a transformer infers
and uses a latent regression component—but they do not expose the model to the
same information. The AR task gives each target run several within-run time
series observations. The packed MLR task gives a query `x` without its `y`, so
the query must borrow evidence from earlier component-matched tasks. Absolute
MSE values are therefore context, not a controlled head-to-head benchmark.

The previous results referenced below are the accepted AR documents at commit
`27c5af8`: `docs/final_run_attention_analysis.md`,
`docs/probe_methodology_review_2026-05-07.md`, and the READMEs for experiments
01, 07, 09, and 10.

## Result-level comparison

| Question | Previous AR-mixture setup | New packed MLR setup | What changed |
|---|---|---|---|
| Predictive behavior | Transformer MSE was 0.047–0.057 across the documented `m=8,10,13` models and was competitive with EM/ridge baselines. | In noisy `K=3`, late-position MSE falls from 0.176 (`T=2`) to 0.053 (`T=5`), versus 0.043 for the fixed-uniform-prior known-pool reference. | Both models learn useful mixture structure. The MLR sweep shows the value of adding support pairs inside each current task; `T` does not vary cross-task context length. |
| Attention depth | Selectivity was strongest in early layers (especially layers 1 and 4 in the position test); late attention was nearly uniform. | Same-component selectivity grows late and reaches 0.496 in layer 5; an untrained control remains near zero. | The routing profile moves later. Packed MLR appears to first form a component/rule representation and then retrieve matching tasks. |
| Decodable state | Binary component accuracy jumped to 0.78–0.87 after layer 1, plateaued, then dipped slightly at layer 6. The review warned that a run’s local `(x,y)` evidence could make identity trivially decodable. | Late-position component accuracy grows from 0.874 at stage 1 to 0.968 at stage 5; coefficient-vector R² grows from 0.549 to 0.861. Raw-linear and untrained controls are null. | MLR directly decodes the regression vector, but its packed query token also contains current-task `(x,y)` supports. The probe does not isolate local computation from historical retrieval. |
| Causal component use | Mean activation ablation produced effects around 0.001 and own/cross-component effects were similar; zero ablation was diagnosed as an OOD artifact. | Independent donor replacement gives a modest final matched-history effect (+0.012 versus −0.006 count-matched random), while replacing all current supports adds 0.906 MSE. | Both setups emphasize local/current evidence. MLR additionally shows a statistically consistent but smaller component-selective historical contribution. |
| Extreme imbalance | At 99/1 skew, the AR transformer stayed close to its self-fit/oracle solution (about 0.055 MSE) even with no minority context, because the target run contained its own AR observations. | With 99% majority context and a forced minority query, transformer MSE rises to 0.966; the assignment oracle remains 0.039. | The difference follows the information boundary. Three MLR supports are insufficient to self-fit an underdetermined four-dimensional rule, so scarce matching history matters. |
| More components | AR hard-EM with four components could underperform two-component EM at 50 runs because support fragmentation dominated. | The transformer partially extrapolates to `K=3,4` but at `K=4` reaches 0.321 MSE versus 0.065 for correctly specified four-component EM. | Both suites show a finite-support/capacity tradeoff rather than exact mixture inference. |

## Story that survives the setup change

1. The transformer is not equivalent to a single pooled ridge regression.
2. It learns component-sensitive attention rather than inheriting that pattern
   from the architecture.
3. Its inference rule is adaptive: behavior changes with support, geometry,
   and imbalance.
4. Exact EM remains the better description only when correctly specified and
   sufficiently supported.

## Story that changes

The AR setup supported a hybrid strategy: use component-matched history when
available, but self-fit from a relatively informative target run when history
is unhelpful. MLR still provides current-task support, but `T=3` observations
cannot fully identify a four-dimensional coefficient. Consequently, current
support remains dominant while selective historical routing supplies a modest
refinement; extreme minority queries become the clearest failure case.

The new results therefore sharpen the paper claim. They support
**latent-rule inference followed by component-selective retrieval**, not a
generic claim that transformers always implement EM. The OOD results further
qualify this as an amortized, capacity-limited procedure.

## Methodological differences

- The MLR suite uses one shared condition generator and explicit packed-token
  mapping across all experiment families.
- Uncertainty is computed over independent coefficient pools, not individual
  prompts or token positions.
- Probes use prompt-grouped cross-validation with shuffled, raw-input, and
  architecture-matched untrained controls.
- Causal conclusions rely on independent same-position donor replacement with
  a count-matched random control; recipient-wide mean and zero ablations are
  rejected as contaminated or off-manifold.
- Every accepted run has a command, environment, seed policy, checkpoint
  record, summary CSV, and narrative result file.

## Remaining apples-to-apples experiment

No new model training is required for the completed paper suite. The cleanest
optional controlled comparison would evaluate both input formats with matched
dimension, component pool, number of support observations, and intervention
target. That would require either a new AR checkpoint with the MLR coefficient
distribution or a new MLR checkpoint that exposes within-query observations;
it is a follow-up, not a prerequisite for the current conclusions.

# Independent experiment audit

## Scope and outcome

An independent agent audited all five MLR experiment families for train/eval
contamination, label leakage, noncausal baselines, invalid controls,
pseudoreplication, duplicate rows, metric/report mismatches, and unsupported
interpretation. It found **no deliberate cheating, train/evaluation pool
contamination, duplicate primary rows, or prompt leakage across probe folds**.

The audit did identify several serious analysis and reporting issues. All P1
items were either corrected and rerun or explicitly demoted below.

## Findings and disposition

| Finding | Severity | Disposition |
|---|---:|---|
| The report treated `T` as prompt depth, but `T` is support pairs inside each task. | P1 | Corrected throughout the HTML report and AR comparison. `N=50` is task/context length. |
| Recipient-wide causal means included current/future positions. | P1 | Rejected the old `+0.819` result. Reran with independently seeded same-position donor task-token pairs and added a count-matched random control. |
| The probe still sees current-task support pairs. | P1 | Retained the valid decodability result but removed infer-then-route/localization claims. This confound remains an explicit limitation. |
| OOD “EM” did not estimate mixture weights; “oracle” labels were overstated. | P1 | EM now estimates weights from completed history. Added a protocol-aware assignment oracle and relabeled fixed-uniform-prior known-pool Bayes. Reran all paired OOD sweeps. |
| Current-task ridge used an untuned `λ=1`. | P1 | Uses prior-matched `λ = noise² × d` (with a numerical floor); behavioral and OOD rows rerun. |
| Figure 1 called ±1 SE bands 95% intervals. | P1 | Caption corrected to one standard error. |
| 95% intervals used normal rather than t critical values for 10 pools. | P2 | Summary tables now use Student-t critical values. |
| Probe late-position error bars ignored within-pool covariance. | P2 | Recomputed each late-position mean within pool, then SE across pools. |
| OOD sweep points used unrelated pools. | P2 | Reused base orientations, assignments, and prompt noise within each sweep family. |
| Stage-6 patch recovery is one by construction. | P2 | Labeled as a positive control; unstable activation-patching localization is removed from the main report. |
| Attention cosine analysis was incidental rather than controlled. | P2 | README corrected; the controlled geometry sweep is only claimed for OOD behavior. |
| Initial manifests predated the code commit. | P2 | Mechanistic families were rerun from committed code so accepted manifests identify a commit containing their runners. |

## Accepted causal conclusion after repair

Independent donor replacement reduces the historical-context effect from the
rejected `+0.819` result to a modest selective contribution. At task position
49, same-component replacement adds `0.012` MSE versus `-0.006` for a random
count-matched replacement (`p=0.014`, paired across pools). Replacing all three
current-task supports adds `0.906` MSE. The accepted interpretation is therefore
**current-task inference with a smaller component-selective historical
refinement**, not dominant historical routing.

## Remaining limitations

1. A probe on support-masked or support-swapped query tokens is still needed to
   isolate historically retrieved coefficient information from nonlinear local
   regression.
2. The repaired donor corruption produces a small activation-patching gap, so
   intermediate recovery overshoots and is not a reliable localization assay.
3. Attention selectivity and head ablation remain converging but not uniquely
   identifying evidence; no individual head is necessary for the full effect.

These limitations do not invalidate the behavioral, attention, probe, or OOD
measurements. They narrow the mechanistic claim and define the cleanest next
experiments.

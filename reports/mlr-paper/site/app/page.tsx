import type { Metadata } from "next";
import TokenExplorer from "./components/TokenExplorer";

export const metadata: Metadata = {
  title: "How Transformers Solve Mixtures of Linear Regressions",
  description: "Behavioral, mechanistic, causal, and out-of-distribution evidence from the MLR experiment suite.",
};

const links = {
  behavior: "https://github.com/eddieqiao23/icl-time-series/blob/main/src/experiments/mlr/01_behavior/RESULTS.md",
  attention: "https://github.com/eddieqiao23/icl-time-series/blob/main/src/experiments/mlr/02_attention/RESULTS.md",
  probing: "https://github.com/eddieqiao23/icl-time-series/blob/main/src/experiments/mlr/03_probing/RESULTS.md",
  causal: "https://github.com/eddieqiao23/icl-time-series/blob/main/src/experiments/mlr/04_causal/RESULTS.md",
  ood: "https://github.com/eddieqiao23/icl-time-series/blob/main/src/experiments/mlr/05_ood/RESULTS.md",
};

function Figure({ src, alt, caption }: { src: string; alt: string; caption: string }) {
  // Result figures are pre-rendered scientific artifacts, so preserving their exact pixels is intentional.
  // eslint-disable-next-line @next/next/no-img-element
  return <figure><img src={src} alt={alt} loading="lazy" /><figcaption>{caption}</figcaption></figure>;
}

export default function Home() {
  return (
    <main>
      <nav aria-label="Report sections">
        <a className="brand" href="#top">MLR / 2026</a>
        <div className="nav-links"><a href="#evidence">Evidence</a><a href="#causal">Causal tests</a><a href="#limits">Limits</a></div>
      </nav>

      <header id="top" className="hero">
        <p className="eyebrow">Mechanistic evaluation · 2 components · 3 supports/task · 6 layers</p>
        <h1>The model learns to identify a hidden task—and route the right examples to it.</h1>
        <p className="lede">A transformer trained on packed mixtures of linear regressions behaves like an adaptive inference procedure. It separates latent components, concentrates attention on relevant examples, and builds a decodable estimate of the active regression. Intervention and shift tests show both where this account is causal and where it is incomplete.</p>
        <div className="hero-stats" aria-label="Key results">
          <div><span>0.496</span><p>final-layer attention selectivity</p></div>
          <div><span>0.861</span><p>linear-probe β R² at stage 5</p></div>
          <div><span>0.906</span><p>MSE increase after replacing all current-task supports</p></div>
        </div>
        <p className="scope">Canonical mechanistic checkpoint: T = 3 support pairs per task, K = 2 components, N = 50 tasks per prompt, noise σ = 0.2.</p>
      </header>

      <section className="setup" aria-labelledby="setup-title">
        <div>
          <p className="section-number">01 / Setup</p><h2 id="setup-title">One sequence, several hidden regressions</h2>
          <p>Each prompt is a sequence of tasks sampled from latent linear models. Every task input packs T labeled support pairs and a held-out query <em>x</em>; its following output token contains the query label. Solving the task requires combining current-task support with matching historical tasks rather than averaging indiscriminately.</p>
        </div>
        <TokenExplorer />
      </section>

      <section id="evidence" className="section-block" aria-labelledby="evidence-title">
        <p className="section-number">02 / Converging evidence</p><h2 id="evidence-title">The same computation appears in behavior, attention, and state.</h2>
        <article className="result-row">
            <div className="result-copy"><p className="result-label">Behavior</p><h3>More within-task support closes most of the gap to the known-pool reference.</h3><p>In the noisy three-component condition, transformer MSE falls from <strong>0.176</strong> with two support pairs per task to <strong>0.053</strong> with five. The known-coefficient-pool Bayesian reference reaches 0.043. The sweep shows that the model uses the extra evidence packed inside each current task.</p><a href={links.behavior}>Behavior results →</a></div>
          <Figure src="/results/behavior-k3-noisy.png" alt="Mean squared error by support-pair count for five regression methods in the noisy three-component setting" caption="Figure 1. Performance by within-task support size in the hardest in-distribution condition. Bands show one standard error across pools." />
        </article>
        <article className="result-row reverse">
          <div className="result-copy"><p className="result-label">Attention</p><h3>Later layers preferentially retrieve the query’s component.</h3><p>Same-component attention selectivity rises sharply through the network and reaches <strong>0.496</strong> in layer 5. An architecture-matched untrained control stays near zero. The learned attention pattern is therefore aligned with the latent partition the task demands.</p><a href={links.attention}>Attention results →</a></div>
          <Figure src="/results/attention-by-layer.png" alt="Same-component attention selectivity by transformer layer for trained and untrained models" caption="Figure 2. Query attention to same- versus different-component examples, aggregated across 10 independent prompt pools." />
        </article>
        <div className="attention-detail">
          <div><p className="result-label">Final-token profile</p><h3>The last query develops both component selectivity and a recency gradient.</h3><p>The final input token is position 98. In layer 5 it assigns 0.0297 average task-level attention to a prior task when that task shares the active component, versus 0.0102 when it does not. Attention also rises from 0.0112 across tasks 0–9 to 0.0251 across tasks 40–48, so position and component identity jointly shape retrieval. The plots show softmax attention probabilities, not causal effects.</p></div>
          <div className="figure-grid">
            <Figure src="/results/final-query-attention-by-task.png" alt="Attention from the final query token to each prior task and itself, faceted by layer for trained and untrained models" caption="Figure 2a. Absolute attention from the final query input to each task. The dashed line marks self-attention at task 49; bands show one standard error across coefficient pools." />
            <Figure src="/results/final-query-attention-by-relation.png" alt="Conditional attention from the final query token to same-component and different-component prior tasks, faceted by layer" caption="Figure 2b. Conditional task-level attention reveals learned component routing in late layers while preserving a positional gradient." />
          </div>
        </div>
        <article className="result-row">
            <div className="result-copy"><p className="result-label">Representation</p><h3>A linear readout recovers the active regression.</h3><p>A held-out linear probe’s β R² climbs from below zero at the embedding to <strong>0.861</strong> at stage 5; component classification reaches <strong>96.8%</strong>. Raw-linear and untrained controls are null, while prompt-grouped cross-validation prevents fold leakage. Because the packed query contains current-task support pairs, this establishes information availability—not whether it was computed locally or retrieved from history.</p><a href={links.probing}>Probe results →</a></div>
          <Figure src="/results/probe-beta.png" alt="Linear-probe beta coefficient R squared by network stage for trained and control representations" caption="Figure 3. Decodability of the active regression coefficients across the residual stream." />
        </article>
      </section>

      <section id="causal" className="causal-section" aria-labelledby="causal-title">
        <div className="section-intro"><p className="section-number">03 / Causal tests</p><h2 id="causal-title">Current support dominates; matched history adds a selective refinement.</h2><p>Independent donor prompts replace selected task-token pairs without using future recipient values. Replacing all three current-task supports adds 0.906 MSE. Same-component history replacement has a smaller but consistently greater effect than both different-component and count-matched random replacement.</p></div>
        <div className="figure-grid">
          <Figure src="/results/context-ablation.png" alt="Change in mean squared error after donor replacement of same-component, different-component, count-matched random, or all context" caption="Figure 4. At position 49, same-component replacement adds 0.012 MSE versus −0.006 for count-matched random replacement; the paired difference is significant across pools (p = 0.014)." />
          <Figure src="/results/support-ablation.png" alt="Change in mean squared error after donor replacement of each current-task support pair or all support pairs" caption="Figure 5. Individual support replacements add 0.253–0.274 MSE; replacing all three adds 0.906." />
        </div><a className="section-link" href={links.causal}>Causal intervention results →</a>
      </section>

      <section id="limits" className="section-block limits" aria-labelledby="limits-title">
        <p className="section-number">04 / Algorithm identification</p><h2 id="limits-title">The learned procedure is adaptive, but not a full mixture solver.</h2>
        <p className="wide-copy">Distribution shifts separate competing algorithmic accounts. The transformer is robust to moderate changes in similarity, imbalance, and hierarchy, and can exploit more components than it saw in the canonical two-component setup. Yet it falls well short of the assignment oracle under severe imbalance and at K = 4. Its behavior resembles an amortized, capacity-limited inference procedure—not exact expectation-maximization.</p>
        <div className="figure-grid">
          <Figure src="/results/ood-imbalance.png" alt="Mean squared error by component imbalance for transformer, assignment oracle, known-pool, and fitted baselines" caption="Figure 6. Forced-minority performance separates a protocol-aware assignment oracle from methods that must infer the component." />
          <Figure src="/results/ood-components.png" alt="Mean squared error across one to four evaluation components for transformer and fitted mixture models" caption="Figure 7. Component-count expansion tests finite-capacity adaptation against uniform-prior EM-ridge fits." />
        </div><a className="section-link" href={links.ood}>OOD results →</a>
      </section>

      <section className="comparison" aria-labelledby="comparison-title">
        <p className="section-number">05 / Previous setup</p><h2 id="comparison-title">The input format changes the mechanism’s fallback strategy.</h2>
        <p className="wide-copy">The earlier AR-mixture model could self-fit from observations inside the target run. The packed MLR query has no target label and must retrieve matching history. That single information-boundary change explains the strongest differences.</p>
        <div className="comparison-table" role="table" aria-label="Comparison of previous AR and new MLR results">
          <div className="comparison-head" role="row"><span>Evidence</span><span>Previous AR setup</span><span>New packed MLR setup</span></div>
          <div role="row"><strong>Attention</strong><span>Early-layer selectivity; late layers nearly uniform.</span><span>Selectivity grows late and reaches 0.496 in layer 5.</span></div>
          <div role="row"><strong>Representation</strong><span>Component ID jumps after layer 1, then plateaus; local evidence is a caveat.</span><span>Component ID and β decoding improve to stage 5, but packed support retains the local-evidence caveat.</span></div>
          <div role="row"><strong>Causality</strong><span>Mean activation ablation was weak and not component-selective.</span><span>Donor replacement finds a modest matched-history effect; current supports dominate.</span></div>
          <div role="row"><strong>99/1 minority</strong><span>Near-oracle by self-fitting the target run (~0.055 MSE).</span><span>Fails without matched history (0.966 MSE versus 0.039 assignment oracle).</span></div>
        </div>
        <p className="comparison-takeaway">The shared conclusion survives: the transformer is adaptive and component-sensitive. The MLR interventions clarify the balance: current-task supports drive the prediction, while matched historical tasks provide a smaller selective contribution.</p>
        <a className="section-link" href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/AR_VS_MLR_COMPARISON.md">Full comparison and comparability caveats →</a>
      </section>

      <section className="conclusion" aria-labelledby="conclusion-title">
        <p className="section-number">06 / Synthesis</p><h2 id="conclusion-title">A coherent circuit-level story, with clear boundaries.</h2>
        <div className="synthesis-grid">
          <div><span>1</span><h3>Infer</h3><p>Early and middle layers turn mixed observations into a representation of the active linear rule.</p></div>
          <div><span>2</span><h3>Route</h3><p>Late attention preferentially retrieves observations belonging to the query’s latent component.</p></div>
          <div><span>3</span><h3>Predict</h3><p>The routed evidence supports near-oracle in-distribution predictions, with graceful but incomplete OOD adaptation.</p></div>
        </div>
        <div className="caveat"><h3>Coverage and open decisions</h3><p>Mechanistic results use one canonical K = 2 checkpoint and pool-level uncertainty; the behavioral grid covers K ∈ {2, 3}, T ∈ {2, 3, 4, 5}, noise ∈ {0, 0.2}, across 16 trained checkpoints. A second K = 3 mechanistic replication, per-head main-text granularity, and whether noiseless results belong in the main narrative remain explicit editorial choices.</p><a href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/DECISIONS_AND_QUESTIONS.md">Decision log →</a><span className="link-separator">·</span><a href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/INDEPENDENT_AUDIT.md">Independent audit →</a></div>
      </section>

      <footer><p>Mixtures of Linear Regressions · reproducible experiment suite</p><a href="https://github.com/eddieqiao23/icl-time-series/tree/main/src/experiments/mlr">Experiment index</a></footer>
    </main>
  );
}

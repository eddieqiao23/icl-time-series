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
        <div className="model-map">
          <div><p className="result-label">Model provenance</p><h3>Which checkpoints feed each result</h3><p><strong>K</strong> is the number of latent regression components; <strong>T</strong> is the number of labeled support pairs packed into each task input. The attention and probe comparisons vary T, not K.</p></div>
          <div className="model-table" role="table" aria-label="Models used by experiment section">
            <div className="model-head" role="row"><span>Result</span><span>Checkpoints</span><span>Evaluation design</span></div>
            <div role="row"><strong>Behavior</strong><span>16 models: T ∈ {'{2,3,4,5}'}, K ∈ {'{2,3}'}, noise σ ∈ {'{0,0.2}'}</span><span>N = 50; final early-stopped or 500k checkpoint for each condition</span></div>
            <div role="row"><strong>Attention + probes</strong><span>Four trained models: T = 2–5; K = 2; σ = 0.2</span><span>N = 50; 500k checkpoints; 10 coefficient pools</span></div>
            <div role="row"><strong>Causal + OOD</strong><span>T3_K2_N50_noisy_500k</span><span>Run 88f7ba1f…; step 500,000</span></div>
          </div>
        </div>
      </section>

      <section id="evidence" className="section-block" aria-labelledby="evidence-title">
        <p className="section-number">02 / Converging evidence</p><h2 id="evidence-title">The same computation appears in behavior, attention, and state.</h2>
        <article className="result-row">
            <div className="result-copy"><p className="result-label">Behavior</p><h3>More within-task support closes most of the gap to the known-pool reference.</h3><p>In the noisy three-component condition, transformer MSE falls from <strong>0.176</strong> with two support pairs per task to <strong>0.053</strong> with five. The known-coefficient-pool Bayesian reference reaches 0.043. The sweep shows that the model uses the extra evidence packed inside each current task.</p><a href={links.behavior}>Behavior results →</a></div>
          <Figure src="/results/behavior-k3-noisy.png" alt="Mean squared error by support-pair count for five regression methods in the noisy three-component setting" caption="Figure 1. Mean performance across coefficient pools by within-task support size in the hardest in-distribution condition. Uncertainty bands are omitted pending a statistical audit." />
        </article>
        <div className="attention-detail">
          <div><p className="result-label">Attention</p><h3>The final query’s retrieval pattern changes by layer and support count.</h3><p>Each panel uses a trained K = 2 model. T varies from 2 to 5. The source is the final task’s packed input token at sequence position 98. For runs 0–48, each bar sums attention to that run’s packed-input and output tokens; run 49 is self-attention to the source token. Bars average 128 prompts, four heads, and 10 coefficient pools. The four T panels share a y-scale within each layer, while different layers use different scales so their structure remains visible.</p><p>Large self-attention peaks appear in the early and middle layers of several checkpoints, especially T = 3 and T = 4. Later layers generally shift probability toward recent history. In the canonical T = 3 model, the component-conditioned analysis still finds 0.0297 attention per same-component task versus 0.0102 per different-component task in layer 6. These probabilities describe routing; they are not causal-effect estimates.</p><a href={links.attention}>Attention methodology and tables →</a></div>
          <Figure src="/results/attention-by-t.png" alt="Six-layer grid of bar charts showing final-query attention weight by run index for trained models with two through five support pairs" caption="Figure 2. Final-query task-level attention for layers 1–6. Each layer contains T = 2, 3, 4, and 5 trained-model panels; no untrained panels or uncertainty bars are shown." />
          <div className="routing-analysis">
            <div><p className="result-label">Component routing ratio</p><h3>Component selectivity appears at different depths for different T.</h3><p>Each cell is mean attention per same-component prior task divided by mean attention per different-component prior task. Run 49 self-attention is excluded. A value of 1 means no preference. Attention is averaged across prompts, heads, and pools before the ratio is taken.</p></div>
            <div className="routing-table-wrap">
              <table className="routing-table">
                <thead><tr><th>T</th><th>Layer 1</th><th>Layer 2</th><th>Layer 3</th><th>Layer 4</th><th>Layer 5</th><th>Layer 6</th></tr></thead>
                <tbody>
                  <tr><th>2</th><td>1.00</td><td>1.02</td><td>1.00</td><td>1.56</td><td>1.04</td><td>1.04</td></tr>
                  <tr><th>3</th><td>1.02</td><td>1.13</td><td>0.94</td><td>1.24</td><td>1.38</td><td className="routing-high">2.90</td></tr>
                  <tr><th>4</th><td>1.04</td><td>1.00</td><td>1.03</td><td className="routing-high">2.15</td><td>1.17</td><td>1.19</td></tr>
                  <tr><th>5</th><td>1.02</td><td>1.03</td><td>1.43</td><td className="routing-high">3.67</td><td className="routing-high">6.35</td><td className="routing-high">2.27</td></tr>
                </tbody>
              </table>
            </div>
            <p className="routing-reading">T = 3 develops its strongest preference only in layer 6. T = 4 peaks transiently in layer 4. T = 5 becomes selective earlier and peaks in layer 5, where matching tasks receive 0.03425 attention each versus 0.00540 for mismatching tasks. T = 2 remains mostly component-neutral. The non-monotonic trajectories suggest that component-based routing can be an intermediate computation that later layers partially integrate or redistribute.</p>
          </div>
        </div>
        <article className="result-row">
            <div className="result-copy"><p className="result-label">Representation</p><h3>A linear readout recovers the active regression across support counts.</h3><p>Prompt-grouped coefficient probes rise through the residual stream for every trained checkpoint. At stage 5, β R² is <strong>0.802</strong>, <strong>0.861</strong>, <strong>0.760</strong>, and <strong>0.929</strong> for T = 2, 3, 4, and 5 respectively. T = 5 produces the clearest representation; the T = 4 checkpoint shows that the trend is not strictly monotonic. Because each packed query contains current-task support pairs, this establishes information availability—not whether it was computed locally or retrieved from history.</p><a href={links.probing}>Probe results →</a></div>
          <Figure src="/results/probe-beta-by-t.png" alt="Coefficient probe R squared by representation stage for trained models with two through five support pairs" caption="Figure 3. Mean coefficient decodability for the four trained support-count checkpoints. No untrained, shuffled-label, or uncertainty series are included." />
        </article>
      </section>

      <section id="causal" className="causal-section" aria-labelledby="causal-title">
        <div className="section-intro"><p className="section-number">03 / Causal tests</p><h2 id="causal-title">Current support dominates; matched history adds a selective refinement.</h2><p>This section uses only the canonical T = 3, K = 2, N = 50, σ = 0.2 checkpoint at step 500,000. It asks whether information identified by attention is actually necessary for prediction.</p></div>
        <div className="causal-method">
          <div><span>1 / Baseline</span><p>Sample 64 recipient prompts from each of 10 coefficient pools and record the model’s MSE at task positions 10, 20, 30, 40, and 49.</p></div>
          <div><span>2 / Donor construction</span><p>Generate an independently seeded donor prompt from the same coefficient pool. Donor values are on-distribution but contain no future or target values from the recipient.</p></div>
          <div><span>3 / Historical replacement</span><p>At the same earlier task positions, replace both the packed-input and output-token embeddings. Conditions replace same-component tasks, different-component tasks, a count-matched random set, or all history.</p></div>
          <div><span>4 / Current support</span><p>Separately replace one or all of the three (x, y) support blocks packed inside the current task input. The reported effect is intervention MSE minus the untouched baseline MSE.</p></div>
        </div>
        <p className="causal-reading">Replacing all current supports adds 0.906 MSE. Same-component historical replacement has a much smaller effect, but it is consistently more damaging than different-component and count-matched random replacement. This supports selective use of matching history without claiming that history dominates the prediction.</p>
        <div className="figure-grid">
          <Figure src="/results/context-ablation.png" alt="Change in mean squared error after donor replacement of same-component, different-component, count-matched random, or all context" caption="Figure 4. Mean intervention effect across pools. At position 49, same-component replacement adds 0.012 MSE versus −0.006 for count-matched random replacement; the paired pool-level difference is p = 0.014." />
          <Figure src="/results/support-ablation.png" alt="Change in mean squared error after donor replacement of each current-task support pair or all support pairs" caption="Figure 5. Mean intervention effect across pools. Individual support replacements add 0.253–0.274 MSE; replacing all three adds 0.906." />
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
        <div className="caveat"><h3>Coverage and open decisions</h3><p>Attention and probing now compare all four noisy K = 2 checkpoints with T ∈ {'{2,3,4,5}'}. Causal and OOD results still use only the canonical T = 3 checkpoint. The behavioral grid covers K ∈ {'{2,3}'}, T ∈ {'{2,3,4,5}'}, and noise ∈ {'{0,0.2}'} across 16 trained checkpoints. Figures show pool means without uncertainty bars while the error calculation is audited; the pool-level raw results remain saved.</p><a href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/DECISIONS_AND_QUESTIONS.md">Decision log →</a><span className="link-separator">·</span><a href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/INDEPENDENT_AUDIT.md">Independent audit →</a></div>
      </section>

      <footer><p>Mixtures of Linear Regressions · reproducible experiment suite</p><a href="https://github.com/eddieqiao23/icl-time-series/tree/main/src/experiments/mlr">Experiment index</a></footer>
    </main>
  );
}

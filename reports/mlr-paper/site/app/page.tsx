import type { Metadata } from "next";

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
        <p className="eyebrow">Mechanistic evaluation · 2 components · 6 layers</p>
        <h1>The model learns to identify a hidden task—and route the right examples to it.</h1>
        <p className="lede">A transformer trained on packed mixtures of linear regressions behaves like an adaptive inference procedure. It separates latent components, concentrates attention on relevant examples, and builds a decodable estimate of the active regression. Intervention and shift tests show both where this account is causal and where it is incomplete.</p>
        <div className="hero-stats" aria-label="Key results">
          <div><span>0.496</span><p>final-layer attention selectivity</p></div>
          <div><span>0.861</span><p>linear-probe β R² at stage 5</p></div>
          <div><span>0.819</span><p>MSE increase after relevant-context ablation</p></div>
        </div>
        <p className="scope">Canonical mechanistic checkpoint: T = 3 packed tasks, K = 2 components, N = 50 examples, noise σ = 0.2.</p>
      </header>

      <section className="setup" aria-labelledby="setup-title">
        <div>
          <p className="section-number">01 / Setup</p><h2 id="setup-title">One sequence, several hidden regressions</h2>
          <p>Each prompt interleaves examples from latent linear models. The transformer receives only the ordered <em>x, y</em> observations and must predict the held-out label for the final query. Solving the task requires inferring which component generated the query, then using the matching evidence rather than averaging indiscriminately.</p>
        </div>
        <div className="method-diagram" role="img" aria-label="Packed prompt with observations from two latent regression components leading to a query prediction">
          <div className="sequence-row"><span className="token component-a">x₁ y₁</span><span className="token component-b">x₂ y₂</span><span className="token component-a">x₃ y₃</span><span className="ellipsis">…</span><span className="token query">x?</span></div>
          <div className="routing-row"><span>latent component A</span><i></i><span>route matching evidence</span><i></i><span>ŷ?</span></div>
          <p className="legend"><span className="dot a"></span> component A <span className="dot b"></span> component B</p>
        </div>
      </section>

      <section id="evidence" className="section-block" aria-labelledby="evidence-title">
        <p className="section-number">02 / Converging evidence</p><h2 id="evidence-title">The same computation appears in behavior, attention, and state.</h2>
        <article className="result-row">
          <div className="result-copy"><p className="result-label">Behavior</p><h3>More context closes most of the gap to an oracle.</h3><p>In the noisy three-component condition, transformer MSE falls from <strong>0.176</strong> at two packed tasks to <strong>0.053</strong> at five. The oracle reaches 0.043. This improvement is hard to explain with a fixed global linear fit: the model benefits from repeated component-specific evidence.</p><a href={links.behavior}>Behavior results →</a></div>
          <Figure src="/results/behavior-k3-noisy.png" alt="Mean squared error by packed task count for five regression methods in the noisy three-component setting" caption="Figure 1. Performance by context depth in the hardest in-distribution condition. Bands show 95% confidence intervals across pools." />
        </article>
        <article className="result-row reverse">
          <div className="result-copy"><p className="result-label">Attention</p><h3>Later layers preferentially retrieve the query’s component.</h3><p>Same-component attention selectivity rises sharply through the network and reaches <strong>0.496</strong> in layer 5. An architecture-matched untrained control stays near zero. The learned attention pattern is therefore aligned with the latent partition the task demands.</p><a href={links.attention}>Attention results →</a></div>
          <Figure src="/results/attention-by-layer.png" alt="Same-component attention selectivity by transformer layer for trained and untrained models" caption="Figure 2. Query attention to same- versus different-component examples, aggregated across 10 independent prompt pools." />
        </article>
        <article className="result-row">
          <div className="result-copy"><p className="result-label">Representation</p><h3>A linear readout recovers the active regression.</h3><p>A held-out linear probe’s β R² climbs from below zero at the embedding to <strong>0.861</strong> at stage 5; component classification reaches <strong>96.8%</strong>. Raw inputs and untrained controls remain near chance, while grouped cross-validation prevents examples from the same prompt leaking across folds.</p><a href={links.probing}>Probe results →</a></div>
          <Figure src="/results/probe-beta.png" alt="Linear-probe beta coefficient R squared by network stage for trained and control representations" caption="Figure 3. Decodability of the active regression coefficients across the residual stream." />
        </article>
      </section>

      <section id="causal" className="causal-section" aria-labelledby="causal-title">
        <div className="section-intro"><p className="section-number">03 / Causal tests</p><h2 id="causal-title">Relevant evidence is not merely correlated—it changes the answer.</h2><p>Two interventions connect the observed circuit to behavior. Replacing same-component context damages predictions increasingly late in the prompt; replacing other-component context has almost no effect. Patching clean internal activations into corrupted runs restores most performance only at late stages.</p></div>
        <div className="figure-grid">
          <Figure src="/results/context-ablation.png" alt="Change in mean squared error after replacing same-component, different-component, or all context" caption="Figure 4. At position 49, same-component replacement adds 0.819 MSE; different-component replacement changes MSE by −0.012." />
          <Figure src="/results/activation-patching.png" alt="Fraction of corrupted performance recovered by patching activations at successive network stages" caption="Figure 5. Recovery rises to 0.900 at stage 5 and 1.000 after the final normalization boundary." />
        </div><a className="section-link" href={links.causal}>Causal intervention results →</a>
      </section>

      <section id="limits" className="section-block limits" aria-labelledby="limits-title">
        <p className="section-number">04 / Algorithm identification</p><h2 id="limits-title">The learned procedure is adaptive, but not a full mixture solver.</h2>
        <p className="wide-copy">Distribution shifts separate competing algorithmic accounts. The transformer is robust to moderate changes in similarity, imbalance, and hierarchy, and can exploit more components than it saw in the canonical two-component setup. Yet it falls well short of the correct mixture oracle under severe imbalance and at K = 4. Its behavior resembles an amortized, capacity-limited routing procedure—not exact expectation-maximization.</p>
        <div className="figure-grid">
          <Figure src="/results/ood-imbalance.png" alt="Mean squared error by component imbalance for transformer, oracle, and EM baselines" caption="Figure 6. With 99% majority context and a forced minority query, transformer MSE is 0.966 versus 0.047 for the oracle and 1.975 for two-component EM." />
          <Figure src="/results/ood-components.png" alt="Mean squared error across one to four evaluation components for transformer and fitted mixture models" caption="Figure 7. At K = 4, transformer MSE is 0.251, compared with 0.525 for two-component EM and 0.067 for correctly specified four-component EM." />
        </div><a className="section-link" href={links.ood}>OOD results →</a>
      </section>

      <section className="conclusion" aria-labelledby="conclusion-title">
        <p className="section-number">05 / Synthesis</p><h2 id="conclusion-title">A coherent circuit-level story, with clear boundaries.</h2>
        <div className="synthesis-grid">
          <div><span>1</span><h3>Infer</h3><p>Early and middle layers turn mixed observations into a representation of the active linear rule.</p></div>
          <div><span>2</span><h3>Route</h3><p>Late attention preferentially retrieves observations belonging to the query’s latent component.</p></div>
          <div><span>3</span><h3>Predict</h3><p>The routed evidence supports near-oracle in-distribution predictions, with graceful but incomplete OOD adaptation.</p></div>
        </div>
        <div className="caveat"><h3>Coverage and open decisions</h3><p>Mechanistic results use one canonical K = 2 checkpoint and pool-level uncertainty; the behavioral grid covers K ∈ {2, 3}, T ∈ {2, 3, 4, 5}, noise ∈ {0, 0.2}, across 16 trained checkpoints. A second K = 3 mechanistic replication, per-head main-text granularity, and whether noiseless results belong in the main narrative remain explicit editorial choices.</p><a href="https://github.com/eddieqiao23/icl-time-series/blob/main/docs/mlr-paper/DECISIONS_AND_QUESTIONS.md">Decision log →</a></div>
      </section>

      <footer><p>Mixtures of Linear Regressions · reproducible experiment suite</p><a href="https://github.com/eddieqiao23/icl-time-series/tree/main/src/experiments/mlr">Experiment index</a></footer>
    </main>
  );
}

import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers:{ accept:"text/html" } }), { ASSETS:{ fetch:async () => new Response("Not found", { status:404 }) } }, { waitUntil(){}, passThroughOnException(){} });
}
test("renders the complete MLR report", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /How Transformers Solve Mixtures of Linear Regressions/);
  assert.match(html, /Converging evidence/);
  assert.match(html, /Causal tests/);
  assert.match(html, /Algorithm identification/);
  assert.match(html, /Previous setup/);
  assert.match(html, /sequence position 98/i);
  assert.match(html, /Donor construction/);
  assert.match(html, /Which checkpoints feed each result/);
  assert.match(html, /6\.35/);
  assert.match(html, /0\.861/);
  assert.doesNotMatch(html, /Your site is taking shape|SkeletonPreview/);
});
test("ships every referenced result figure", async () => {
  const figures = ["behavior-k3-noisy.png","attention-by-t.png","probe-beta-by-t.png","context-ablation.png","support-ablation.png","ood-imbalance.png","ood-components.png"];
  await Promise.all(figures.map(name => access(new URL(`public/results/${name}`, root))));
});

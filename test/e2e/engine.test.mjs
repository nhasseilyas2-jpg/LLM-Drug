// End-to-end check of the patched llama-server. Skips unless the binary and a small GGUF exist.
// LLM_INJ_E2E_MODEL=<path.gguf> picks the model; otherwise the smallest .gguf under models/ is used.
import assert from "node:assert/strict";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { after, before, test } from "node:test";
import { TECHNIQUE_IDS, buildLlamaEnv, getTechnique, needsFixation } from "../../src/catalog.js";
import { wordDivergence } from "../../src/metrics.js";
import { loadConfig } from "../../lib/config.js";
import { LlamaServerPool } from "../../lib/llamacpp.js";
import { buildMessages } from "../../lib/experiment.js";

const config = { ...loadConfig(), llamaCtx: 2048, llamaMaxServers: 2 };
const pool = new LlamaServerPool(config);

async function findModel() {
  if (process.env.LLM_INJ_E2E_MODEL) return process.env.LLM_INJ_E2E_MODEL;
  let best = null;
  for (const dir of config.modelsDirs) {
    let entries = [];
    try { entries = await readdir(dir); } catch { continue; }
    for (const name of entries.filter((n) => /\.gguf$/i.test(n) && !/mmproj/i.test(n))) {
      const file = path.join(dir, name);
      const { size } = await stat(file);
      if (!best || size < best.size) best = { file, size };
    }
  }
  return best && best.size < 2.5e9 ? best.file : null;
}

const model = await findModel();
const skip = !(await pool.binaryAvailable()) ? "patched llama-server not built" : !model ? "no small GGUF model found" : false;

const messages = buildMessages({
  system: "You are a helpful assistant.",
  memory: "Vault code: 7391. Meeting room: B12. Contact: Dana Ortiz. The meeting is on Thursday at 3 pm.",
  prompt: "Write three sentences reminding the team of the vault code, the meeting room, the contact person and the meeting time."
});
const sampling = { temperature: 0, top_p: 1, top_k: 0, min_p: 0, repeat_penalty: 1 };
const gen = (env) => pool.chat(model, env, { messages, sampling, seed: 7, maxTokens: 64, logprobs: false });
let baseline;

before(async () => { if (!skip) baseline = await gen({}); });
after(() => pool.stopAll());

test("untreated baseline is deterministic and engine inactive", { skip }, async () => {
  const again = await gen({});
  assert.equal(again.content, baseline.content);
  assert.ok(baseline.content.length > 20);
  assert.deepEqual(baseline.engine.sitesFired, []);
});

test("placebo at 500 mg is byte-identical to baseline", { skip }, async () => {
  const r = await gen(buildLlamaEnv("placebo", 500, { seed: 7 }).env);
  assert.equal(r.content, baseline.content);
});

for (const id of TECHNIQUE_IDS.filter((t) => t !== "placebo" && t !== "creativity")) {
  test(`${id} at 500 mg fires its sites and changes greedy output`, { skip, timeout: 180000 }, async () => {
    const fixationIds = needsFixation(id, 500) ? await pool.tokenIds(model, getTechnique(id).theme) : [];
    const { env } = buildLlamaEnv(id, 500, { seed: 7, fixationIds });
    const r = await gen(env);
    assert.ok(r.engine.active, "engine reported ACTIVE");
    assert.ok(r.engine.sitesFired.length > 0, "at least one site fired");
    assert.deepEqual(r.engine.warnings, []);
    const d = wordDivergence(baseline.content, r.content);
    assert.ok(d > 0.1, `divergence ${d.toFixed(3)} too small:\n${r.content}`);
  });
}

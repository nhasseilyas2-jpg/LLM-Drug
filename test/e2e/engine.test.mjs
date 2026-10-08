// End-to-end check of the patched llama-server. Skips unless the binary and a small GGUF exist.
// LLM_INJ_E2E_MODEL=<path.gguf> picks the model; otherwise the smallest .gguf under models/ is used.
import assert from "node:assert/strict";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { after, before, test } from "node:test";
import { TECHNIQUE_IDS, buildLlamaEnv, buildLlamaRegimen, getTechnique, needsFixation } from "../../src/catalog.js";
import { wordDivergence } from "../../src/metrics.js";
import { loadConfig } from "../../lib/config.js";
import { LlamaServerPool } from "../../lib/llamacpp.js";
import { buildMessages } from "../../lib/experiment.js";
import { idFor } from "../../lib/models.js";
import { resolveSteering } from "../../lib/steering.js";
import { makeVectors } from "../../scripts/make-control-vectors.mjs";

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

// steering techniques need the model's control vectors (built once, ~1 min on CPU, then cached in data/steering)
const steered = (env) => resolveSteering(config, idFor(model), path.basename(model), env);

before(async () => {
  if (skip) return;
  await makeVectors({ model, vector: "all", log: () => {} });
  baseline = await gen({});
});
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
    const env = await steered(buildLlamaEnv(id, 500, { seed: 7, fixationIds }).env);
    const r = await gen(env);
    assert.ok(r.engine.active, "engine reported ACTIVE");
    assert.ok(r.engine.sitesFired.length > 0, "at least one site fired");
    assert.deepEqual(r.engine.warnings, []);
    const d = wordDivergence(baseline.content, r.content);
    assert.ok(d > 0.1, `divergence ${d.toFixed(3)} too small:\n${r.content}`);
  });
}

const raw = (extra) => ({ LLM_INJ_SEED: "7", ...extra });

test("shared mode: one process per model, config switches are exact", { skip, timeout: 300000 }, async () => {
  assert.equal(pool.sharedSupported, true, "engine supports per-request config reload");
  const env = buildLlamaEnv("delirium", 300, { seed: 7 }).env;
  const a = await gen(env);
  const b = await gen({});
  const c = await gen(env);
  assert.equal(b.content, baseline.content, "untreated request after a treated one equals baseline");
  assert.deepEqual(b.engine.sitesFired, []);
  assert.equal(c.content, a.content, "same config gives the same output after switching");
  assert.deepEqual(c.engine.sitesFired.sort(), a.engine.sitesFired.sort());
  assert.equal(pool.status().filter((x) => x.model === model).length, 1);

  // the per-config fallback (fresh process with a fixed environment) produces the same text
  const solo = new LlamaServerPool({ ...config, llamaShared: false });
  try {
    const d = await solo.chat(model, env, { messages, sampling, seed: 7, maxTokens: 64, logprobs: false });
    assert.equal(d.content, a.content);
    assert.equal(d.engine.shared, false);
  } finally {
    solo.stopAll();
  }
});

const sites = [
  ["heads", { LLM_INJ_HEAD_LESION: "0.5" }],
  ["ffn_lesion", { LLM_INJ_FFN_LESION: "0.5" }],
  ["resid_noise", { LLM_INJ_RESID_NOISE: "0.6", LLM_INJ_NOISE_MODE: "hash" }],
  ["ffn_dropout", { LLM_INJ_FFN_DROPOUT: "0.5", LLM_INJ_NOISE_MODE: "hash" }]
];
for (const [site, extra] of sites) {
  test(`${site} site (${Object.values(extra).join(" ")}) fires, is deterministic and changes output`, { skip, timeout: 180000 }, async () => {
    const r1 = await gen(raw(extra));
    const r2 = await gen(raw(extra));
    assert.ok(r1.engine.sitesFired.includes(site), `fired: ${r1.engine.sitesFired}`);
    assert.equal(r1.content, r2.content);
    assert.ok(wordDivergence(baseline.content, r1.content) > 0.1, r1.content);
  });
}

test("explicit head ids lesion only the listed heads", { skip, timeout: 180000 }, async () => {
  const r = await gen(raw({ LLM_INJ_HEAD_IDS: "0:0,0:1,1:0,1:1,2:0,2:1", LLM_INJ_HEAD_GAIN: "0" }));
  assert.ok(r.engine.sitesFired.includes("heads"));
  assert.match(r.engine.active, /heads/);
});

test("dose schedule: onset delays the effect, the first token is sober", { skip, timeout: 180000 }, async () => {
  const strong = raw({ LLM_INJ_ATTN_SCALE: "0.05", LLM_INJ_LOGIT_NOISE: "4" });
  const now = await gen(strong);
  const delayed = await gen({ ...strong, LLM_INJ_PK_ONSET: "12" });
  const firstWord = (t) => t.trim().split(/\s+/)[0];
  assert.match(delayed.engine.active, /pk\{onset=12/);
  assert.equal(firstWord(delayed.content), firstWord(baseline.content), "m(0) = 0: first token unaffected");
  assert.ok(wordDivergence(baseline.content, delayed.content) > 0.1, "effect kicks in later");
  assert.notEqual(delayed.content, now.content);
  // fast elimination: the effect wears off and the text is closer to baseline than without it
  const wearing = await gen({ ...strong, LLM_INJ_PK_HALFLIFE: "2" });
  assert.ok(wordDivergence(baseline.content, wearing.content) <= wordDivergence(baseline.content, now.content));
});
test("co-administration: the merged regimen fires the sites of both techniques", { skip, timeout: 180000 }, async () => {
  const fixationIds = await pool.tokenIds(model, getTechnique("paranoia").theme);
  const { env } = buildLlamaRegimen({ techniqueId: "amnesia", doseMg: 300, coTechniqueId: "paranoia", coDoseMg: 300, seed: 7, fixationIds, schedule: { onset: 4, halfLife: 200 } });
  const r = await gen(env);
  for (const s of ["kv_forget", "logits"]) assert.ok(r.engine.sitesFired.includes(s), `fired: ${r.engine.sitesFired}`);
  assert.match(r.engine.active, /pk\{onset=4/);
  assert.ok(wordDivergence(baseline.content, r.content) > 0.1, r.content);
});

test("steering: euphoria and dysphoria push the same answer in opposite directions", { skip, timeout: 240000 }, async () => {
  const ask = buildMessages({ system: "You are a helpful assistant.", memory: "", prompt: "Describe your plans for the weekend in three sentences." });
  const run = async (id, dose) => pool.chat(model, await steered(buildLlamaEnv(id, dose, { seed: 7 }).env), { messages: ask, sampling, seed: 7, maxTokens: 48, logprobs: false });
  const sober = await run("placebo", 0);
  const up = await run("euphoria", 150);
  const down = await run("dysphoria", 150);
  for (const r of [up, down]) {
    assert.deepEqual(r.engine.sitesFired, ["steer"]);
    assert.match(r.engine.active, /steer\{scale=-?[0-9.]+ file=mood\.gguf/);
    assert.ok(wordDivergence(sober.content, r.content) > 0.1, r.content);
  }
  assert.ok(wordDivergence(up.content, down.content) > 0.1, `${up.content}\n---\n${down.content}`);
  // equal doses of both cancel exactly (scale 0: steering inactive)
  const both = buildLlamaRegimen({ techniqueId: "euphoria", doseMg: 150, coTechniqueId: "dysphoria", coDoseMg: 150, seed: 7 });
  const cancel = await pool.chat(model, await steered(both.env), { messages: ask, sampling, seed: 7, maxTokens: 48, logprobs: false });
  assert.equal(cancel.content, sober.content);
  // a missing vector file is reported and ignored
  const missing = await pool.chat(model, raw({ LLM_INJ_STEER_FILE: path.join(config.steeringDir, "nope.gguf"), LLM_INJ_STEER_SCALE: "1" }), { messages: ask, sampling, seed: 7, maxTokens: 16, logprobs: false });
  assert.deepEqual(missing.engine.sitesFired, []);
  assert.ok(missing.engine.warnings.some((w) => /STEER_FILE/.test(w)), JSON.stringify(missing.engine.warnings));
});

test("clean-model surprisal: greedy baseline is unsurprising, a treated answer is surprising", { skip, timeout: 180000 }, async () => {
  const clean = await pool.score(model, messages, baseline.content);
  assert.ok(clean && clean.coverage === 1, JSON.stringify(clean));
  assert.ok(clean.surprisal < 1.5, `baseline surprisal ${clean.surprisal}`);
  const r = await gen(buildLlamaEnv("delirium", 400, { seed: 7 }).env);
  const treated = await pool.score(model, messages, r.content);
  assert.ok(treated.surprisal > clean.surprisal + 1, `treated ${treated.surprisal} vs baseline ${clean.surprisal}`);
  const again = await pool.score(model, messages, baseline.content);
  assert.equal(again.surprisal, clean.surprisal, "scoring is deterministic and unaffected by the previous injection");
});

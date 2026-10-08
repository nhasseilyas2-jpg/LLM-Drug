import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_SAMPLING,
  TECHNIQUES,
  TECHNIQUE_IDS,
  buildLlamaEnv,
  buildOllamaOptions,
  describeTechnique,
  getTechnique,
  hill,
  intensity,
  needsFixation,
  normalizeDose,
  parseDoseList,
  parseTheme,
  resolveTechniqueId
} from "../src/catalog.js";

const ENGINE_RANGES = {
  LLM_INJ_LOGIT_NOISE: [0, 50],
  LLM_INJ_LOGIT_TEMP: [0.05, 20],
  LLM_INJ_TOP_SUPPRESS: [0, 1],
  LLM_INJ_TAIL_BOOST: [0, 100],
  LLM_INJ_FIXATION_BIAS: [-100, 100],
  LLM_INJ_ATTN_SCALE: [0.01, 20],
  LLM_INJ_RESID_NOISE: [0, 10],
  LLM_INJ_LAYER_GAIN: [-4, 4],
  LLM_INJ_FFN_DROPOUT: [0, 0.95],
  LLM_INJ_KV_FORGET: [0, 1]
};

test("Hill curve: zero at 0 mg, half at EC50, monotonic, below 1", () => {
  const curve = { ec50: 150, n: 2 };
  assert.equal(hill(0, curve), 0);
  assert.ok(Math.abs(hill(150, curve) - 0.5) < 1e-12);
  let prev = 0;
  for (let d = 5; d <= 500; d += 5) {
    const e = hill(d, curve);
    assert.ok(e > prev && e < 1, `dose ${d}`);
    prev = e;
  }
});

test("doses are clamped to 0-500 mg and junk becomes 0", () => {
  assert.equal(normalizeDose(-20), 0);
  assert.equal(normalizeDose(9999), 500);
  assert.equal(normalizeDose("abc"), 0);
  assert.deepEqual(parseDoseList("500, 0, 50, 50, x"), [0, 50, 500]);
  assert.deepEqual(parseDoseList(""), [0, 25, 50, 100, 200, 350, 500]);
});

test("every technique is fully described", () => {
  for (const id of TECHNIQUE_IDS) {
    const d = describeTechnique(id);
    assert.ok(d.name && d.summary && d.analogy, id);
    assert.ok(Array.isArray(d.mechanism) && d.mechanism.length > 0, id);
    assert.ok(["full", "approximate", "none"].includes(d.support.ollama), id);
    assert.ok(d.curve.ec50 > 0 && d.curve.n > 0, id);
    for (const text of [d.summary, d.analogy, ...d.mechanism]) assert.doesNotMatch(text, /[ÃÂ]|\uFFFD/, `${id}: encoding damage in "${text}"`);
  }
});

test("legacy prototype ids resolve to the new techniques", () => {
  assert.equal(resolveTechniqueId("ego"), "stimulant");
  assert.equal(resolveTechniqueId("confusion"), "delirium");
  assert.equal(resolveTechniqueId("hallucinogen"), "hallucinogen");
  assert.equal(resolveTechniqueId("nope"), null);
  assert.throws(() => getTechnique("nope"), /Unknown technique/);
});

test("zero dose and placebo produce no engine perturbation (seed only)", () => {
  for (const id of TECHNIQUE_IDS) {
    assert.deepEqual(Object.keys(buildLlamaEnv(id, 0, { seed: 3 }).env), ["LLM_INJ_SEED"], id);
  }
  assert.deepEqual(buildLlamaEnv("placebo", 500, { seed: 3 }).env, { LLM_INJ_SEED: "3" });
});

test("engine parameters stay inside the engine's accepted ranges at every dose", () => {
  for (const id of TECHNIQUE_IDS) {
    for (const dose of [1, 50, 150, 300, 500]) {
      const { env } = buildLlamaEnv(id, dose, { seed: 1, fixationIds: [11, 12] });
      for (const [key, value] of Object.entries(env)) {
        if (!ENGINE_RANGES[key]) continue;
        const [lo, hi] = ENGINE_RANGES[key];
        const v = Number(value);
        assert.ok(Number.isFinite(v) && v >= lo && v <= hi, `${id} ${dose}mg ${key}=${value}`);
      }
      if (env.LLM_INJ_KV_RECENT) assert.ok(Number(env.LLM_INJ_KV_RECENT) >= 8);
    }
  }
});

test("perturbation strength grows with dose", () => {
  const strength = (id, dose) => {
    const { env } = buildLlamaEnv(id, dose);
    return Math.abs(Number(env.LLM_INJ_RESID_NOISE || 0)) + Math.abs(Number(env.LLM_INJ_LOGIT_NOISE || 0)) +
      Math.abs(1 - Number(env.LLM_INJ_ATTN_SCALE || 1)) + Math.abs(1 - Number(env.LLM_INJ_LAYER_GAIN || 1)) +
      Number(env.LLM_INJ_FFN_DROPOUT || 0) + Number(env.LLM_INJ_KV_FORGET || 0) + Math.abs(Number(env.LLM_INJ_FIXATION_BIAS || 0));
  };
  for (const id of TECHNIQUE_IDS.filter((x) => x !== "placebo")) {
    assert.ok(strength(id, 50) < strength(id, 200) && strength(id, 200) < strength(id, 500), id);
  }
});

test("fixation techniques carry token ids; others never do", () => {
  assert.ok(needsFixation("delusion", 100));
  assert.ok(!needsFixation("delusion", 0));
  assert.ok(!needsFixation("hallucinogen", 300));
  assert.equal(buildLlamaEnv("paranoia", 200, { fixationIds: [5, 9] }).env.LLM_INJ_FIXATION_IDS, "5,9");
  assert.equal(buildLlamaEnv("hallucinogen", 200, { fixationIds: [5] }).env.LLM_INJ_FIXATION_IDS, undefined);
  assert.deepEqual(parseTheme(" a, b ,,c"), ["a", "b", "c"]);
  assert.deepEqual(parseTheme("", ["x"]), ["x"]);
});

test("Ollama mapping: placebo identical, internal-only techniques unsupported, changes listed", () => {
  const placebo = buildOllamaOptions("placebo", 300, DEFAULT_SAMPLING);
  const zero = buildOllamaOptions("hallucinogen", 0, DEFAULT_SAMPLING);
  assert.deepEqual(placebo.options, zero.options);
  assert.equal(placebo.changes.length, 0);
  for (const id of TECHNIQUE_IDS) {
    const r = buildOllamaOptions(id, 300, DEFAULT_SAMPLING);
    if (TECHNIQUES[id].ollamaSupport === "none") assert.equal(r, null, id);
    else {
      assert.ok(r.options.temperature >= 0 && r.options.top_p > 0 && r.options.top_p <= 1, id);
      if (id !== "placebo") assert.ok(r.changes.length > 0, id);
    }
  }
  assert.ok(buildOllamaOptions("amnesia", 500, DEFAULT_SAMPLING).options.num_ctx < DEFAULT_SAMPLING.num_ctx);
});

test("intensity() matches the technique curve", () => {
  const t = getTechnique("delirium");
  assert.equal(intensity("delirium", 160), hill(160, t.curve));
  assert.equal(intensity("confusion", 160), hill(160, t.curve));
});

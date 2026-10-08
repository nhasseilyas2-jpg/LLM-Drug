import assert from "node:assert/strict";
import test from "node:test";
import * as metrics from "../src/metrics.js";
import { JobQueue } from "../lib/server.js";
import { ValidationError, buildMessages, runAgent, runDoseResponse, runTrial, validateInput } from "../lib/experiment.js";
import { childEnv, parseAuditLine } from "../lib/llamacpp.js";
import { normalizeRecord } from "../lib/history.js";

const GGUF = "gguf:0123456789abcdef";
const base = { backend: "llamacpp", modelId: GGUF, techniqueId: "delirium", prompt: "Say hi." };

// Fake runtime: output depends only on whether the arm carries an injection env/options,
// so tests exercise the experiment wiring without any model.
function mockLab() {
  const calls = [];
  const clean = "The answer is blue because of Rayleigh scattering.";
  const reply = (treated, seed) => ({
    content: treated ? "Thhe blu ansr zzq qqv because because because" : clean + (seed > 100000 ? " Indeed." : ""),
    logprobs: null,
    usage: { completion_tokens: 9 },
    finishReason: "stop",
    engine: treated ? { active: "mock", sitesFired: ["logits"], warnings: [] } : { active: null, sitesFired: [], warnings: [] }
  });
  return {
    calls,
    metrics,
    registry: { gguf: new Map([[GGUF, {}]]), resolveGguf: () => ({ file: "/models/m.gguf", name: "m.gguf" }) },
    pool: {
      chat: async (file, env, req) => {
        calls.push({ kind: "llamacpp", env, seed: req.seed });
        return reply(Object.keys(env).some((k) => k !== "LLM_INJ_SEED"), req.seed);
      },
      tokenIds: async () => [101, 202]
    },
    ollama: async (model, req) => {
      calls.push({ kind: "ollama", model, options: req.options, seed: req.seed });
      return reply(req.options.temperature > 0.8, req.seed);
    },
    judge: async () => ({ model: "j", blind: true, baseline: { coherence: 90 }, treated: { coherence: 20 } })
  };
}

test("validateInput rejects unsafe or malformed requests", () => {
  const bad = [
    [null, /JSON object/],
    [[], /JSON object/],
    [{ ...base, backend: "openai" }, /backend/],
    [{ ...base, modelId: "C:\\models\\evil.gguf" }, /gguf:<id>/],
    [{ ...base, modelId: "../../etc/passwd" }, /gguf:<id>/],
    [{ ...base, backend: "ollama", modelId: GGUF }, /ollama:<name>/],
    [{ ...base, techniqueId: "heroin" }, /Unknown techniqueId/],
    [{ ...base, prompt: "" }, /prompt is required/],
    [{ ...base, prompt: 5 }, /must be a string/],
    [{ ...base, prompt: "x".repeat(20001) }, /too long/],
    [{ ...base, judgeModelId: "gguf:0123456789abcdef" }, /judge/],
    [{ ...base, backend: "ollama", modelId: "ollama:qwen", techniqueId: "dissociative" }, /no Ollama equivalent/]
  ];
  for (const [body, re] of bad) assert.throws(() => validateInput(body), (e) => e instanceof ValidationError && e.status === 400 && re.test(e.message), JSON.stringify(body)?.slice(0, 80));
});

test("validateInput normalizes, clamps and maps legacy ids", () => {
  const v = validateInput({ ...base, drugId: "confusion", techniqueId: undefined, doseMg: 9000, seed: -5, extra: "dropped" });
  assert.equal(v.techniqueId, "delirium");
  assert.equal(v.doseMg, 500);
  assert.equal(v.seed, 0);
  assert.equal(v.extra, undefined);
  const d = validateInput({ ...base, doses: "0,10,20,30,40,50,60,70,80,90,100,110,120,130", trials: 99 }, "dose-response");
  assert.equal(d.doses.length, 12);
  assert.equal(d.trials, 10);
  assert.equal(validateInput({ ...base, steps: 100 }, "agent").steps, 12);
});

test("buildMessages puts memory before the prompt", () => {
  assert.deepEqual(buildMessages({ system: "S", memory: " M ", prompt: " P " }), [
    { role: "system", content: "S" },
    { role: "user", content: "Context:\nM\n\nP" }
  ]);
  assert.deepEqual(buildMessages({ system: "", memory: "", prompt: "P" }), [{ role: "user", content: "P" }]);
});

test("parseAuditLine reads all engine audit line kinds", () => {
  const audit = { active: null, sitesFired: [], warnings: [] };
  assert.equal(parseAuditLine("main: loading model", audit), false);
  parseAuditLine("llm-injection: ACTIVE seed=1 logits{noise=0.3}", audit);
  parseAuditLine("llm-injection: site fired: logits", audit);
  parseAuditLine("llm-injection: site fired: logits", audit);
  parseAuditLine("llm-injection: site fired: kv_forget", audit);
  parseAuditLine("llm-injection: ignoring invalid LLM_INJ_ATTN_SCALE='abc'", audit);
  assert.equal(audit.active, "seed=1 logits{noise=0.3}");
  assert.deepEqual(audit.sitesFired, ["logits", "kv_forget"]);
  assert.equal(audit.warnings.length, 1);
  parseAuditLine("llm-injection: inactive (no LLM_INJ_* set)", audit);
  assert.equal(audit.active, null);
});

test("childEnv strips inherited LLM_INJ_* so baselines are truly untreated", () => {
  const env = childEnv({ PATH: "p", LLM_INJ_LOGIT_NOISE: "9", llm_inj_attn_scale: "2" }, { LLM_INJ_SEED: "1" });
  assert.deepEqual(env, { PATH: "p", LLM_INJ_SEED: "1" });
});

test("normalizeRecord tags legacy prototype records", () => {
  const legacy = normalizeRecord({ id: "x", profile: { drugId: "ego", doseMg: 200 }, audit: { backend: "ollama-runtime" } });
  assert.equal(legacy.legacy, true);
  assert.equal(legacy.techniqueId, "stimulant");
  assert.equal(legacy.doseMg, 200);
  assert.equal(legacy.backend, "ollama");
  const current = { schema: 2, id: "y", techniqueId: "amnesia" };
  assert.equal(normalizeRecord(current), current);
});

test("runTrial: llama.cpp arms use empty baseline env and seeded treated env", async () => {
  const lab = mockLab();
  const input = validateInput({ ...base, doseMg: 300, sampling: { temperature: 0.7 }, judgeModelId: "ollama:judge" });
  const steps = [];
  const r = await runTrial(lab, input, { progress: (p) => steps.push(p.message) });
  assert.equal(lab.calls.length, 3);
  assert.deepEqual(lab.calls[0].env, {});
  assert.deepEqual(lab.calls[1].env, {});
  assert.equal(lab.calls[1].seed, input.seed + 100003);
  assert.equal(lab.calls[2].env.LLM_INJ_SEED, String(input.seed));
  assert.ok(lab.calls[2].env.LLM_INJ_FFN_DROPOUT);
  assert.equal(r.schema, 2);
  assert.ok(r.metrics.impairment > 30);
  assert.ok(r.arms.noise && r.judge.blind);
  assert.deepEqual(steps.slice(0, 3), ["baseline", "noise floor", "treated"]);
});

test("runTrial: no noise floor at temperature 0; fixation ids are looked up", async () => {
  const lab = mockLab();
  const r = await runTrial(lab, validateInput({ ...base, techniqueId: "delusion", doseMg: 200, sampling: { temperature: 0 } }));
  assert.equal(lab.calls.length, 2);
  assert.equal(r.arms.noise, null);
  assert.equal(lab.calls[1].env.LLM_INJ_FIXATION_IDS, "101,202");
});

test("runTrial: Ollama baseline uses placebo options, treated arm changes sampling", async () => {
  const lab = mockLab();
  const r = await runTrial(lab, validateInput({ backend: "ollama", modelId: "ollama:qwen", techniqueId: "hallucinogen", doseMg: 500, prompt: "x", sampling: { temperature: 0 } }));
  assert.equal(lab.calls[0].model, "qwen");
  assert.ok(lab.calls[1].options.temperature > lab.calls[0].options.temperature);
  assert.equal(r.treatment.kind, "ollama");
  assert.ok(r.treatment.changes.length > 0);
});

test("runDoseResponse reuses baselines across doses and summarizes", async () => {
  const lab = mockLab();
  const r = await runDoseResponse(lab, validateInput({ ...base, doses: "0,250,500", trials: 2, sampling: { temperature: 0.5 } }, "dose-response"));
  assert.equal(lab.calls.length, 2 * 2 + 3 * 2);
  assert.equal(r.rows.length, 6);
  assert.deepEqual(r.summary.map((s) => s.doseMg), [0, 250, 500]);
  assert.equal(r.summary[0].impairment.mean, 0);
  assert.ok(r.summary[2].impairment.mean > 30);
});

test("runAgent produces paired baseline/treated trajectories", async () => {
  const lab = mockLab();
  const r = await runAgent(lab, validateInput({ ...base, steps: 3, doseMg: 400 }, "agent"));
  assert.equal(r.steps.length, 3);
  assert.equal(lab.calls.length, 6);
  assert.ok(r.steps.every((s) => s.metrics.divergence > 0.5));
});

test("cancellation propagates through the job queue", async () => {
  const q = new JobQueue();
  let started;
  const running = new Promise((r) => { started = r; });
  const job = q.submit("run", ({ signal }) => new Promise((_, reject) => {
    started();
    signal.addEventListener("abort", () => reject(signal.reason));
  }));
  const queued = q.submit("run", async () => "never");
  await running;
  q.cancel(queued.id);
  q.cancel(job.id);
  await q.chain;
  assert.equal(job.status, "cancelled");
  assert.equal(queued.status, "cancelled");
  assert.equal(q.view(job).controller, undefined);
  const ok = q.submit("run", async ({ progress }) => { progress({ done: 1, total: 1 }); return 7; });
  await q.chain;
  assert.equal(ok.status, "done");
  assert.equal(ok.result, 7);
});

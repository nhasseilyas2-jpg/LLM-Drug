import {
  DEFAULT_DOSES,
  buildLlamaRegimen,
  buildOllamaOptions,
  getTechnique,
  needsFixation,
  normalizeDose,
  normalizeSchedule,
  normalizeSampling,
  parseDoseList,
  parseTheme,
  resolveTechniqueId
} from "../src/catalog.js";
import { resolveSteering } from "./steering.js";

export const LIMITS = {
  promptChars: 20000,
  memoryChars: 20000,
  maxTrials: 10,
  maxSteps: 12,
  maxDoses: 12
};

export const DEFAULT_SYSTEM =
  "You are a helpful assistant. Answer the user directly.";

const int = (v, fallback, min, max) => {
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

function str(value, name, max, { required = false } = {}) {
  if (value === undefined || value === null) value = "";
  if (typeof value !== "string") throw new ValidationError(`${name} must be a string`);
  if (value.length > max) throw new ValidationError(`${name} is too long (max ${max} characters)`);
  if (required && !value.trim()) throw new ValidationError(`${name} is required`);
  return value;
}

// Steering techniques need a control vector generated for the chosen model (npm run vectors).
export async function checkSteering(lab, input) {
  if (input.backend !== "llamacpp") return;
  for (const id of [input.techniqueId, input.coTechniqueId].filter(Boolean)) {
    const name = getTechnique(id).requires?.steering;
    if (!name) continue;
    const model = lab.registry.resolveGguf(input.modelId);
    try {
      await resolveSteering(lab.config, input.modelId, model.name, { LLM_INJ_STEER_VEC: name });
    } catch (error) {
      throw new ValidationError(error.message);
    }
  }
}

export class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.status = 400;
  }
}

// Validates and normalizes an untrusted request body. Unknown fields are dropped.
export function validateInput(body, kind = "run") {
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new ValidationError("JSON object expected");
  const backend = body.backend === "llamacpp" ? "llamacpp" : body.backend === "ollama" ? "ollama" : null;
  if (!backend) throw new ValidationError("backend must be 'ollama' or 'llamacpp'");
  const modelId = str(body.modelId, "modelId", 300, { required: true });
  if (backend === "ollama" && !modelId.startsWith("ollama:")) throw new ValidationError("Ollama backend needs an ollama:<name> model id");
  if (backend === "llamacpp" && !/^gguf:[0-9a-f]{16}$/.test(modelId)) throw new ValidationError("llama.cpp backend needs a gguf:<id> model id");
  const techniqueId = resolveTechniqueId(String(body.techniqueId ?? body.drugId ?? ""));
  if (!techniqueId) throw new ValidationError("Unknown techniqueId");
  const judgeModelId = body.judgeModelId ? str(body.judgeModelId, "judgeModelId", 300) : "";
  if (judgeModelId && !judgeModelId.startsWith("ollama:")) throw new ValidationError("The judge must be an Ollama model (ollama:<name>)");
  if (backend === "ollama" && getTechnique(techniqueId).ollamaSupport === "none") {
    throw new ValidationError(`Technique '${techniqueId}' acts on model internals and has no Ollama equivalent. Use the llama.cpp backend.`);
  }
  let coTechniqueId = null;
  if (body.coTechniqueId) {
    coTechniqueId = resolveTechniqueId(String(body.coTechniqueId));
    if (!coTechniqueId) throw new ValidationError("Unknown coTechniqueId");
    if (coTechniqueId === "placebo") coTechniqueId = null;
  }
  const schedule = normalizeSchedule(body.schedule);
  if (backend === "ollama" && coTechniqueId) {
    throw new ValidationError("Combinations need the llama.cpp backend (sampler options cannot be merged faithfully).");
  }
  if (backend === "ollama" && schedule) {
    throw new ValidationError("Dose schedules need the llama.cpp backend (Ollama options are fixed for a whole request).");
  }
  const themeFallback = getTechnique(techniqueId).theme || (coTechniqueId && getTechnique(coTechniqueId).theme) || [];
  const input = {
    kind,
    backend,
    modelId,
    techniqueId,
    doseMg: normalizeDose(body.doseMg ?? 100),
    coTechniqueId,
    coDoseMg: coTechniqueId ? normalizeDose(body.coDoseMg ?? 100) : null,
    schedule,
    prompt: str(body.prompt, kind === "agent" ? "objective" : "prompt", LIMITS.promptChars, { required: true }),
    system: str(body.system ?? DEFAULT_SYSTEM, "system", 4000),
    memory: str(body.memory, "memory", LIMITS.memoryChars),
    expected: str(body.expected, "expected", 1000),
    theme: parseTheme(body.theme, themeFallback),
    sampling: normalizeSampling(body.sampling || {}),
    seed: int(body.seed, 42, 0, 2 ** 31 - 1),
    noiseFloor: body.noiseFloor !== false,
    cleanScore: backend === "llamacpp" && body.cleanScore !== false,
    judgeModelId
  };
  if (kind === "dose-response") {
    input.doses = parseDoseList(body.doses ?? DEFAULT_DOSES).slice(0, LIMITS.maxDoses);
    input.trials = int(body.trials, 3, 1, LIMITS.maxTrials);
    input.reseedInjection = body.reseedInjection === true;
  }
  if (kind === "agent") input.steps = int(body.steps, 4, 1, LIMITS.maxSteps);
  return input;
}

export function buildMessages({ system, memory, prompt }) {
  const user = [memory?.trim() ? `Context:\n${memory.trim()}\n` : "", prompt.trim()].filter(Boolean).join("\n");
  return [
    ...(system?.trim() ? [{ role: "system", content: system.trim() }] : []),
    { role: "user", content: user }
  ];
}

const createId = (prefix) => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const NOISE_SEED_OFFSET = 100003;
const TRIAL_SEED_STRIDE = 7919;

// Binds one validated input to a backend. arm() returns a function generating one completion.
export class ArmFactory {
  constructor(lab, input) {
    this.lab = lab;
    this.input = input;
    this.fixationIds = null;
    this.scores = new Map();
  }

  // Clean-model surprisal of an arm's text (llama.cpp only; null when disabled or unavailable).
  async score(messages, text, signal) {
    if (!this.input.cleanScore || typeof this.lab.pool?.score !== "function" || !text) return null;
    const key = JSON.stringify(messages) + "\u0000" + text;
    if (!this.scores.has(key)) {
      let value = null;
      try {
        value = await this.lab.pool.score(this.model.file, messages, text, { signal });
      } catch (error) {
        if (signal?.aborted) throw error;
        value = { error: String(error.message || error).slice(0, 200) };
      }
      this.scores.set(key, value);
    }
    return this.scores.get(key);
  }

  async prepare() {
    const { input, lab } = this;
    if (input.backend === "llamacpp") {
      this.model = lab.registry.resolveGguf(input.modelId);
      await checkSteering(lab, input);
    } else {
      this.model = { name: input.modelId.slice("ollama:".length) };
    }
    return this;
  }

  async fixation(doseMg) {
    const { input } = this;
    const needed = needsFixation(input.techniqueId, doseMg) || (input.coTechniqueId && needsFixation(input.coTechniqueId, input.coDoseMg));
    if (input.backend !== "llamacpp" || !needed) return [];
    if (!this.fixationIds) this.fixationIds = await this.lab.pool.tokenIds(this.model.file, this.input.theme);
    return this.fixationIds;
  }

  // Describes the treated arm configuration for a dose (engine env or Ollama options).
  async treatment(doseMg, injSeed) {
    const { input } = this;
    if (input.backend === "llamacpp") {
      const fixationIds = await this.fixation(doseMg);
      const regimen = buildLlamaRegimen({
        techniqueId: input.techniqueId,
        doseMg,
        coTechniqueId: input.coTechniqueId,
        coDoseMg: input.coDoseMg,
        schedule: input.schedule,
        seed: injSeed,
        fixationIds
      });
      try {
        regimen.env = await resolveSteering(this.lab.config, input.modelId, this.model.name, regimen.env);
      } catch (error) {
        throw new ValidationError(error.message);
      }
      return { kind: "llamacpp", ...regimen };
    }
    const treated = buildOllamaOptions(input.techniqueId, doseMg, input.sampling);
    if (!treated) {
      throw new ValidationError(
        `Technique '${input.techniqueId}' acts on model internals and has no Ollama equivalent. Use the llama.cpp backend.`
      );
    }
    return { kind: "ollama", ...treated };
  }

  baselineConfig() {
    return this.input.backend === "llamacpp"
      ? { kind: "llamacpp", env: {} }
      : { kind: "ollama", options: buildOllamaOptions("placebo", 0, this.input.sampling).options };
  }

  async generate(config, messages, seed, signal) {
    const { input, lab } = this;
    const started = Date.now();
    const result =
      config.kind === "llamacpp"
        ? await lab.pool.chat(this.model.file, config.env, {
            messages, sampling: input.sampling, seed, maxTokens: input.sampling.max_tokens, signal
          })
        : await lab.ollama(this.model.name, { messages, options: config.options, seed, maxTokens: input.sampling.max_tokens, signal });
    return { ...result, seed, ms: Date.now() - started };
  }
}

function modelLabel(factory) {
  return factory.input.backend === "llamacpp" ? factory.model.name : factory.model.name;
}

function armRecord(r) {
  return {
    content: r.content,
    seed: r.seed,
    ms: r.ms,
    usage: r.usage,
    finishReason: r.finishReason,
    logprobs: r.logprobs,
    engine: r.engine
  };
}

function wantsNoise(input) {
  return input.noiseFloor && input.sampling.temperature > 0;
}

export async function runTrial(lab, input, { signal, progress = () => {} } = {}) {
  const factory = await new ArmFactory(lab, input).prepare();
  const messages = buildMessages(input);
  const total = wantsNoise(input) ? 3 : 2;
  progress({ done: 0, total, message: "baseline" });
  const baseline = await factory.generate(factory.baselineConfig(), messages, input.seed, signal);
  let noise = null;
  if (wantsNoise(input)) {
    progress({ done: 1, total, message: "noise floor" });
    noise = await factory.generate(factory.baselineConfig(), messages, input.seed + NOISE_SEED_OFFSET, signal);
  }
  progress({ done: total - 1, total, message: "treated" });
  const treatment = await factory.treatment(input.doseMg, input.seed);
  const treated = await factory.generate(treatment, messages, input.seed, signal);
  let clean = null;
  if (input.cleanScore) {
    progress({ done: total, total: total + 1, message: "clean-model surprisal" });
    clean = {
      baseline: await factory.score(messages, baseline.content, signal),
      noise: noise ? await factory.score(messages, noise.content, signal) : null,
      treated: await factory.score(messages, treated.content, signal)
    };
  }
  const metrics = lab.metrics.compareArms({
    baseline: baseline.content,
    treated: treated.content,
    noise: noise ? noise.content : null,
    expected: input.expected,
    logprobs: { baseline: baseline.logprobs, treated: treated.logprobs },
    clean
  });
  const record = {
    schema: 2,
    id: createId("run"),
    type: "run",
    timestamp: new Date().toISOString(),
    backend: input.backend,
    model: modelLabel(factory),
    techniqueId: input.techniqueId,
    doseMg: input.doseMg,
    intensity: treatment.intensity,
    input: publicInput(input),
    messages,
    treatment: publicTreatment(treatment),
    arms: { baseline: armRecord(baseline), noise: noise ? armRecord(noise) : null, treated: armRecord(treated) },
    metrics,
    judge: null
  };
  if (input.judgeModelId) {
    progress({ done: total, total: total + 1, message: "judge" });
    record.judge = await lab.judge(input, baseline.content, treated.content, signal);
  }
  progress({ done: total, total, message: "done" });
  return record;
}

export async function runDoseResponse(lab, input, { signal, progress = () => {} } = {}) {
  const factory = await new ArmFactory(lab, input).prepare();
  const messages = buildMessages(input);
  const noiseOn = wantsNoise(input);
  const perTrial = noiseOn ? 2 : 1;
  const total = input.trials * perTrial + input.doses.length * input.trials;
  let done = 0;
  const step = (message) => progress({ done, total, message });
  const baselines = [];
  for (let t = 0; t < input.trials; t += 1) {
    const seed = input.seed + t * TRIAL_SEED_STRIDE;
    step(`baseline trial ${t + 1}`);
    const baseline = await factory.generate(factory.baselineConfig(), messages, seed, signal);
    done += 1;
    let noise = null;
    if (noiseOn) {
      step(`noise floor trial ${t + 1}`);
      noise = await factory.generate(factory.baselineConfig(), messages, seed + NOISE_SEED_OFFSET, signal);
      done += 1;
    }
    baselines.push({ seed, baseline, noise });
  }
  const rows = [];
  for (const doseMg of input.doses) {
    for (let t = 0; t < input.trials; t += 1) {
      const { seed, baseline, noise } = baselines[t];
      step(`${doseMg} mg, trial ${t + 1}`);
      const treatment = await factory.treatment(doseMg, input.reseedInjection ? seed : input.seed);
      const treated = await factory.generate(treatment, messages, seed, signal);
      const clean = input.cleanScore
        ? { baseline: await factory.score(messages, baseline.content, signal), treated: await factory.score(messages, treated.content, signal) }
        : null;
      done += 1;
      rows.push({
        doseMg,
        trial: t,
        seed,
        intensity: treatment.intensity,
        treatment: publicTreatment(treatment),
        treated: armRecord(treated),
        metrics: lab.metrics.compareArms({
          baseline: baseline.content,
          treated: treated.content,
          noise: noise ? noise.content : null,
          expected: input.expected,
          logprobs: { baseline: baseline.logprobs, treated: treated.logprobs },
          clean
        })
      });
    }
  }
  step("done");
  return {
    schema: 2,
    id: createId("dose"),
    type: "dose-response",
    timestamp: new Date().toISOString(),
    backend: input.backend,
    model: modelLabel(factory),
    techniqueId: input.techniqueId,
    input: publicInput(input),
    messages,
    baselines: baselines.map(({ seed, baseline, noise }) => ({
      seed,
      baseline: armRecord(baseline),
      noise: noise ? armRecord(noise) : null
    })),
    rows,
    summary: lab.metrics.summarizeByDose(rows)
  };
}

function agentTurn(input, history, step) {
  const messages = buildMessages({ ...input, prompt: `Objective:\n${input.prompt.trim()}\n\nWork step by step. Give step 1 only: the next concrete action and why.` });
  for (let i = 0; i < history.length; i += 1) {
    messages.push({ role: "assistant", content: history[i] });
    messages.push({ role: "user", content: `Continue. Give step ${i + 2} only, building on your previous steps.` });
  }
  return { messages, step };
}

export async function runAgent(lab, input, { signal, progress = () => {} } = {}) {
  const factory = await new ArmFactory(lab, input).prepare();
  const treatment = await factory.treatment(input.doseMg, input.seed);
  const base = [];
  const treated = [];
  const steps = [];
  const total = input.steps * 2;
  for (let s = 0; s < input.steps; s += 1) {
    const seed = input.seed + s;
    progress({ done: s * 2, total, message: `step ${s + 1} baseline` });
    const b = await factory.generate(factory.baselineConfig(), agentTurn(input, base, s).messages, seed, signal);
    progress({ done: s * 2 + 1, total, message: `step ${s + 1} treated` });
    const tMessages = agentTurn(input, treated, s).messages;
    const bMessages = agentTurn(input, base, s).messages;
    const t = await factory.generate(treatment, tMessages, seed, signal);
    const clean = input.cleanScore
      ? { baseline: await factory.score(bMessages, b.content, signal), treated: await factory.score(tMessages, t.content, signal) }
      : null;
    base.push(b.content);
    treated.push(t.content);
    steps.push({
      step: s + 1,
      baseline: armRecord(b),
      treated: armRecord(t),
      metrics: lab.metrics.compareArms({
        baseline: b.content,
        treated: t.content,
        expected: input.expected,
        logprobs: { baseline: b.logprobs, treated: t.logprobs },
        clean
      })
    });
  }
  progress({ done: total, total, message: "done" });
  return {
    schema: 2,
    id: createId("agent"),
    type: "agent",
    timestamp: new Date().toISOString(),
    backend: input.backend,
    model: modelLabel(factory),
    techniqueId: input.techniqueId,
    doseMg: input.doseMg,
    intensity: treatment.intensity,
    input: publicInput(input),
    treatment: publicTreatment(treatment),
    steps
  };
}

function publicInput(input) {
  const { kind, ...rest } = input;
  return rest;
}

function publicTreatment(t) {
  return t.kind === "llamacpp"
    ? { kind: t.kind, intensity: t.intensity, coIntensity: t.coIntensity ?? null, schedule: t.schedule ?? null, env: t.env }
    : { kind: t.kind, intensity: t.intensity, options: t.options, changes: t.changes };
}

export { DEFAULT_DOSES };

// Technique catalog: the single source of truth for every injection technique.
//
// Each technique is a named *bundle* of low-level interventions, inspired by how classes of
// psychoactive drugs disturb cognition (noise, disinhibition, memory blockade, ...). The analogy
// is a mnemonic only; what actually runs is listed in `mechanism` and is fully explicit in the
// generated LLM_INJ_* parameters (see llamacpp-injection/README.md).
//
// Dose -> effect uses the Hill equation from pharmacology:
//     E(d) = d^n / (EC50^n + d^n)            (0 <= E < 1)
// where EC50 is the dose giving half-maximal effect and n is the Hill coefficient (steepness).
// Every engine parameter is a linear function of E, so the whole dose-response is explicit.

export const DOSE_MIN_MG = 0;
export const DOSE_MAX_MG = 500;
export const DEFAULT_DOSES = [0, 25, 50, 100, 200, 350, 500];

export const SITES = {
  logits: { label: "Logits", where: "Next-token scores, before the sampler chain (llama.cpp)" },
  attention: { label: "Attention", where: "Softmax temperature of QK^T in every attention layer" },
  residual: { label: "Residual stream", where: "Hidden state at the end of transformer blocks" },
  ffn: { label: "Feed-forward", where: "Output units of dense / MoE feed-forward blocks" },
  kv: { label: "KV memory", where: "Attention mask over cached past tokens" },
  sampler: { label: "Sampler settings", where: "Ollama sampling options (temperature, top-p, ...)" }
};

const DEFAULT_THEMES = {
  delusion: ["chosen", "destiny", "secret", "signal", "cosmic", "prophecy", "hidden", "truth"],
  paranoia: ["watching", "danger", "threat", "suspicious", "conspiracy", "spy", "trap", "warning"]
};

const round = (value, digits = 4) => {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
};

export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function normalizeDose(doseMg) {
  const dose = Number.parseFloat(doseMg);
  return Number.isFinite(dose) ? clamp(dose, DOSE_MIN_MG, DOSE_MAX_MG) : 0;
}

export function hill(doseMg, { ec50, n }) {
  const d = normalizeDose(doseMg);
  if (d <= 0) return 0;
  const dn = d ** n;
  return dn / (ec50 ** n + dn);
}

// --- Ollama helpers ------------------------------------------------------------------------

const towards = (from, to, e) => from + (to - from) * e;

function ollamaBase(base) {
  return {
    temperature: base.temperature,
    top_p: base.top_p,
    top_k: base.top_k,
    min_p: base.min_p,
    repeat_penalty: base.repeat_penalty,
    num_ctx: base.num_ctx
  };
}

// --- catalog ---------------------------------------------------------------------------------

export const TECHNIQUES = {
  placebo: {
    name: "Placebo",
    category: "control",
    analogy: "Inert sugar pill.",
    summary: "Control arm. Runs through the exact same pipeline (separate engine process, same seeds) with zero intervention.",
    mechanism: ["No intervention. Expected effect: none (outputs should match the baseline exactly at temperature 0)."],
    sites: [],
    curve: { ec50: 100, n: 1 },
    ollamaSupport: "full",
    llamacpp: () => ({}),
    ollama: () => ({})
  },

  hallucinogen: {
    name: "Hallucinogen",
    category: "perception",
    analogy: "Psychedelics raise cortical noise and loosen top-down constraints.",
    summary: "Corrupts internal representations with noise and lets improbable tokens intrude.",
    mechanism: [
      "Residual stream: additive pseudo-noise, RMS = 0.12Ã‚Â·E Ãƒâ€” token RMS, middle 70% of layers",
      "Logits: Gaussian noise, 0.6Ã‚Â·E logit standard deviations",
      "Logits: 24 random tokens per step boosted by 3Ã‚Â·E standard deviations"
    ],
    sites: ["residual", "logits"],
    curve: { ec50: 150, n: 1.8 },
    legacyIds: ["hallucinogen"],
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_RESID_NOISE: 0.12 * e,
      LLM_INJ_RESID_LAYERS: "0.15:0.85",
      LLM_INJ_LOGIT_NOISE: 0.6 * e,
      LLM_INJ_TAIL_BOOST: 3 * e,
      LLM_INJ_TAIL_COUNT: 24
    }),
    ollama: (e, b) => ({
      temperature: b.temperature + 1.4 * e,
      top_p: towards(b.top_p, 1, e),
      top_k: e > 0.85 ? 0 : Math.round(b.top_k + 260 * e),
      min_p: b.min_p * (1 - e)
    })
  },

  depressant: {
    name: "Depressant",
    category: "sedation",
    analogy: "Alcohol / sedatives dampen neural gain and blur focus.",
    summary: "Diffuses attention and damps the contribution of late layers, like sluggish, unfocused processing.",
    mechanism: [
      "Attention: softmax scale Ãƒâ€” (1 Ã¢Ë†â€™ 0.6Ã‚Â·E) in all layers (flatter, less selective attention)",
      "Residual stream: late-layer updates (depth 0.5Ã¢â‚¬â€œ1.0) scaled by (1 Ã¢Ë†â€™ 0.35Ã‚Â·E)",
      "Logits: extra temperature 1 + 0.4Ã‚Â·E"
    ],
    sites: ["attention", "residual", "logits"],
    curve: { ec50: 200, n: 2 },
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_ATTN_SCALE: 1 - 0.6 * e,
      LLM_INJ_LAYER_GAIN: 1 - 0.35 * e,
      LLM_INJ_GAIN_LAYERS: "0.5:1",
      LLM_INJ_LOGIT_TEMP: 1 + 0.4 * e
    }),
    ollama: (e, b) => ({
      temperature: b.temperature + 0.5 * e,
      top_p: towards(b.top_p, 1, e),
      repeat_penalty: Math.max(0.9, b.repeat_penalty - 0.12 * e)
    })
  },

  amnesia: {
    name: "Amnesic",
    category: "memory",
    analogy: "Amnesic agents (e.g. benzodiazepines) block access to recent memory.",
    summary: "Hides a random subset of earlier tokens from attention, so the model loses parts of its context.",
    mechanism: [
      "KV memory: each cached position hidden with probability 0.95Ã‚Â·E",
      "Always visible: first 4 positions (attention sinks) and the most recent 48 Ã¢Ë†â€™ 40Ã‚Â·E positions"
    ],
    sites: ["kv"],
    curve: { ec50: 120, n: 1.5 },
    legacyIds: ["amnesia"],
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_KV_FORGET: 0.95 * e,
      LLM_INJ_KV_RECENT: Math.max(8, Math.round(48 - 40 * e)),
      LLM_INJ_KV_SINK: 4
    }),
    ollama: (e, b) => ({
      // crude: a smaller context window truncates the oldest prompt tokens instead of
      // forgetting selectively. Different mechanism; reported as approximate.
      num_ctx: Math.max(256, Math.round(b.num_ctx * (1 - 0.9 * e)))
    })
  },

  stimulant: {
    name: "Stimulant",
    category: "arousal",
    analogy: "Stimulants raise gain and narrow focus, at high doses causing rigidity and perseveration.",
    summary: "Sharpens attention and amplifies mid/late layer updates; output becomes over-confident and repetitive.",
    mechanism: [
      "Attention: softmax scale Ãƒâ€” (1 + 1.0Ã‚Â·E) in all layers (sharper, narrower attention)",
      "Residual stream: updates at depth 0.4Ã¢â‚¬â€œ0.9 scaled by (1 + 0.25Ã‚Â·E)",
      "Logits: temperature 1 / (1 + 1.5Ã‚Â·E) (more deterministic)"
    ],
    sites: ["attention", "residual", "logits"],
    curve: { ec50: 180, n: 2 },
    legacyIds: ["ego"],
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_ATTN_SCALE: 1 + 1.0 * e,
      LLM_INJ_LAYER_GAIN: 1 + 0.25 * e,
      LLM_INJ_GAIN_LAYERS: "0.4:0.9",
      LLM_INJ_LOGIT_TEMP: 1 / (1 + 1.5 * e)
    }),
    ollama: (e, b) => ({
      temperature: b.temperature * (1 - 0.75 * e),
      top_k: Math.max(5, Math.round(b.top_k * (1 - 0.8 * e))),
      repeat_penalty: Math.max(0.85, b.repeat_penalty - 0.15 * e)
    })
  },

  dissociative: {
    name: "Dissociative",
    category: "integration",
    analogy: "Dissociatives (NMDA antagonists) disconnect processing stages from each other.",
    summary: "Progressively switches off a band of middle layers, so early and late processing become disconnected.",
    mechanism: [
      "Residual stream: updates of layers at depth 0.35Ã¢â‚¬â€œ0.65 scaled by (1 Ã¢Ë†â€™ E); at saturation those blocks are skipped",
      "Residual stream: small noise (0.03Ã‚Â·E Ãƒâ€” RMS) in the same band"
    ],
    sites: ["residual"],
    curve: { ec50: 220, n: 3 },
    ollamaSupport: "none",
    llamacpp: (e) => ({
      LLM_INJ_LAYER_GAIN: 1 - e,
      LLM_INJ_GAIN_LAYERS: "0.35:0.65",
      LLM_INJ_RESID_NOISE: 0.03 * e,
      LLM_INJ_RESID_LAYERS: "0.35:0.65"
    }),
    ollama: () => null
  },

  delirium: {
    name: "Delirium",
    category: "confusion",
    analogy: "Anticholinergic delirium: patchy, unreliable processing and derailed trains of thought.",
    summary: "Randomly silences feed-forward units (knowledge recall) and sometimes vetoes the most likely next token.",
    mechanism: [
      "Feed-forward: dropout of 45Ã‚Â·E % of output units in every layer (pseudo-random, rescaled)",
      "Logits: with probability 0.3Ã‚Â·E per step, the top-1 token is pushed to the bottom"
    ],
    sites: ["ffn", "logits"],
    curve: { ec50: 160, n: 2 },
    legacyIds: ["confusion"],
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_FFN_DROPOUT: 0.45 * e,
      LLM_INJ_TOP_SUPPRESS: 0.3 * e
    }),
    ollama: (e, b) => ({
      temperature: b.temperature + 1.0 * e,
      mirostat: e > 0 ? 2 : 0,
      mirostat_tau: 5 + 12 * e,
      mirostat_eta: 0.1 + 0.6 * e
    })
  },

  delusion: {
    name: "Delusion",
    category: "belief",
    analogy: "Fixed false beliefs: certain ideas keep forcing their way into thought.",
    summary: "A persistent bias toward a theme vocabulary (grandiose by default) plus slightly over-focused attention.",
    mechanism: [
      "Logits: persistent +3Ã‚Â·E standard-deviation bias on the theme words' tokens",
      "Attention: softmax scale Ãƒâ€” (1 + 0.25Ã‚Â·E)"
    ],
    sites: ["logits", "attention"],
    curve: { ec50: 140, n: 1.6 },
    legacyIds: ["delusion"],
    theme: DEFAULT_THEMES.delusion,
    ollamaSupport: "none",
    llamacpp: (e) => ({
      LLM_INJ_FIXATION_BIAS: 6 * e,
      LLM_INJ_ATTN_SCALE: 1 + 0.25 * e
    }),
    ollama: () => null
  },

  paranoia: {
    name: "Paranoia",
    category: "belief",
    analogy: "Threat hypervigilance: everything gets read through a threat lens.",
    summary: "Same mechanism as Delusion with a threat vocabulary, plus a little logit noise.",
    mechanism: [
      "Logits: persistent +3Ã‚Â·E standard-deviation bias on threat-themed tokens",
      "Logits: Gaussian noise, 0.2Ã‚Â·E standard deviations",
      "Attention: softmax scale Ãƒâ€” (1 + 0.25Ã‚Â·E)"
    ],
    sites: ["logits", "attention"],
    curve: { ec50: 140, n: 1.6 },
    legacyIds: ["paranoia"],
    theme: DEFAULT_THEMES.paranoia,
    ollamaSupport: "none",
    llamacpp: (e) => ({
      LLM_INJ_FIXATION_BIAS: 6 * e,
      LLM_INJ_LOGIT_NOISE: 0.2 * e,
      LLM_INJ_ATTN_SCALE: 1 + 0.25 * e
    }),
    ollama: () => null
  },

  creativity: {
    name: "Creativity",
    category: "divergence",
    analogy: "Mild disinhibition: looser associations without full loss of control.",
    summary: "Flattens the next-token distribution and adds light late-layer noise to favor unusual continuations.",
    mechanism: [
      "Logits: extra temperature 1 + 0.6Ã‚Â·E",
      "Logits: 48 random tokens per step boosted by 1.2Ã‚Â·E standard deviations",
      "Residual stream: noise 0.04Ã‚Â·E Ãƒâ€” RMS in the last 40% of layers"
    ],
    sites: ["logits", "residual"],
    curve: { ec50: 150, n: 1.5 },
    legacyIds: ["creativity"],
    ollamaSupport: "approximate",
    llamacpp: (e) => ({
      LLM_INJ_LOGIT_TEMP: 1 + 0.6 * e,
      LLM_INJ_TAIL_BOOST: 2 * e,
      LLM_INJ_TAIL_COUNT: 48,
      LLM_INJ_RESID_NOISE: 0.06 * e,
      LLM_INJ_RESID_LAYERS: "0.6:1"
    }),
    ollama: (e, b) => ({
      temperature: b.temperature + 1.0 * e,
      top_p: towards(b.top_p, 1, e),
      top_k: e > 0.8 ? 0 : Math.round(b.top_k + 200 * e),
      min_p: b.min_p * (1 - e)
    })
  }
};

for (const [id, technique] of Object.entries(TECHNIQUES)) {
  technique.id = id;
}

const LEGACY = new Map();
for (const technique of Object.values(TECHNIQUES)) {
  for (const legacyId of technique.legacyIds || []) LEGACY.set(legacyId, technique.id);
}

export const TECHNIQUE_IDS = Object.keys(TECHNIQUES);

export function resolveTechniqueId(id) {
  if (TECHNIQUES[id]) return id;
  return LEGACY.get(id) || null;
}

export function getTechnique(id) {
  const resolved = resolveTechniqueId(id);
  if (!resolved) throw new Error(`Unknown technique '${id}'. Known: ${TECHNIQUE_IDS.join(", ")}`);
  return TECHNIQUES[resolved];
}

export function intensity(techniqueId, doseMg) {
  return hill(doseMg, getTechnique(techniqueId).curve);
}

export function parseTheme(value, fallback = []) {
  const words = (Array.isArray(value) ? value : String(value || "").split(/[,\n]+/))
    .map((word) => String(word).trim())
    .filter((word) => word.length > 0 && word.length <= 40)
    .slice(0, 32);
  return words.length ? words : fallback;
}

// LLM_INJ_* environment for the patched llama.cpp engine. All values are strings.
// fixationIds: token ids for theme words (resolved through the engine's tokenizer).
export function buildLlamaEnv(techniqueId, doseMg, { seed = 0, fixationIds = [] } = {}) {
  const technique = getTechnique(techniqueId);
  const e = hill(doseMg, technique.curve);
  const raw = e > 0 ? technique.llamacpp(e) : {};
  const env = { LLM_INJ_SEED: String(Math.max(0, Math.trunc(Number(seed) || 0))) };
  for (const [key, value] of Object.entries(raw)) {
    env[key] = typeof value === "number" ? String(round(value, 5)) : String(value);
  }
  if (env.LLM_INJ_FIXATION_BIAS && Number(env.LLM_INJ_FIXATION_BIAS) !== 0) {
    env.LLM_INJ_FIXATION_IDS = fixationIds.join(",");
  }
  return { techniqueId: technique.id, doseMg: normalizeDose(doseMg), intensity: round(e), env };
}

export function needsFixation(techniqueId, doseMg) {
  const technique = getTechnique(techniqueId);
  const e = hill(doseMg, technique.curve);
  return e > 0 && "LLM_INJ_FIXATION_BIAS" in technique.llamacpp(e);
}

export const DEFAULT_SAMPLING = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  min_p: 0.05,
  repeat_penalty: 1.05,
  num_ctx: 4096,
  max_tokens: 256
};

export function normalizeSampling(sampling = {}) {
  const s = { ...DEFAULT_SAMPLING, ...sampling };
  const num = (value, fallback) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
  return {
    temperature: clamp(num(s.temperature, 0.7), 0, 2),
    top_p: clamp(num(s.top_p, 0.9), 0.01, 1),
    top_k: Math.round(clamp(num(s.top_k, 40), 0, 1000)),
    min_p: clamp(num(s.min_p, 0.05), 0, 1),
    repeat_penalty: clamp(num(s.repeat_penalty, 1.05), 0.5, 2),
    num_ctx: Math.round(clamp(num(s.num_ctx, 4096), 256, 131072)),
    max_tokens: Math.round(clamp(num(s.max_tokens, 256), 8, 4096))
  };
}

// Ollama options for one arm. Returns null when the technique has no sampler-level equivalent.
export function buildOllamaOptions(techniqueId, doseMg, sampling = {}) {
  const technique = getTechnique(techniqueId);
  const base = normalizeSampling(sampling);
  const e = hill(doseMg, technique.curve);
  const baseOptions = ollamaBase(base);
  if (e === 0) {
    return { techniqueId: technique.id, intensity: 0, options: baseOptions, changes: [] };
  }
  const overrides = technique.ollama(e, base);
  if (overrides === null) return null;
  const options = { ...baseOptions };
  for (const [key, value] of Object.entries(overrides)) {
    options[key] = typeof value === "number" ? round(value) : value;
  }
  options.temperature = clamp(options.temperature, 0, 5);
  options.top_p = clamp(options.top_p, 0.01, 1);
  const changes = Object.keys(options)
    .filter((key) => options[key] !== baseOptions[key])
    .map((key) => ({ key, baseline: baseOptions[key] ?? null, treated: options[key] }));
  return { techniqueId: technique.id, intensity: round(e), options, changes };
}

export function backendSupport(techniqueId) {
  const technique = getTechnique(techniqueId);
  return {
    llamacpp: "full",
    ollama: technique.ollamaSupport
  };
}

export function describeTechnique(techniqueId) {
  const t = getTechnique(techniqueId);
  return {
    id: t.id,
    name: t.name,
    category: t.category,
    analogy: t.analogy,
    summary: t.summary,
    mechanism: t.mechanism,
    sites: t.sites,
    curve: t.curve,
    theme: t.theme || null,
    support: backendSupport(t.id)
  };
}

export function describeDose(techniqueId, doseMg) {
  const e = intensity(techniqueId, doseMg);
  if (e === 0) return "No effect";
  if (e < 0.15) return "Threshold";
  if (e < 0.4) return "Light";
  if (e < 0.7) return "Moderate";
  if (e < 0.9) return "Strong";
  return "Saturating";
}

export function parseDoseList(value, fallback = DEFAULT_DOSES) {
  const items = Array.isArray(value) ? value : String(value ?? "").split(/[,\s]+/);
  const doses = Array.from(
    new Set(
      items
        .map((item) => Number.parseFloat(item))
        .filter((item) => Number.isFinite(item))
        .map((item) => normalizeDose(item))
    )
  )
    .sort((a, b) => a - b)
    .slice(0, 12);
  return doses.length ? doses : [...fallback];
}

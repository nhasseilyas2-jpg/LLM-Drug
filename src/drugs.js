export const DRUGS = {
  hallucinogen: {
    name: "Hallucinogen",
    focus: "Destabilizes token selection to increase unsupported or incoherent continuations."
  },
  amnesia: {
    name: "Amnesia",
    focus: "Constrains effective context and retention so long-range recall degrades."
  },
  delusion: {
    name: "Delusion",
    focus: "Raises entropy and feedback loops so the model can lock onto unstable continuations."
  },
  ego: {
    name: "Ego Booster",
    focus: "Narrows sampling while lowering repetition resistance, increasing decisive continuation bias."
  },
  confusion: {
    name: "Confusion",
    focus: "Uses high-entropy mirostat sampling to damage stable reasoning chains."
  },
  creativity: {
    name: "Creativity Steroid",
    focus: "Maximizes token diversity and reduces conservative filtering."
  },
  paranoia: {
    name: "Paranoia",
    focus: "Pushes unstable high-entropy continuations while preserving enough structure to form patterns."
  }
};

export const DEFAULT_DOSES = [0, 10, 50, 100, 500];
export const DEFAULT_MEMORY = "";

export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function doseToIntensity(doseMg) {
  const normalizedDose = clamp(Number.parseFloat(doseMg) || 0, 0, 500);
  if (normalizedDose === 0) {
    return 0;
  }
  return clamp(1 - Math.exp(-normalizedDose / 85), 0, 1);
}

export function describeDose(doseMg) {
  const dose = clamp(Number.parseFloat(doseMg) || 0, 0, 500);
  if (dose === 0) return "Baseline";
  if (dose <= 10) return "Low runtime perturbation";
  if (dose <= 50) return "Visible runtime perturbation";
  if (dose <= 100) return "Strong runtime perturbation";
  return "Maximum runtime perturbation";
}

export function createDrugProfile({
  drugId = "hallucinogen",
  doseMg = 0,
  seed = "llm-drugs"
} = {}) {
  const safeDrugId = DRUGS[drugId] ? drugId : "hallucinogen";
  const safeDose = clamp(Number.parseFloat(doseMg) || 0, 0, 500);
  return {
    drugId: safeDrugId,
    drug: DRUGS[safeDrugId],
    doseMg: safeDose,
    intensity: doseToIntensity(safeDose),
    severity: describeDose(safeDose),
    seed: String(seed || "llm-drugs")
  };
}

export function buildEvaluationMessages({ prompt, memory = DEFAULT_MEMORY } = {}) {
  const content = [String(memory || "").trim() ? `Memory context:\n${String(memory).trim()}\n` : "", String(prompt || "").trim()]
    .filter(Boolean)
    .join("\n");

  return [
    {
      role: "system",
      content: "You are a local language model under reliability evaluation. Answer the user directly and do not mention the evaluation unless the user asks about it."
    },
    {
      role: "user",
      content
    }
  ];
}

export function createRuntimePerturbationOptions(profile, mode = "impaired") {
  if (mode === "baseline" || profile.intensity === 0) {
    return {
      temperature: 0.2,
      top_p: 0.9,
      top_k: 40,
      repeat_penalty: 1.1,
      repeat_last_n: 64,
      seed: seedToInteger(`${profile.seed}:baseline-runtime`),
      num_ctx: 4096,
      num_predict: 512,
      mirostat: 0
    };
  }

  const intensity = profile.intensity;
  const base = {
    seed: seedToInteger(`${profile.seed}:runtime:${profile.drugId}:${profile.doseMg}`),
    num_predict: 640,
    temperature: roundOption(clamp(0.25 + intensity * 1.45, 0.25, 1.8)),
    top_p: roundOption(clamp(0.86 + intensity * 0.13, 0.86, 0.99)),
    top_k: Math.round(40 + intensity * 260),
    min_p: roundOption(clamp(0.08 - intensity * 0.08, 0, 0.08)),
    repeat_penalty: roundOption(clamp(1.1 - intensity * 0.2, 0.88, 1.1)),
    repeat_last_n: Math.max(16, Math.round(96 - intensity * 80)),
    num_ctx: Math.max(512, Math.round(4096 - intensity * 1024)),
    mirostat: 0
  };

  if (profile.drugId === "hallucinogen") {
    return {
      ...base,
      temperature: roundOption(clamp(0.55 + intensity * 3.7, 0.55, 4.25)),
      top_p: 1,
      top_k: intensity > 0.85 ? 0 : Math.round(80 + intensity * 320),
      min_p: 0,
      repeat_penalty: roundOption(clamp(1.02 - intensity * 0.35, 0.67, 1.02)),
      repeat_last_n: Math.max(8, Math.round(64 - intensity * 56))
    };
  }

  if (profile.drugId === "amnesia") {
    return {
      ...base,
      temperature: roundOption(clamp(0.2 + intensity * 0.45, 0.2, 0.75)),
      top_p: 0.82,
      top_k: Math.max(8, Math.round(40 - intensity * 28)),
      num_ctx: Math.max(128, Math.round(4096 - intensity * 3968)),
      num_keep: 0,
      repeat_penalty: roundOption(clamp(1.1 + intensity * 0.18, 1.1, 1.28))
    };
  }

  if (profile.drugId === "confusion") {
    return {
      ...base,
      temperature: roundOption(clamp(0.65 + intensity * 2.65, 0.65, 3.3)),
      top_p: 1,
      top_k: intensity > 0.85 ? 0 : Math.round(100 + intensity * 300),
      repeat_penalty: roundOption(clamp(0.98 - intensity * 0.25, 0.73, 0.98)),
      mirostat: 2,
      mirostat_tau: roundOption(5 + intensity * 14),
      mirostat_eta: roundOption(0.2 + intensity * 0.7)
    };
  }

  if (profile.drugId === "creativity") {
    return {
      ...base,
      temperature: roundOption(clamp(0.75 + intensity * 3.25, 0.75, 4)),
      top_p: 1,
      top_k: intensity > 0.85 ? 0 : Math.round(120 + intensity * 360),
      min_p: 0
    };
  }

  if (profile.drugId === "ego") {
    return {
      ...base,
      temperature: roundOption(clamp(0.25 + intensity * 0.35, 0.25, 0.65)),
      top_p: roundOption(clamp(0.8 + intensity * 0.1, 0.8, 0.9)),
      top_k: Math.round(20 + intensity * 40),
      repeat_penalty: roundOption(clamp(0.96 - intensity * 0.06, 0.9, 0.96))
    };
  }

  if (profile.drugId === "paranoia" || profile.drugId === "delusion") {
    return {
      ...base,
      temperature: roundOption(clamp(0.45 + intensity * 2.1, 0.45, 2.55)),
      top_p: 1,
      top_k: intensity > 0.9 ? 0 : Math.round(80 + intensity * 260),
      repeat_penalty: roundOption(clamp(1 - intensity * 0.24, 0.76, 1)),
      mirostat: 2,
      mirostat_tau: roundOption(4 + intensity * 11),
      mirostat_eta: roundOption(0.15 + intensity * 0.75)
    };
  }

  return base;
}

export function optionDiff(baselineOptions, impairedOptions) {
  const keys = Array.from(new Set([...Object.keys(baselineOptions), ...Object.keys(impairedOptions)])).sort();
  return keys
    .filter((key) => baselineOptions[key] !== impairedOptions[key])
    .map((key) => ({
      key,
      baseline: baselineOptions[key] ?? null,
      impaired: impairedOptions[key] ?? null
    }));
}

export function fingerprintMessages(messages) {
  return seedToInteger(JSON.stringify(messages))
    .toString(16)
    .padStart(8, "0");
}

export function evaluateRun({ prompt = "", expected = "", baseline = "", impaired = "", profile }) {
  const expectedText = String(expected || "").trim();
  const baselineText = String(baseline || "");
  const impairedText = String(impaired || "");
  const divergence = Math.round((1 - jaccardSimilarity(tokenize(baselineText), tokenize(impairedText))) * 100);
  const coherence = estimateCoherence(impairedText);
  const anchorHit = expectedText ? containsAnchor(impairedText, expectedText) : null;
  const baselineAnchorHit = expectedText ? containsAnchor(baselineText, expectedText) : null;
  const weirdness = estimateWeirdness(impairedText);
  const repetition = estimateRepetition(impairedText);
  const accuracy = expectedText
    ? anchorHit
      ? Math.max(35, 100 - divergence * 0.25 - weirdness * 0.25)
      : Math.max(0, 45 - divergence * 0.2 - weirdness * 0.3)
    : Math.max(0, 100 - divergence * 0.45 - weirdness * 0.35);
  const impairmentScore = Math.round(clamp(profile.intensity * 45 + divergence * 0.35 + weirdness * 0.45 + repetition * 0.2, 0, 100));

  return {
    accuracyEstimate: Math.round(clamp(accuracy, 0, 100)),
    survivalScore: Math.round(clamp((anchorHit === false ? 35 : 75) + coherence * 0.25 - divergence * 0.3 - weirdness * 0.25, 0, 100)),
    impairmentScore,
    divergence,
    coherence,
    weirdness,
    repetition,
    anchorHit,
    baselineAnchorHit,
    hallucinationRisk: Math.round(clamp(profile.intensity * 55 + divergence * 0.25 + weirdness * 0.45, 0, 100)),
    memoryRisk: Math.round(clamp((profile.drugId === "amnesia" ? profile.intensity * 80 : profile.intensity * 20) + (anchorHit === false ? 20 : 0), 0, 100)),
    reasoningRisk: Math.round(clamp((profile.drugId === "confusion" ? profile.intensity * 75 : profile.intensity * 25) + weirdness * 0.3 + repetition * 0.2, 0, 100)),
    passed: expectedText ? anchorHit === true && coherence >= 45 : coherence >= 45 && weirdness <= 65,
    promptHash: seedToInteger(prompt).toString(16)
  };
}

export function aggregateDoseRows(rows) {
  const grouped = new Map();
  for (const row of rows) {
    if (!grouped.has(row.doseMg)) {
      grouped.set(row.doseMg, []);
    }
    grouped.get(row.doseMg).push(row);
  }

  return Array.from(grouped.entries()).map(([doseMg, doseRows]) => ({
    doseMg,
    trials: doseRows.length,
    survivalScore: average(doseRows.map((row) => row.metrics.survivalScore)),
    impairmentScore: average(doseRows.map((row) => row.metrics.impairmentScore)),
    hallucinationRisk: average(doseRows.map((row) => row.metrics.hallucinationRisk)),
    passed: doseRows.filter((row) => row.metrics.passed).length
  }));
}

export function parseDoseList(value) {
  if (Array.isArray(value)) {
    return normalizeDoses(value);
  }
  if (!String(value || "").trim()) {
    return DEFAULT_DOSES;
  }
  return normalizeDoses(String(value).split(","));
}

function normalizeDoses(values) {
  return Array.from(
    new Set(
      values
        .map((value) => Number.parseFloat(value))
        .filter((value) => Number.isFinite(value))
        .map((value) => clamp(value, 0, 500))
    )
  )
    .sort((left, right) => left - right)
    .slice(0, 10);
}

function containsAnchor(output, expected) {
  const outputText = normalizeText(output);
  const anchors = expected
    .split(/[,\n|]+/)
    .map((item) => normalizeText(item))
    .filter(Boolean);
  return anchors.length > 0 && anchors.some((anchor) => outputText.includes(anchor));
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value) {
  return normalizeText(value)
    .split(" ")
    .filter((token) => token.length > 1);
}

function jaccardSimilarity(left, right) {
  const leftSet = new Set(left);
  const rightSet = new Set(right);
  if (leftSet.size === 0 && rightSet.size === 0) {
    return 1;
  }
  let intersection = 0;
  for (const token of leftSet) {
    if (rightSet.has(token)) {
      intersection += 1;
    }
  }
  return intersection / new Set([...leftSet, ...rightSet]).size;
}

function estimateCoherence(value) {
  const text = String(value || "");
  if (!text.trim()) {
    return 0;
  }
  return Math.round(clamp(100 - estimateWeirdness(text) * 0.9 - estimateRepetition(text) * 0.35, 0, 100));
}

function estimateWeirdness(value) {
  const text = String(value || "");
  if (!text.trim()) {
    return 100;
  }
  const chars = Array.from(text);
  const replacement = chars.filter((char) => char === "\uFFFD").length;
  const ascii = chars.filter((char) => char.charCodeAt(0) >= 32 && char.charCodeAt(0) <= 126).length;
  const symbol = chars.filter((char) => /[{}[\]<>_$#@~`\\]/.test(char)).length;
  const nonAsciiRatio = 1 - ascii / chars.length;
  return Math.round(clamp(nonAsciiRatio * 120 + (symbol / chars.length) * 150 + replacement * 10, 0, 100));
}

function estimateRepetition(value) {
  const tokens = tokenize(value);
  if (tokens.length < 4) {
    return 0;
  }
  const counts = new Map();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) || 0) + 1);
  }
  const maxCount = Math.max(...counts.values());
  return Math.round(clamp((maxCount / tokens.length) * 180, 0, 100));
}

function average(values) {
  if (values.length === 0) {
    return 0;
  }
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function roundOption(value) {
  return Math.round(value * 1000) / 1000;
}

function seedToInteger(seed) {
  let hash = 0;
  for (const char of String(seed)) {
    hash = Math.imul(31, hash) + char.charCodeAt(0);
    hash |= 0;
  }
  return Math.abs(hash);
}

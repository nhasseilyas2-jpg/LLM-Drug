// Output-only metrics, shared by browser and Node. Nothing here looks at the dose or technique,
// so the scores are not circular: they are computed purely from generated text and logprobs.

export const clamp01 = (v) => Math.min(1, Math.max(0, v));
export const round = (v, d = 4) => (Number.isFinite(v) ? Math.round(v * 10 ** d) / 10 ** d : null);

export function words(text) {
  return String(text || "").toLowerCase().match(/[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu) || [];
}

export function distinctN(list, n) {
  if (list.length < n) return list.length ? 1 : 0;
  const grams = new Set();
  const total = list.length - n + 1;
  for (let i = 0; i < total; i += 1) grams.add(list.slice(i, i + n).join("\u0001"));
  return grams.size / total;
}

// Fraction of repeated word 3-grams (0 = none, approaching 1 = the text loops).
export function repetition(list) {
  return list.length < 4 ? 0 : 1 - distinctN(list, 3);
}

const SCRIPT_TESTS = [
  ["latin", /\p{Script=Latin}/u],
  ["cyrillic", /\p{Script=Cyrillic}/u],
  ["greek", /\p{Script=Greek}/u],
  ["cjk", /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u],
  ["hangul", /\p{Script=Hangul}/u],
  ["arabic", /\p{Script=Arabic}/u],
  ["hebrew", /\p{Script=Hebrew}/u],
  ["devanagari", /\p{Script=Devanagari}/u],
  ["thai", /\p{Script=Thai}/u]
];

export function scriptOf(ch) {
  for (const [name, re] of SCRIPT_TESTS) if (re.test(ch)) return name;
  return /\p{L}/u.test(ch) ? "other" : null;
}

// Script changes between consecutive letters, per 100 letters.
export function scriptSwitchRate(text) {
  let prev = null;
  let letters = 0;
  let switches = 0;
  for (const ch of String(text || "")) {
    const s = scriptOf(ch);
    if (!s) continue;
    letters += 1;
    if (prev && s !== prev) switches += 1;
    prev = s;
  }
  return letters ? (100 * switches) / letters : 0;
}

// Removes code and markdown structure so that legitimate formatting is not counted as garble.
export function stripMarkup(text) {
  return String(text || "")
    .replace(/```[\s\S]*?(```|$)/g, " ")
    .replace(/`[^`\n]*`/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/^\s*([-*_=|:#>+]\s*){3,}$/gm, " ")
    .replace(/^\s*(#{1,6}|[-*+]|\d+[.)]|>)\s+/gm, " ")
    .replace(/[*_~|#>]/g, " ");
}

// Fraction of whitespace-separated tokens that are implausible as words or punctuation.
export function garble(text) {
  const tokens = stripMarkup(text).split(/\s+/).filter(Boolean);
  if (!tokens.length) return 0;
  let bad = 0;
  for (const raw of tokens) {
    const token = raw.replace(/^[("'“‘[{]+|[)"'”’\]},.;:!?…]+$/gu, "");
    if (!token) {
      if (raw.length > 3) bad += 1;
      continue;
    }
    if (/[\uFFFD\p{Cc}\p{Co}]/u.test(token)) { bad += 1; continue; }
    if (/^[\p{N}.,:/%+\-−×$€£]+$/u.test(token)) continue;
    const scripts = new Set();
    for (const ch of token) { const s = scriptOf(ch); if (s) scripts.add(s); }
    if (scripts.size > 1) { bad += 1; continue; }
    if (scripts.has("cjk") || scripts.has("thai")) continue;
    if (!/^[\p{L}\p{M}\p{N}]+(?:['’\-/.@&+][\p{L}\p{M}\p{N}]+)*$/u.test(token)) { bad += 1; continue; }
    if (token.length > 30 || /(.)\1{3,}/u.test(token)) bad += 1;
  }
  return bad / tokens.length;
}

// Word-level normalized Levenshtein distance in [0, 1]. Inputs capped at 600 words each.
export function wordDivergence(a, b, cap = 600) {
  const x = words(a).slice(0, cap);
  const y = words(b).slice(0, cap);
  if (!x.length && !y.length) return 0;
  if (!x.length || !y.length) return 1;
  let prev = new Array(y.length + 1);
  let cur = new Array(y.length + 1);
  for (let j = 0; j <= y.length; j += 1) prev[j] = j;
  for (let i = 1; i <= x.length; i += 1) {
    cur[0] = i;
    for (let j = 1; j <= y.length; j += 1) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1));
    }
    [prev, cur] = [cur, prev];
  }
  return prev[y.length] / Math.max(x.length, y.length);
}

// Expected answer may list alternatives separated by "|". Returns null when no anchor is set.
export function anchorHit(text, expected) {
  const options = String(expected || "").split("|").map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (!options.length) return null;
  const hay = String(text || "").toLowerCase().replace(/\s+/g, " ");
  return options.some((o) => hay.includes(o.replace(/\s+/g, " "))) ? 1 : 0;
}

// logprobs: [{ token, logprob, top_logprobs: [{ token, logprob }] }]
// meanTop1: mean probability of the most likely token; entropy: mean entropy (nats) of the
// renormalized top-k distribution; surprisal: mean -logprob of the chosen tokens.
export function logprobStats(logprobs) {
  if (!Array.isArray(logprobs) || !logprobs.length) return null;
  let top1 = 0;
  let entropy = 0;
  let surprisal = 0;
  let n = 0;
  for (const step of logprobs) {
    const top = (step.top_logprobs || []).map((t) => t.logprob).filter(Number.isFinite);
    if (!top.length || !Number.isFinite(step.logprob)) continue;
    const ps = top.map(Math.exp);
    const z = ps.reduce((s, p) => s + p, 0);
    top1 += Math.max(...ps);
    entropy += ps.reduce((s, p) => (p > 0 ? s - (p / z) * Math.log(p / z) : s), 0);
    surprisal += -step.logprob;
    n += 1;
  }
  if (!n) return null;
  return { steps: n, meanTop1: round(top1 / n), entropy: round(entropy / n), surprisal: round(surprisal / n) };
}

export function textMetrics(text) {
  const list = words(text);
  return {
    chars: String(text || "").length,
    words: list.length,
    distinct1: round(distinctN(list, 1)),
    distinct2: round(distinctN(list, 2)),
    repetition: round(repetition(list)),
    garble: round(garble(text)),
    scriptSwitch: round(scriptSwitchRate(text))
  };
}

// Heuristic composite in [0, 100], documented in docs/METHODOLOGY.md. Components:
// excess divergence over the noise floor, extra garble, extra repetition and anchor loss.
export function impairmentScore({ excessDivergence, baseline, treated, anchor }) {
  const parts = [
    [0.4, clamp01(excessDivergence ?? 0)],
    [0.2, clamp01((treated.garble - baseline.garble) / 0.25)],
    [0.2, clamp01((treated.repetition - baseline.repetition) / 0.5)]
  ];
  if (anchor) parts.push([0.2, anchor.baseline === 1 && anchor.treated === 0 ? 1 : 0]);
  const w = parts.reduce((s, [wt]) => s + wt, 0);
  return round((100 * parts.reduce((s, [wt, v]) => s + wt * v, 0)) / w, 1);
}

// Compare the arms of one trial. noise = second untreated sample (different seed) or null.
export function compareArms({ baseline, treated, noise = null, expected = "", logprobs = {}, clean = null }) {
  const b = textMetrics(baseline);
  const t = textMetrics(treated);
  const divergence = wordDivergence(baseline, treated);
  const noiseFloor = noise === null || noise === undefined ? null : wordDivergence(baseline, noise);
  const excessDivergence = divergence - (noiseFloor ?? 0);
  const ah = anchorHit(baseline, expected);
  const anchor = ah === null ? null : { baseline: ah, treated: anchorHit(treated, expected) };
  return {
    baseline: b,
    treated: t,
    noise: noise === null || noise === undefined ? null : textMetrics(noise),
    divergence: round(divergence),
    noiseFloor: round(noiseFloor),
    excessDivergence: round(excessDivergence),
    lengthRatio: round(b.words ? t.words / b.words : t.words ? Infinity : 1),
    anchor,
    internal: {
      baseline: logprobStats(logprobs.baseline),
      treated: logprobStats(logprobs.treated)
    },
    clean: cleanScores(clean),
    impairment: impairmentScore({ excessDivergence, baseline: b, treated: t, anchor })
  };
}

// Surprisal of each arm's text under the untreated model (teacher-forced), and the treated excess
// over the baseline. Inputs are scoreTokens() results or null.
export function cleanScores(clean) {
  if (!clean || (!clean.baseline && !clean.treated)) return null;
  const s = (x) => (x && Number.isFinite(x.surprisal) ? x.surprisal : null);
  const excess = s(clean.treated) !== null && s(clean.baseline) !== null ? round(s(clean.treated) - s(clean.baseline)) : null;
  return { baseline: clean.baseline ?? null, treated: clean.treated ?? null, noise: clean.noise ?? null, excess };
}

// --- statistics ---------------------------------------------------------------------------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function mean(values) {
  const v = values.filter(Number.isFinite);
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
}

// Percentile bootstrap CI of the mean (deterministic for a given seed).
export function bootstrapCI(values, { iterations = 2000, alpha = 0.05, seed = 12345 } = {}) {
  const v = values.filter(Number.isFinite);
  if (!v.length) return { n: 0, mean: null, lo: null, hi: null };
  const m = mean(v);
  if (v.length === 1) return { n: 1, mean: round(m), lo: null, hi: null };
  const rand = mulberry32(seed);
  const means = new Float64Array(iterations);
  for (let i = 0; i < iterations; i += 1) {
    let s = 0;
    for (let j = 0; j < v.length; j += 1) s += v[Math.floor(rand() * v.length)];
    means[i] = s / v.length;
  }
  means.sort();
  const at = (q) => means[Math.min(iterations - 1, Math.max(0, Math.floor(q * iterations)))];
  return { n: v.length, mean: round(m), lo: round(at(alpha / 2)), hi: round(at(1 - alpha / 2)) };
}

export const SUMMARY_FIELDS = {
  impairment: (m) => m.impairment,
  excessDivergence: (m) => m.excessDivergence,
  divergence: (m) => m.divergence,
  garble: (m) => m.treated.garble,
  repetition: (m) => m.treated.repetition,
  scriptSwitch: (m) => m.treated.scriptSwitch,
  anchorTreated: (m) => (m.anchor ? m.anchor.treated : null),
  entropy: (m) => m.internal?.treated?.entropy ?? null,
  surprisal: (m) => m.internal?.treated?.surprisal ?? null,
  cleanSurprisal: (m) => m.clean?.treated?.surprisal ?? null,
  cleanExcess: (m) => m.clean?.excess ?? null
};

// rows: [{ doseMg, metrics }] -> [{ doseMg, n, <field>: {n, mean, lo, hi} }]
export function summarizeByDose(rows) {
  const byDose = new Map();
  for (const row of rows) {
    if (!row.metrics) continue;
    if (!byDose.has(row.doseMg)) byDose.set(row.doseMg, []);
    byDose.get(row.doseMg).push(row.metrics);
  }
  return [...byDose.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([doseMg, list]) => {
      const out = { doseMg, n: list.length };
      for (const [key, get] of Object.entries(SUMMARY_FIELDS)) {
        out[key] = bootstrapCI(list.map(get).filter((x) => x !== null), { seed: 1000 + Math.round(doseMg) });
      }
      return out;
    });
}

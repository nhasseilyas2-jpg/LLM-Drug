// Scoring helpers for exact-answer benchmarks under perturbation (scripts/benchmark.mjs).

// Lowercase, drop <think> blocks, markdown, punctuation and articles-only noise; collapse spaces.
export function normalizeAnswer(text) {
  return String(text ?? "")
    .replace(/<think>[\s\S]*?(<\/think>|$)/gi, " ")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[*_`#>"“”‘’'()[\]{}]/g, " ")
    .replace(/[.,!?;:]+(\s|$)/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Correct when the (normalized) response is an accepted answer, or a short response (at most
// 6 words, e.g. "The answer is 43") contains an accepted answer as whole words.
export function isCorrect(response, answers) {
  const r = normalizeAnswer(response);
  if (!r) return false;
  const words = r.split(" ");
  return answers.some((a) => {
    const n = normalizeAnswer(a);
    if (!n) return false;
    if (r === n) return true;
    if (words.length > 6) return false;
    return new RegExp(`(^| )${n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}( |$)`).test(r);
  });
}

// Wilson score interval for a binomial proportion (95 % by default).
export function wilson(successes, n, z = 1.96) {
  if (!n) return { p: null, lo: null, hi: null };
  const p = successes / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return { p, lo: Math.max(0, c - h), hi: Math.min(1, c + h) };
}

// D50: the dose at which accuracy first falls to half of the 0 mg accuracy (linear interpolation).
// rows: [{ doseMg, accuracy }] sorted by dose, first row at 0 mg. null when it never gets there.
export function d50(rows) {
  if (!rows.length || rows[0].doseMg !== 0 || !(rows[0].accuracy > 0)) return null;
  const target = rows[0].accuracy / 2;
  for (let i = 1; i < rows.length; i += 1) {
    const a = rows[i - 1];
    const b = rows[i];
    if (b.accuracy <= target) {
      if (a.accuracy === b.accuracy) return b.doseMg;
      return Math.round(a.doseMg + ((a.accuracy - target) / (a.accuracy - b.accuracy)) * (b.doseMg - a.doseMg));
    }
  }
  return null;
}

export function markdownReport(result) {
  const doses = result.doses;
  const lines = [
    `# Benchmark: ${result.taskSet} on ${result.model}`,
    "",
    `${result.tasks} tasks, greedy decoding, max ${result.maxTokens} tokens, seed ${result.seed}. ` +
      `Accuracy in % (Wilson 95 % CI in the JSON). D50 = dose where accuracy falls to half of 0 mg.`,
    `Generated ${result.timestamp} by scripts/benchmark.mjs.`,
    "",
    `| Technique | ${doses.map((d) => `${d} mg`).join(" | ")} | D50 |`,
    `|---|${doses.map(() => "---:").join("|")}|---:|`
  ];
  for (const t of result.techniques) {
    const cells = doses.map((d) => {
      const row = t.rows.find((r) => r.doseMg === d);
      return row ? (row.accuracy * 100).toFixed(0) : "–";
    });
    lines.push(`| ${t.name} | ${cells.join(" | ")} | ${t.d50 === null ? "> " + doses[doses.length - 1] : t.d50} |`);
  }
  return lines.join("\n") + "\n";
}

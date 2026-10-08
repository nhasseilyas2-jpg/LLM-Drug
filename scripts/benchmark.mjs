#!/usr/bin/env node
// Robustness curves: accuracy on an exact-answer task set versus dose, for every technique.
//
//   npm run bench -- [--model <name|gguf:id|path.gguf>] [--techniques a,b|all] [--doses 0,50,150,300,500]
//                    [--tasks bench/exact-answer.json] [--limit N] [--max-tokens 24] [--seed 7] [--out data/bench]
//
// Writes <out>/<timestamp>-<model>.json (per-task answers, accuracy, Wilson CIs, D50) and a .md table.
// The 0 mg arm is run once and shared by all techniques (it is the same untreated model).
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { TECHNIQUE_IDS, buildLlamaRegimen, getTechnique, needsFixation, parseDoseList } from "../src/catalog.js";
import { d50, isCorrect, markdownReport, wilson } from "../lib/benchmark.js";
import { loadConfig, ROOT } from "../lib/config.js";
import { LlamaServerPool } from "../lib/llamacpp.js";
import { ModelRegistry, idFor } from "../lib/models.js";
import { listVectors, resolveSteering } from "../lib/steering.js";

function parseArgs(argv) {
  const args = { model: null, techniques: "all", doses: "0,50,150,300,500", tasks: path.join(ROOT, "bench", "exact-answer.json"),
    maxTokens: 24, seed: 7, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--model") args.model = next();
    else if (a === "--techniques") args.techniques = next();
    else if (a === "--doses") args.doses = next();
    else if (a === "--tasks") args.tasks = next();
    else if (a === "--limit") args.limit = Number.parseInt(next(), 10);
    else if (a === "--max-tokens") args.maxTokens = Number(next());
    else if (a === "--seed") args.seed = Number(next());
    else if (a === "--out") args.out = next();
    else if (a === "--help" || a === "-h") args.help = true;
    else throw new Error(`Unknown argument '${a}'`);
  }
  return args;
}

async function pickModel(config, query) {
  if (query && (await stat(query).catch(() => null))?.isFile()) {
    return { id: idFor(query), name: path.basename(query), file: path.resolve(query) };
  }
  const registry = new ModelRegistry(config);
  await registry.refresh();
  const models = [...registry.gguf.values()];
  if (!models.length) throw new Error("No GGUF models found.");
  if (!query) return models.sort((a, b) => a.size - b.size)[0];
  const q = query.toLowerCase();
  const hit = models.find((m) => m.id === query) || models.find((m) => m.name.toLowerCase().includes(q));
  if (!hit) throw new Error(`No model matches '${query}'.`);
  return hit;
}

export async function runBenchmark(opts = {}, log = console.log) {
  const config = loadConfig();
  const set = JSON.parse(await readFile(opts.tasks || path.join(ROOT, "bench", "exact-answer.json"), "utf8"));
  if (opts.limit > 0) set.tasks = set.tasks.slice(0, opts.limit);
  const model = await pickModel(config, opts.model);
  const doses = parseDoseList(opts.doses ?? "0,50,150,300,500");
  if (doses[0] !== 0) doses.unshift(0);
  const vectors = await listVectors(config, model.id);
  let ids = !opts.techniques || opts.techniques === "all" ? TECHNIQUE_IDS.filter((t) => t !== "placebo") : opts.techniques.split(",").map((s) => s.trim());
  ids = ids.filter((id) => {
    const need = getTechnique(id).requires?.steering;
    if (need && !vectors.includes(need)) {
      log(`skip ${id}: no '${need}' control vector for ${model.name} (npm run vectors -- --model "${model.name}")`);
      return false;
    }
    return true;
  });
  const maxTokens = opts.maxTokens || 24;
  const seed = Number.isFinite(opts.seed) ? opts.seed : 7;
  const pool = new LlamaServerPool(config);
  const sampling = { temperature: 0, top_p: 1, top_k: 0, min_p: 0, repeat_penalty: 1 };

  const runArm = async (env) => {
    const answers = [];
    for (const task of set.tasks) {
      const r = await pool.chat(model.file, env, {
        messages: [{ role: "system", content: set.system }, { role: "user", content: task.q }],
        sampling, seed, maxTokens, logprobs: false
      });
      answers.push({ id: task.id, response: r.content.slice(0, 300), correct: isCorrect(r.content, task.a), sitesFired: r.engine?.sitesFired ?? [] });
    }
    const correct = answers.filter((a) => a.correct).length;
    const ci = wilson(correct, answers.length);
    return { correct, n: answers.length, accuracy: ci.p, ci: [ci.lo, ci.hi], answers };
  };

  const started = Date.now();
  try {
    log(`model ${model.name}: ${set.tasks.length} tasks × ${ids.length} techniques × ${doses.length - 1} doses (+ shared 0 mg)`);
    const zero = await runArm({});
    log(`0 mg: ${(zero.accuracy * 100).toFixed(0)} %`);
    const techniques = [];
    for (const id of ids) {
      const t = getTechnique(id);
      const fixationIds = doses.some((d) => d > 0 && needsFixation(id, d)) ? await pool.tokenIds(model.file, t.theme || []) : [];
      const rows = [{ doseMg: 0, ...zero }];
      for (const dose of doses.filter((d) => d > 0)) {
        const regimen = buildLlamaRegimen({ techniqueId: id, doseMg: dose, seed, fixationIds });
        const env = await resolveSteering(config, model.id, model.name, regimen.env);
        const arm = await runArm(env);
        rows.push({ doseMg: dose, intensity: regimen.intensity, ...arm });
      }
      const summary = rows.map(({ doseMg, accuracy }) => ({ doseMg, accuracy }));
      techniques.push({ id, name: t.name, d50: d50(summary), rows });
      log(`${t.name.padEnd(14)} ${rows.map((r) => `${r.doseMg}:${(r.accuracy * 100).toFixed(0)}%`).join("  ")}  D50=${d50(summary) ?? "n/a"}`);
    }
    return {
      schema: 1,
      type: "benchmark",
      timestamp: new Date().toISOString(),
      model: model.name,
      modelId: model.id,
      taskSet: set.name,
      tasks: set.tasks.length,
      doses,
      seed,
      maxTokens,
      durationMs: Date.now() - started,
      techniques
    };
  } finally {
    pool.stopAll();
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exit(2);
  }
  if (args.help) {
    console.log("usage: npm run bench -- [--model M] [--techniques a,b|all] [--doses 0,50,150,300,500] [--tasks file.json] [--max-tokens N] [--seed N] [--out dir]");
    process.exit(0);
  }
  try {
    const result = await runBenchmark(args);
    const outDir = path.resolve(args.out || path.join(loadConfig().dataDir, "bench"));
    await mkdir(outDir, { recursive: true });
    const stem = `${result.timestamp.replace(/[:.]/g, "-")}-${result.model.replace(/[^\w.-]+/g, "_")}`;
    await writeFile(path.join(outDir, `${stem}.json`), JSON.stringify(result, null, 2));
    await writeFile(path.join(outDir, `${stem}.md`), markdownReport(result));
    console.log(`\nwrote ${path.join(outDir, stem)}.{json,md}`);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}

#!/usr/bin/env node
// Builds steering control vectors for a local GGUF model with llama-cvector-generator.
//
//   npm run vectors -- --model <file name | gguf:id | path to .gguf> [--vector mood|all] [--force]
//
// For every prompt set in steering/<name>.json, contrastive prompt pairs (positive / negative persona,
// same question, same assistant prefix) are rendered with the model's own chat template, and the
// per-layer mean difference of hidden states (--method mean) is written to
//   <steeringDir>/<model id>/<name>.gguf   (default steeringDir: data/steering, gitignored)
import { spawn } from "node:child_process";
import { access, mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { loadConfig, ROOT } from "../lib/config.js";
import { LlamaServerPool } from "../lib/llamacpp.js";
import { ModelRegistry, idFor } from "../lib/models.js";
import { VECTOR_NAME, vectorFile } from "../lib/steering.js";

const SETS_DIR = path.join(ROOT, "steering");

function parseArgs(argv) {
  const args = { model: null, vector: "all", force: false, threads: null };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--model") args.model = argv[++i];
    else if (a === "--vector") args.vector = argv[++i];
    else if (a === "--threads") args.threads = argv[++i];
    else if (a === "--force") args.force = true;
    else if (a === "--help" || a === "-h") args.help = true;
    else throw new Error(`Unknown argument '${a}'`);
  }
  return args;
}

// cvector-generator reads one prompt per line and unescapes "\n"
const escapeLine = (text) => text.replace(/\\/g, "\\\\").replace(/\r?\n/g, "\\n");

export function buildPairs(set) {
  const pairs = [];
  set.personas.forEach(([positive, negative], i) => {
    set.questions.forEach((question, j) => {
      const prefix = set.completions[(i + j) % set.completions.length];
      pairs.push({ positive, negative, question, prefix });
    });
  });
  return pairs;
}

async function loadSets(which) {
  const names = (await readdir(SETS_DIR)).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5));
  const chosen = which === "all" ? names : which.split(",").map((s) => s.trim());
  const sets = [];
  for (const name of chosen) {
    if (!VECTOR_NAME.test(name) || !names.includes(name)) throw new Error(`No prompt set steering/${name}.json`);
    const set = JSON.parse(await readFile(path.join(SETS_DIR, `${name}.json`), "utf8"));
    if (!Array.isArray(set.personas) || !set.personas.length || !Array.isArray(set.questions) || !Array.isArray(set.completions)) {
      throw new Error(`steering/${name}.json needs personas[[pos, neg]], questions[] and completions[]`);
    }
    sets.push({ ...set, name });
  }
  return sets;
}

function pickModel(models, query) {
  if (!models.length) throw new Error("No GGUF models found (models/ or Ollama blobs).");
  if (!query) return [...models].sort((a, b) => a.size - b.size)[0];
  const q = query.toLowerCase();
  const hit = models.find((m) => m.id === query) || models.find((m) => m.name.toLowerCase() === q) || models.find((m) => m.name.toLowerCase().includes(q));
  if (!hit) throw new Error(`No model matches '${query}'. Known: ${models.map((m) => m.name).join(", ")}`);
  return hit;
}

function run(bin, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    let tail = "";
    const keep = (chunk) => {
      tail = (tail + chunk.toString()).slice(-4000);
    };
    child.stdout.on("data", keep);
    child.stderr.on("data", keep);
    child.on("error", reject);
    child.on("exit", (code) => (code === 0 ? resolve(tail) : reject(new Error(`${path.basename(bin)} exited with ${code}:\n${tail}`))));
  });
}

export async function makeVectors({ model: query, vector = "all", force = false, threads = null, log = console.log } = {}) {
  const config = loadConfig();
  await access(config.cvectorBin).catch(() => {
    throw new Error(`llama-cvector-generator not found at ${config.cvectorBin}. Re-run scripts/setup-llamacpp-injection.ps1 (or .sh).`);
  });
  let model;
  if (query && (await stat(query).catch(() => null))?.isFile()) {
    model = { id: idFor(query), name: path.basename(query), file: path.resolve(query) };
  } else {
    const registry = new ModelRegistry(config);
    await registry.refresh();
    model = pickModel([...registry.gguf.values()], query);
  }
  const sets = await loadSets(vector);
  const pool = new LlamaServerPool(config);
  const written = [];
  const tmp = await mkdtemp(path.join(os.tmpdir(), "llm-inj-cvec-"));
  try {
    for (const set of sets) {
      const out = vectorFile(config, model.id, set.name);
      if (!force && (await stat(out).catch(() => null))) {
        log(`skip ${set.name}: ${out} exists (use --force to rebuild)`);
        written.push(out);
        continue;
      }
      const server = await pool.acquire(model.file, {});
      const render = async (system, question, prefix) => {
        const r = await server.request("/apply-template", { messages: [{ role: "system", content: system }, { role: "user", content: question }] });
        return escapeLine(r.prompt + prefix);
      };
      const positive = [];
      const negative = [];
      for (const p of buildPairs(set)) {
        positive.push(await render(p.positive, p.question, p.prefix));
        negative.push(await render(p.negative, p.question, p.prefix));
      }
      pool.stopAll();
      const pos = path.join(tmp, `${set.name}.pos.txt`);
      const neg = path.join(tmp, `${set.name}.neg.txt`);
      await writeFile(pos, positive.join("\n") + "\n");
      await writeFile(neg, negative.join("\n") + "\n");
      await mkdir(path.dirname(out), { recursive: true });
      log(`building ${set.name} for ${model.name} from ${positive.length} prompt pairs ...`);
      const args = ["-m", model.file, "--positive-file", pos, "--negative-file", neg, "--method", "mean", "-o", out, "-c", "512"];
      args.push("-t", String(threads || config.llamaThreads));
      if (config.llamaGpuLayers !== null) args.push("-ngl", String(config.llamaGpuLayers));
      await run(config.cvectorBin, args);
      log(`wrote ${out}`);
      written.push(out);
    }
  } finally {
    pool.stopAll();
    await rm(tmp, { recursive: true, force: true });
  }
  return { model: { id: model.id, name: model.name }, files: written };
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
    console.log("usage: npm run vectors -- [--model <name|gguf:id>] [--vector mood|all] [--force] [--threads N]");
    process.exit(0);
  }
  makeVectors(args).catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

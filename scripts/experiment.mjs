#!/usr/bin/env node
// Headless experiment runner (no web UI). Runs a dose-response sweep per technique defined in a
// JSON file and writes JSON + CSV results to data/experiments/.
//   node scripts/experiment.mjs experiments/smoke-llamacpp.json [--model <substring>]
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { loadConfig } from "../lib/config.js";
import { Lab } from "../lib/lab.js";
import { runDoseResponse, validateInput } from "../lib/experiment.js";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
const modelFlag = args.includes("--model") ? args[args.indexOf("--model") + 1] : null;
if (!file) {
  console.error("usage: experiment.mjs <experiment.json> [--model <name substring>]");
  process.exit(2);
}

const spec = JSON.parse(await readFile(file, "utf8"));
const config = loadConfig();
const lab = new Lab(config);
process.on("SIGINT", () => { lab.shutdown(); process.exit(130); });

try {
  const models = await lab.registry.refresh();
  const wanted = modelFlag || spec.model;
  const pool = spec.backend === "llamacpp" ? models.gguf : models.ollama;
  const model = pool.find((m) => m.id === wanted) || pool.find((m) => m.name.toLowerCase().includes(String(wanted).toLowerCase()));
  if (!model) throw new Error(`Model '${wanted}' not found for backend ${spec.backend}. Available: ${pool.map((m) => m.name).join(", ")}`);

  const outDir = path.join(config.dataDir, "experiments");
  await mkdir(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const base = path.join(outDir, `${spec.name || path.basename(file, ".json")}-${stamp}`);
  const results = [];
  const csv = ["technique,prompt_index,dose_mg,trial,seed,intensity,impairment,divergence,noise_floor,excess_divergence,garble,repetition,script_switch,anchor,entropy,surprisal,words"];

  for (const techniqueId of spec.techniques) {
    for (const [pi, prompt] of spec.prompts.entries()) {
      const input = validateInput({ ...spec, ...prompt, techniqueId, modelId: model.id }, "dose-response");
      console.log(`\n== ${techniqueId} | prompt ${pi + 1}/${spec.prompts.length} | ${model.name}`);
      const record = await runDoseResponse(lab, input, {
        progress: ({ done, total, message }) => process.stdout.write(`\r  [${done}/${total}] ${message}`.padEnd(60))
      });
      process.stdout.write("\n");
      await lab.history.append(record);
      results.push(record);
      for (const row of record.rows) {
        const m = row.metrics;
        csv.push([
          techniqueId, pi, row.doseMg, row.trial, row.seed, row.intensity, m.impairment, m.divergence, m.noiseFloor ?? "",
          m.excessDivergence, m.treated.garble, m.treated.repetition, m.treated.scriptSwitch, m.anchor ? m.anchor.treated : "",
          m.internal.treated?.entropy ?? "", m.internal.treated?.surprisal ?? "", m.treated.words
        ].join(","));
      }
      for (const s of record.summary) {
        const ci = (x) => (x.lo === null ? `${x.mean}` : `${x.mean} [${x.lo}, ${x.hi}]`);
        console.log(`  ${String(s.doseMg).padStart(4)} mg  impairment ${ci(s.impairment).padEnd(24)} excess-div ${ci(s.excessDivergence)}`);
      }
    }
  }
  await writeFile(`${base}.json`, JSON.stringify({ spec, model: model.name, results }, null, 2));
  await writeFile(`${base}.csv`, `${csv.join("\n")}\n`);
  console.log(`\nWrote ${base}.json and .csv`);
} catch (error) {
  console.error(`\nerror: ${error.message}`);
  process.exitCode = 1;
} finally {
  lab.shutdown();
}

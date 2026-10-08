#!/usr/bin/env node
// Prints the LLM_INJ_* environment for a technique and dose, for use with any patched
// llama.cpp tool (llama-server, llama-completion, ...).
//   node scripts/inject-env.mjs <technique> <doseMg> [--seed N] [--ids 1,2,3] [--format ps|sh|json]
import { TECHNIQUE_IDS, buildLlamaEnv, describeDose, getTechnique } from "../src/catalog.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] !== undefined ? args.splice(i, 2)[1] : fallback;
};
const seed = flag("seed", "0");
const ids = flag("ids", "");
const format = flag("format", process.platform === "win32" ? "ps" : "sh");
const [techniqueId, dose] = args;

if (!techniqueId || dose === undefined) {
  console.error(`usage: inject-env.mjs <technique> <doseMg> [--seed N] [--ids 1,2] [--format ps|sh|json]\ntechniques: ${TECHNIQUE_IDS.join(", ")}`);
  process.exit(2);
}
try {
  getTechnique(techniqueId);
} catch (error) {
  console.error(error.message);
  process.exit(2);
}
const result = buildLlamaEnv(techniqueId, dose, {
  seed,
  fixationIds: ids.split(",").map((x) => Number.parseInt(x, 10)).filter(Number.isInteger)
});
if (format === "json") {
  console.log(JSON.stringify(result, null, 2));
} else {
  const comment = `# ${result.techniqueId} ${result.doseMg} mg -> intensity ${result.intensity} (${describeDose(techniqueId, dose)})`;
  console.log(comment);
  for (const [k, v] of Object.entries(result.env)) {
    console.log(format === "ps" ? `$env:${k} = "${v}"` : `export ${k}="${v}"`);
  }
  if (result.env.LLM_INJ_FIXATION_BIAS && !result.env.LLM_INJ_FIXATION_IDS) {
    console.log("# note: fixation needs token ids (--ids); get them from llama-server POST /tokenize");
  }
}

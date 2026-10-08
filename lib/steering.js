import { access, readdir } from "node:fs/promises";
import path from "node:path";

// Control vectors (steering directions) live outside the repository, one folder per model id:
//   <steeringDir>/<16-hex model id>/<name>.gguf
// They are produced by scripts/make-control-vectors.mjs from the prompt sets in steering/.

export const VECTOR_NAME = /^[a-z][a-z0-9_-]{0,39}$/;

export function vectorFile(config, modelId, name) {
  const id = String(modelId).replace(/^gguf:/, "");
  if (!/^[0-9a-f]{16}$/.test(id)) throw new Error(`Bad model id '${modelId}'`);
  if (!VECTOR_NAME.test(name)) throw new Error(`Bad vector name '${name}'`);
  return path.join(config.steeringDir, id, `${name}.gguf`);
}

export async function listVectors(config, modelId) {
  try {
    const dir = path.dirname(vectorFile(config, modelId, "x"));
    return (await readdir(dir))
      .filter((f) => f.endsWith(".gguf"))
      .map((f) => f.slice(0, -5))
      .filter((n) => VECTOR_NAME.test(n))
      .sort();
  } catch {
    return [];
  }
}

// Replaces the catalog's symbolic LLM_INJ_STEER_VEC=<name> with the engine's LLM_INJ_STEER_FILE=<path>.
// Throws (with a fix-it hint) when the vector has not been generated for this model yet.
export async function resolveSteering(config, modelId, modelName, env) {
  const name = env.LLM_INJ_STEER_VEC;
  if (!name) return env;
  const { LLM_INJ_STEER_VEC: _name, ...rest } = env;
  const file = vectorFile(config, modelId, name);
  try {
    await access(file);
  } catch {
    const error = new Error(
      `No '${name}' control vector for ${modelName}. Generate it once with: npm run vectors -- --model "${modelName}" --vector ${name}`
    );
    error.code = "STEERING_VECTOR_MISSING";
    throw error;
  }
  return { ...rest, LLM_INJ_STEER_FILE: file };
}

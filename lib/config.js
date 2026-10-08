import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function defaultBin(name) {
  const exe = process.platform === "win32" ? `${name}.exe` : name;
  const buildBin = path.join(ROOT, "vendor", "llama.cpp", "build", "bin");
  return process.platform === "win32" ? path.join(buildBin, "Release", exe) : path.join(buildBin, exe);
}

const int = (value, fallback) => {
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
};

// All filesystem locations come from the environment, never from HTTP clients.
export function loadConfig(env = process.env) {
  return {
    host: "127.0.0.1",
    port: int(env.PORT, 4173),
    staticDir: path.join(ROOT, env.LAB_SERVE_DIST === "1" ? "dist" : "src"),
    dataDir: path.resolve(env.LAB_DATA_DIR || path.join(ROOT, "data")),
    ollamaUrl: (env.OLLAMA_URL || "http://127.0.0.1:11434").replace(/\/+$/, ""),
    ollamaModelsDir: env.OLLAMA_MODELS || path.join(os.homedir(), ".ollama", "models"),
    llamaServerBin: path.resolve(env.LLAMA_SERVER_BIN || defaultBin("llama-server")),
    cvectorBin: path.resolve(env.LLAMA_CVECTOR_BIN || defaultBin("llama-cvector-generator")),
    // control vectors (steering directions), one folder per model id: <steeringDir>/<id>/<name>.gguf
    steeringDir: path.resolve(env.LAB_STEERING_DIR || path.join(env.LAB_DATA_DIR || path.join(ROOT, "data"), "steering")),
    modelsDirs: (env.LLAMA_MODELS_DIR || path.join(ROOT, "models")).split(path.delimiter).filter(Boolean),
    llamaThreads: int(env.LLAMA_THREADS, Math.max(1, Math.min(8, os.cpus().length - 1))),
    llamaCtx: int(env.LLAMA_CTX, 4096),
    llamaGpuLayers: env.LLAMA_GPU_LAYERS ?? null,
    llamaMaxServers: Math.max(2, int(env.LLAMA_MAX_SERVERS, 2)),
    // one llama-server per model that re-reads its injection config per request (falls back automatically)
    llamaShared: !/^(0|false|no|off)$/i.test(env.LLAMA_SHARED || ""),
    llamaStartTimeoutMs: int(env.LLAMA_START_TIMEOUT_MS, 300000),
    requestTimeoutMs: int(env.LAB_REQUEST_TIMEOUT_MS, 600000),
    historyMaxBytes: int(env.LAB_HISTORY_MAX_BYTES, 20 * 1024 * 1024)
  };
}

export async function fetchJson(url, { method = "GET", body, headers = {}, timeoutMs = 60000, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs} ms: ${url}`)), timeoutMs);
  const onAbort = () => controller.abort(signal.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const response = await fetch(url, {
      method,
      headers: body === undefined ? headers : { "content-type": "application/json", ...headers },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal
    });
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
    if (!response.ok) {
      const detail = data?.error?.message || data?.error || text.slice(0, 300);
      throw new Error(`${method} ${url} -> HTTP ${response.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    return data;
  } catch (error) {
    if (controller.signal.aborted && controller.signal.reason instanceof Error) throw controller.signal.reason;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

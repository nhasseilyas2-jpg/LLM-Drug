import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { access } from "node:fs/promises";
import net from "node:net";
import { fetchJson } from "./config.js";

// Pool of patched llama-server processes. Each process has a fixed LLM_INJ_* environment
// (the engine reads it once at start-up), so every distinct (model, injection env) pair is a
// separate process. Processes are reused and evicted least-recently-used.

const STDERR_KEEP = 200;

export function parseAuditLine(line, audit) {
  const m = /llm-injection: (ACTIVE|inactive|site fired:|ignoring)\s*(.*)$/.exec(line);
  if (!m) return false;
  if (m[1] === "ACTIVE") audit.active = m[2].trim();
  else if (m[1] === "inactive") audit.active = null;
  else if (m[1] === "site fired:") {
    const site = m[2].trim();
    if (!audit.sitesFired.includes(site)) audit.sitesFired.push(site);
  } else audit.warnings.push(`ignoring ${m[2].trim()}`);
  return true;
}

// Environment for a child: the parent's environment minus any LLM_INJ_* plus the injection env.
export function childEnv(base, injectionEnv) {
  const env = {};
  for (const [k, v] of Object.entries(base)) if (!/^LLM_INJ_/i.test(k)) env[k] = v;
  return { ...env, ...injectionEnv };
}

function envKey(modelPath, injectionEnv) {
  const sorted = Object.keys(injectionEnv).sort().map((k) => `${k}=${injectionEnv[k]}`);
  return `${modelPath}\n${sorted.join("\n")}`;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class LlamaServer {
  constructor(config, modelPath, injectionEnv) {
    this.config = config;
    this.modelPath = modelPath;
    this.injectionEnv = injectionEnv;
    this.key = envKey(modelPath, injectionEnv);
    this.apiKey = randomBytes(18).toString("hex");
    this.audit = { active: null, sitesFired: [], warnings: [] };
    this.stderr = [];
    this.busy = 0;
    this.lastUsed = Date.now();
    this.exited = false;
  }

  async start() {
    this.port = await freePort();
    const c = this.config;
    const args = [
      "-m", this.modelPath,
      "--host", "127.0.0.1",
      "--port", String(this.port),
      "-c", String(c.llamaCtx),
      "-np", "1",
      "-t", String(c.llamaThreads),
      "--no-webui",
      "--no-cache-prompt",
      "--api-key", this.apiKey,
      "--reasoning", "off"
    ];
    if (c.llamaGpuLayers !== null && c.llamaGpuLayers !== undefined) args.push("-ngl", String(c.llamaGpuLayers));
    this.child = spawn(c.llamaServerBin, args, {
      env: childEnv(process.env, this.injectionEnv),
      stdio: ["ignore", "ignore", "pipe"],
      windowsHide: true
    });
    let partial = "";
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk) => {
      const lines = (partial + chunk).split(/\r?\n/);
      partial = lines.pop();
      for (const line of lines) {
        parseAuditLine(line, this.audit);
        this.stderr.push(line);
        if (this.stderr.length > STDERR_KEEP) this.stderr.shift();
      }
    });
    const exited = new Promise((resolve) => {
      this.child.on("exit", (code) => { this.exited = true; this.exitCode = code; resolve(); });
      this.child.on("error", (error) => { this.exited = true; this.spawnError = error; resolve(); });
    });
    const deadline = Date.now() + c.llamaStartTimeoutMs;
    while (Date.now() < deadline) {
      if (this.exited) break;
      try {
        await fetchJson(`${this.url}/health`, { timeoutMs: 2000 });
        return this;
      } catch {
        await Promise.race([sleep(300), exited]);
      }
    }
    this.stop();
    const tail = this.stderr.slice(-12).join("\n");
    throw new Error(
      `llama-server failed to start (${this.spawnError?.message || `exit code ${this.exitCode ?? "timeout"}`}).\n${tail}`
    );
  }

  get url() {
    return `http://127.0.0.1:${this.port}`;
  }

  request(pathname, body, { timeoutMs, signal } = {}) {
    return fetchJson(`${this.url}${pathname}`, {
      method: body === undefined ? "GET" : "POST",
      body,
      headers: { authorization: `Bearer ${this.apiKey}` },
      timeoutMs: timeoutMs ?? this.config.requestTimeoutMs,
      signal
    });
  }

  stop() {
    if (this.child && !this.exited) this.child.kill();
  }
}

export class LlamaServerPool {
  constructor(config) {
    this.config = config;
    this.servers = new Map();
    this.starting = new Map();
  }

  async binaryAvailable() {
    try {
      await access(this.config.llamaServerBin);
      return true;
    } catch {
      return false;
    }
  }

  async acquire(modelPath, injectionEnv) {
    const key = envKey(modelPath, injectionEnv);
    const existing = this.servers.get(key);
    if (existing && !existing.exited) {
      existing.lastUsed = Date.now();
      return existing;
    }
    if (this.starting.has(key)) return this.starting.get(key);
    if (!(await this.binaryAvailable())) {
      throw new Error(
        `Patched llama-server not found at ${this.config.llamaServerBin}. Run scripts/setup-llamacpp-injection.ps1 (or .sh) or set LLAMA_SERVER_BIN.`
      );
    }
    this.evict(this.config.llamaMaxServers - 1);
    const server = new LlamaServer(this.config, modelPath, injectionEnv);
    const promise = server.start().then(
      (s) => { this.servers.set(key, s); this.starting.delete(key); return s; },
      (error) => { this.starting.delete(key); throw error; }
    );
    this.starting.set(key, promise);
    return promise;
  }

  // Stops least-recently-used idle servers until at most `keep` remain.
  evict(keep) {
    for (const [key, s] of this.servers) if (s.exited) this.servers.delete(key);
    const idle = [...this.servers.values()].filter((s) => s.busy === 0).sort((a, b) => a.lastUsed - b.lastUsed);
    while (this.servers.size > keep && idle.length) {
      const victim = idle.shift();
      victim.stop();
      this.servers.delete(victim.key);
    }
  }

  async chat(modelPath, injectionEnv, { messages, sampling, seed, maxTokens, logprobs = true, signal }) {
    const server = await this.acquire(modelPath, injectionEnv);
    server.busy += 1;
    try {
      const body = {
        messages,
        max_tokens: maxTokens,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        repeat_penalty: sampling.repeat_penalty,
        seed,
        cache_prompt: false
      };
      if (logprobs) {
        body.logprobs = true;
        body.top_logprobs = 5;
      }
      const data = await server.request("/v1/chat/completions", body, { signal });
      const choice = data?.choices?.[0] || {};
      return {
        content: choice.message?.content ?? "",
        logprobs: choice.logprobs?.content ?? null,
        usage: { promptTokens: data?.usage?.prompt_tokens ?? null, completionTokens: data?.usage?.completion_tokens ?? null },
        finishReason: choice.finish_reason ?? null,
        engine: {
          injectionEnv: server.injectionEnv,
          active: server.audit.active,
          sitesFired: [...server.audit.sitesFired],
          warnings: [...server.audit.warnings]
        }
      };
    } finally {
      server.busy -= 1;
      server.lastUsed = Date.now();
    }
  }

  // Token ids of the first token of each word variant (" word", " Word", "word"), used for
  // fixation. Pieces shorter than 3 characters are skipped to avoid biasing common sub-words.
  async tokenIds(modelPath, wordsList) {
    const server = await this.acquire(modelPath, {});
    const ids = new Set();
    for (const word of wordsList) {
      const variants = [` ${word}`, ` ${word[0].toUpperCase()}${word.slice(1)}`, word];
      for (const content of variants) {
        const data = await server.request("/tokenize", { content, add_special: false, with_pieces: true }, { timeoutMs: 10000 });
        const first = data?.tokens?.[0];
        const piece = typeof first?.piece === "string" ? first.piece.trim() : "";
        if (first && Number.isInteger(first.id) && piece.length >= 3) ids.add(first.id);
      }
    }
    return [...ids].sort((a, b) => a - b);
  }

  status() {
    return [...this.servers.values()].map((s) => ({
      model: s.modelPath,
      port: s.port,
      injectionEnv: s.injectionEnv,
      active: s.audit.active,
      sitesFired: s.audit.sitesFired,
      busy: s.busy
    }));
  }

  stopAll() {
    for (const s of this.servers.values()) s.stop();
    this.servers.clear();
  }
}

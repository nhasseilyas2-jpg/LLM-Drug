import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { access, unlink, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fetchJson } from "./config.js";

// Pool of patched llama-server processes.
//
// Shared mode (default): one process per model. The process is started with LLM_INJ_CONFIG_FILE
// pointing at a private file; the engine re-reads it whenever a request creates its sampler chain,
// so the pool writes the request's injection config to the file and then sends the request.
// Requests to a shared process are serialized. The engine prints "config gen=N" when it loads a new
// configuration, which also marks the start of a fresh audit (ACTIVE line and fired sites).
//
// Per-config mode (fallback for engines without reload support, or LLAMA_SHARED=0): the engine reads
// a fixed LLM_INJ_* environment once at start-up, so every distinct (model, injection env) pair is a
// separate process. Processes are reused and evicted least-recently-used.

const STDERR_KEEP = 200;

export function parseAuditLine(line, audit) {
  const m = /llm-injection: (ACTIVE|inactive|site fired:|ignoring|config gen=)\s*(.*)$/.exec(line);
  if (!m) return false;
  if (m[1] === "config gen=") {
    audit.gen = Number(m[2]);
    audit.active = null;
    audit.sitesFired = [];
    audit.warnings = [];
  } else if (m[1] === "ACTIVE") audit.active = m[2].trim();
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

// Config-file text for an injection env (one KEY=VALUE per line, sorted).
export function envText(injectionEnv) {
  return Object.keys(injectionEnv)
    .sort()
    .map((k) => `${k}=${String(injectionEnv[k]).replace(/[\r\n]+/g, " ")}\n`)
    .join("");
}

export function parseEnvText(text) {
  const env = {};
  for (const line of (text || "").split("\n")) {
    const i = line.indexOf("=");
    if (i > 0) env[line.slice(0, i)] = line.slice(i + 1);
  }
  return env;
}

// GBNF string literal that matches exactly `text`.
export function gbnfLiteral(text) {
  let out = '"';
  for (const ch of text) {
    const c = ch.codePointAt(0);
    if (ch === "\\" || ch === '"') out += `\\${ch}`;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += `\\x${c.toString(16).padStart(2, "0")}`;
    else out += ch;
  }
  return `${out}"`;
}

// The part of a generated text that can be teacher-forced: decoding damage (U+FFFD from a split
// multi-byte character) cannot be reproduced by a grammar, so scoring stops before it.
export function scorableText(text) {
  if (typeof text !== "string") return "";
  const i = text.indexOf("\uFFFD");
  return i < 0 ? text : text.slice(0, i);
}

// Aligns forced-token log-probabilities with the target text. Returns mean surprisal in nats per
// token and per character, or null when nothing was scored.
export function scoreTokens(logprobs, target, original = target) {
  if (!Array.isArray(logprobs) || !target) return null;
  let covered = 0;
  const lps = [];
  for (const entry of logprobs) {
    if (covered >= target.length) break;
    if (!Number.isFinite(entry?.logprob)) continue;
    lps.push(entry.logprob);
    covered += typeof entry.token === "string" ? entry.token.length : 0;
  }
  if (!lps.length) return null;
  const total = -lps.reduce((s, x) => s + x, 0);
  const r = (x) => Math.round(x * 1e4) / 1e4;
  return {
    tokens: lps.length,
    surprisal: r(total / lps.length),
    perChar: r(total / Math.max(1, Math.min(covered, target.length))),
    max: r(-Math.min(...lps)),
    coverage: r(Math.min(1, covered / Math.max(1, original.length)))
  };
}

function envKey(modelPath, injectionEnv, shared = false) {
  if (shared) return `${modelPath}\n#shared`;
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
  constructor(config, modelPath, injectionEnv, shared = false) {
    this.config = config;
    this.modelPath = modelPath;
    this.shared = shared;
    this.injectionEnv = shared ? {} : injectionEnv;
    this.key = envKey(modelPath, injectionEnv, shared);
    this.apiKey = randomBytes(18).toString("hex");
    this.audit = { gen: null, active: null, sitesFired: [], warnings: [] };
    this.stderr = [];
    this.lastStderrAt = 0;
    this.busy = 0;
    this.lastUsed = Date.now();
    this.exited = false;
    this.queue = Promise.resolve();
    if (shared) {
      this.configFile = path.join(os.tmpdir(), `llm-inj-${process.pid}-${this.apiKey.slice(0, 12)}.cfg`);
      this.configText = "";
    }
  }

  // true when the engine announced a reloadable configuration (supports shared mode)
  get reloadable() {
    return this.audit.gen !== null;
  }

  // runs fn with exclusive use of this server
  async exclusive(fn) {
    const prev = this.queue;
    let release;
    this.queue = new Promise((resolve) => { release = resolve; });
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  // waits until stderr has been quiet for quietMs (audit lines are flushed before the HTTP response,
  // but arrive through a separate pipe)
  async drainStderr({ untilGen = null, quietMs = 40, maxMs = 2000 } = {}) {
    const deadline = Date.now() + maxMs;
    while (Date.now() < deadline) {
      const genOk = untilGen === null || (this.audit.gen ?? -1) >= untilGen;
      if (genOk && Date.now() - this.lastStderrAt >= quietMs) return;
      await sleep(10);
    }
  }

  async start() {
    if (this.shared) await writeFile(this.configFile, "");
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
      env: childEnv(process.env, this.shared ? { LLM_INJ_CONFIG_FILE: this.configFile } : this.injectionEnv),
      stdio: ["ignore", "ignore", "pipe"],
      windowsHide: true
    });
    let partial = "";
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk) => {
      const lines = (partial + chunk).split(/\r?\n/);
      partial = lines.pop();
      this.lastStderrAt = Date.now();
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
    if (this.configFile) unlink(this.configFile).catch(() => {});
  }
}

export class LlamaServerPool {
  constructor(config) {
    this.config = config;
    this.servers = new Map();
    this.starting = new Map();
    // null = unknown, true/false once a shared server reported (or failed to report) reload support
    this.sharedSupported = config.llamaShared === false ? false : null;
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
    if (this.sharedSupported !== false) {
      const server = await this.acquireKey(modelPath, injectionEnv, true);
      if (server.reloadable) {
        this.sharedSupported = true;
        return server;
      }
      // engine without per-request reload: fall back to one process per configuration
      this.sharedSupported = false;
      server.stop();
      this.servers.delete(server.key);
    }
    return this.acquireKey(modelPath, injectionEnv, false);
  }

  async acquireKey(modelPath, injectionEnv, shared) {
    const key = envKey(modelPath, injectionEnv, shared);
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
    const server = new LlamaServer(this.config, modelPath, injectionEnv, shared);
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

  async chat(modelPath, injectionEnv, { messages, sampling, seed, maxTokens, logprobs = true, signal, extra = null }) {
    const server = await this.acquire(modelPath, injectionEnv);
    server.busy += 1;
    try {
      if (!server.shared) return await this.complete(server, injectionEnv, { messages, sampling, seed, maxTokens, logprobs, signal, extra });
      return await server.exclusive(async () => {
        const text = envText(injectionEnv);
        let untilGen = null;
        if (text !== server.configText) {
          await writeFile(server.configFile, text);
          server.configText = text;
          untilGen = (server.audit.gen ?? 0) + 1;
        }
        return this.complete(server, injectionEnv, { messages, sampling, seed, maxTokens, logprobs, signal, untilGen, extra });
      });
    } finally {
      server.busy -= 1;
      server.lastUsed = Date.now();
    }
  }

  // Clean-model surprisal of `text` as the answer to `messages`: teacher forcing in one request.
  // A grammar that only accepts `text` forces the tokens, and the reported log-probabilities are
  // taken from the raw (pre-sampling) logits of the untreated model. The tokenization chosen under
  // the grammar may differ from the original one, so the result is the log-probability of one
  // tokenization path (a lower bound on the string probability).
  async score(modelPath, messages, text, { signal } = {}) {
    const target = scorableText(text);
    if (!target) return null;
    const r = await this.chat(modelPath, {}, {
      messages,
      sampling: { temperature: 0, top_p: 1, top_k: 0, min_p: 0, repeat_penalty: 1 },
      seed: 0,
      maxTokens: Math.min(target.length + 8, this.config.llamaCtx),
      logprobs: 1,
      signal,
      extra: { grammar: `root ::= ${gbnfLiteral(target)}` }
    });
    return scoreTokens(r.logprobs, target, text);
  }

  async complete(server, injectionEnv, { messages, sampling, seed, maxTokens, logprobs, signal, untilGen = null, extra = null }) {
    {
      const body = {
        messages,
        max_tokens: maxTokens,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        repeat_penalty: sampling.repeat_penalty,
        seed,
        cache_prompt: false,
        ...(extra || {})
      };
      if (logprobs) {
        body.logprobs = true;
        body.top_logprobs = logprobs === true ? 5 : logprobs;
      }
      const data = await server.request("/v1/chat/completions", body, { signal });
      if (server.shared) await server.drainStderr({ untilGen });
      const choice = data?.choices?.[0] || {};
      return {
        content: choice.message?.content ?? "",
        logprobs: choice.logprobs?.content ?? null,
        usage: { promptTokens: data?.usage?.prompt_tokens ?? null, completionTokens: data?.usage?.completion_tokens ?? null },
        finishReason: choice.finish_reason ?? null,
        engine: {
          injectionEnv,
          shared: server.shared,
          active: server.audit.active,
          sitesFired: [...server.audit.sitesFired],
          warnings: [...server.audit.warnings]
        }
      };
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
      shared: s.shared,
      injectionEnv: s.shared ? parseEnvText(s.configText) : s.injectionEnv,
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

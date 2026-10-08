import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { TECHNIQUE_IDS, describeTechnique, DEFAULT_DOSES, DEFAULT_SAMPLING } from "../src/catalog.js";
import { runAgent, runDoseResponse, runTrial, validateInput, LIMITS, DEFAULT_SYSTEM } from "./experiment.js";

const BODY_LIMIT = 256 * 1024;
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};
const SECURITY_HEADERS = {
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY",
  "referrer-policy": "no-referrer",
  "cross-origin-resource-policy": "same-origin",
  "content-security-policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function send(res, status, body, headers = {}) {
  const isJson = typeof body !== "string" && !Buffer.isBuffer(body);
  res.writeHead(status, {
    ...SECURITY_HEADERS,
    "cache-control": "no-store",
    ...(isJson ? { "content-type": "application/json; charset=utf-8" } : {}),
    ...headers
  });
  res.end(isJson ? JSON.stringify(body) : body);
}

// Only loopback host names are accepted (blocks DNS-rebinding), and cross-origin writes are refused.
function checkOrigin(req, port) {
  const allowed = new Set([`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`]);
  if (!allowed.has(String(req.headers.host || "").toLowerCase())) throw new HttpError(403, "Forbidden host");
  const origin = req.headers.origin;
  if (origin && origin !== "null") {
    let host;
    try { host = new URL(origin).host.toLowerCase(); } catch { throw new HttpError(403, "Bad origin"); }
    if (!allowed.has(host)) throw new HttpError(403, "Cross-origin request refused");
  } else if (origin === "null") {
    throw new HttpError(403, "Cross-origin request refused");
  }
}

async function readBody(req) {
  if (!/^application\/json\b/i.test(String(req.headers["content-type"] || ""))) {
    throw new HttpError(415, "Content-Type must be application/json");
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > BODY_LIMIT) throw new HttpError(413, "Request body too large");
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  } catch {
    throw new HttpError(400, "Invalid JSON");
  }
}

// Serial job queue: model runs share CPU/GPU, so jobs execute one at a time.
export class JobQueue {
  constructor() {
    this.jobs = new Map();
    this.chain = Promise.resolve();
  }

  submit(kind, fn) {
    const id = `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
    const controller = new AbortController();
    const job = { id, kind, status: "queued", progress: { done: 0, total: 1, message: "queued" }, result: null, error: null, controller, created: Date.now() };
    this.jobs.set(id, job);
    this.chain = this.chain.then(async () => {
      if (controller.signal.aborted) {
        job.status = "cancelled";
        return;
      }
      job.status = "running";
      try {
        job.result = await fn({ signal: controller.signal, progress: (p) => { job.progress = p; } });
        job.status = "done";
      } catch (error) {
        job.status = controller.signal.aborted ? "cancelled" : "error";
        job.error = error.message;
      }
    });
    this.prune();
    return job;
  }

  cancel(id) {
    const job = this.jobs.get(id);
    if (!job) return null;
    if (job.status === "queued" || job.status === "running") job.controller.abort(new Error("Cancelled by user"));
    if (job.status === "queued") job.status = "cancelled";
    return job;
  }

  prune(keep = 50) {
    const finished = [...this.jobs.values()].filter((j) => !["queued", "running"].includes(j.status));
    for (const j of finished.slice(0, Math.max(0, finished.length - keep))) this.jobs.delete(j.id);
  }

  view(job) {
    const { controller, ...rest } = job;
    return rest;
  }
}

export function createApp(lab, config) {
  const jobs = new JobQueue();
  const staticRoot = path.resolve(config.staticDir);
  const runners = { run: runTrial, "dose-response": runDoseResponse, agent: runAgent };

  async function api(req, res, url) {
    const route = `${req.method} ${url.pathname}`;
    if (route === "GET /api/status") {
      return send(res, 200, {
        llamaServer: { available: await lab.pool.binaryAvailable(), servers: lab.pool.status().map(({ model, ...s }) => ({ model: path.basename(model), ...s })) },
        ollamaUrl: config.ollamaUrl,
        limits: LIMITS
      });
    }
    if (route === "GET /api/catalog") {
      return send(res, 200, {
        techniques: TECHNIQUE_IDS.map(describeTechnique),
        defaults: { doses: DEFAULT_DOSES, sampling: DEFAULT_SAMPLING, system: DEFAULT_SYSTEM }
      });
    }
    if (route === "GET /api/models") return send(res, 200, await lab.registry.refresh());
    if (route === "GET /api/history") {
      const limit = Math.min(500, Math.max(1, Number.parseInt(url.searchParams.get("limit") || "50", 10) || 50));
      return send(res, 200, { items: await lab.history.list(limit) });
    }
    const historyMatch = /^GET \/api\/history\/([\w-]{1,80})$/.exec(route);
    if (historyMatch) {
      const item = await lab.history.get(historyMatch[1]);
      return item ? send(res, 200, item) : send(res, 404, { error: "Not found" });
    }
    const runMatch = /^POST \/api\/(run|dose-response|agent)$/.exec(route);
    if (runMatch) {
      const kind = runMatch[1];
      const input = validateInput(await readBody(req), kind);
      if (input.backend === "llamacpp") {
        if (!lab.registry.gguf.size) await lab.registry.refresh();
        try { lab.registry.resolveGguf(input.modelId); } catch (error) { throw new HttpError(400, error.message); }
      }
      const job = jobs.submit(kind, async (ctx) => {
        const record = await runners[kind](lab, input, ctx);
        await lab.history.append(record);
        return record;
      });
      return send(res, 202, jobs.view(job));
    }
    const jobMatch = /^(GET|POST) \/api\/jobs\/([\w-]{1,80})(\/cancel)?$/.exec(route);
    if (jobMatch) {
      if (jobMatch[1] === "POST") {
        if (!jobMatch[3]) throw new HttpError(404, "Not found");
        await readBody(req);
        const job = jobs.cancel(jobMatch[2]);
        return job ? send(res, 200, jobs.view(job)) : send(res, 404, { error: "Unknown job" });
      }
      if (jobMatch[3]) throw new HttpError(405, "Use POST");
      const job = jobs.jobs.get(jobMatch[2]);
      return job ? send(res, 200, jobs.view(job)) : send(res, 404, { error: "Unknown job" });
    }
    throw new HttpError(404, "Unknown API route");
  }

  async function staticFile(req, res, url) {
    if (req.method !== "GET" && req.method !== "HEAD") throw new HttpError(405, "Method not allowed");
    let rel;
    try { rel = decodeURIComponent(url.pathname); } catch { throw new HttpError(400, "Bad path"); }
    if (rel === "/") rel = "/index.html";
    const file = path.resolve(staticRoot, `.${rel}`);
    if (!file.startsWith(staticRoot + path.sep) || rel.includes("\0")) throw new HttpError(403, "Forbidden");
    const type = MIME[path.extname(file).toLowerCase()];
    if (!type) throw new HttpError(404, "Not found");
    let data;
    try { data = await readFile(file); } catch { throw new HttpError(404, "Not found"); }
    send(res, 200, req.method === "HEAD" ? "" : data, { "content-type": type });
  }

  const server = http.createServer(async (req, res) => {
    try {
      checkOrigin(req, server.address()?.port ?? config.port);
      const url = new URL(req.url, "http://localhost");
      if (url.pathname.startsWith("/api/")) await api(req, res, url);
      else await staticFile(req, res, url);
    } catch (error) {
      const status = error.status || 500;
      if (status === 500) console.error(error);
      if (!res.headersSent) send(res, status, { error: error.message });
      else res.end();
    }
  });
  server.requestTimeout = 120000;
  return { server, jobs };
}

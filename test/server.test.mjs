import assert from "node:assert/strict";
import http from "node:http";
import { after, before, test } from "node:test";
import path from "node:path";
import * as metrics from "../src/metrics.js";
import { TECHNIQUE_IDS } from "../src/catalog.js";
import { ROOT } from "../lib/config.js";
import { createApp } from "../lib/server.js";

const GGUF = "gguf:0123456789abcdef";
let server;
let port;
const appended = [];

const lab = {
  metrics,
  registry: {
    gguf: new Map([[GGUF, {}]]),
    resolveGguf(id) {
      if (id !== GGUF) throw new Error("Unknown model id");
      return { file: "/m.gguf", name: "m.gguf" };
    },
    refresh: async () => ({ gguf: [], ollama: [] })
  },
  pool: {
    binaryAvailable: async () => false,
    status: () => [],
    chat: async (file, env) => ({ content: Object.keys(env).length > 1 ? "zz qq" : "hello there", logprobs: null }),
    tokenIds: async () => []
  },
  history: {
    append: async (r) => { appended.push(r); },
    list: async () => [{ id: "run-1" }],
    get: async (id) => (id === "run-1" ? { id } : null)
  }
};

function request(pathname, { method = "GET", headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: "127.0.0.1", port, path: pathname, method, headers: { host: `127.0.0.1:${port}`, ...headers } }, (res) => {
      let data = "";
      res.on("data", (c) => { data += c; });
      res.on("end", () => {
        let json = null;
        try { json = JSON.parse(data); } catch { /* not json */ }
        resolve({ status: res.statusCode, headers: res.headers, text: data, json });
      });
    });
    req.on("error", reject);
    if (body !== undefined) req.write(body);
    req.end();
  });
}

const post = (p, obj, headers = {}) =>
  request(p, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: typeof obj === "string" ? obj : JSON.stringify(obj) });

before(async () => {
  ({ server } = createApp(lab, { staticDir: path.join(ROOT, "src"), port: 0, ollamaUrl: "http://127.0.0.1:1" }));
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  port = server.address().port;
});

after(() => new Promise((r) => server.close(r)));

test("serves the UI with a strict CSP", async () => {
  const r = await request("/");
  assert.equal(r.status, 200);
  assert.match(r.headers["content-type"], /text\/html/);
  assert.match(r.headers["content-security-policy"], /script-src 'self'/);
  assert.equal(r.headers["x-frame-options"], "DENY");
  assert.doesNotMatch(r.text, /<script>|style="/, "no inline scripts or styles");
  assert.equal((await request("/catalog.js")).status, 200);
});

test("rejects DNS-rebinding hosts and cross-origin requests", async () => {
  assert.equal((await request("/api/status", { headers: { host: "evil.example" } })).status, 403);
  assert.equal((await request("/api/status", { headers: { host: `attacker.com:${port}` } })).status, 403);
  assert.equal((await post("/api/run", {}, { origin: "https://evil.example" })).status, 403);
  assert.equal((await post("/api/run", {}, { origin: "null" })).status, 403);
  assert.equal((await request("/api/status", { headers: { origin: `http://localhost:${port}`, host: `localhost:${port}` } })).status, 200);
});

test("static paths cannot escape the web root", async () => {
  for (const p of ["/../package.json", "/%2e%2e/package.json", "/..%5cpackage.json", "/%2e%2e%2flib%2fserver.js"]) {
    const r = await request(p);
    assert.ok([403, 404].includes(r.status), `${p} -> ${r.status}`);
  }
  assert.ok([403, 404].includes((await request("/index.html%00.js")).status));
  assert.equal((await request("/missing.exe")).status, 404);
});

test("write endpoints require JSON, a size limit and valid input", async () => {
  assert.equal((await request("/api/run", { method: "POST", headers: { "content-type": "text/plain" }, body: "{}" })).status, 415);
  assert.equal((await post("/api/run", "{bad")).status, 400);
  assert.equal((await post("/api/run", "x".repeat(300 * 1024))).status, 413);
  const pathModel = await post("/api/run", { backend: "llamacpp", modelId: "D:\\models\\x.gguf", techniqueId: "delirium", prompt: "hi" });
  assert.equal(pathModel.status, 400);
  const unknown = await post("/api/run", { backend: "llamacpp", modelId: "gguf:ffffffffffffffff", techniqueId: "delirium", prompt: "hi" });
  assert.equal(unknown.status, 400);
  assert.match(unknown.json.error, /Unknown model/);
  assert.equal((await request("/api/run")).status, 404);
});

test("catalog, status and history routes", async () => {
  const cat = await request("/api/catalog");
  assert.equal(cat.json.techniques.length, TECHNIQUE_IDS.length);
  assert.ok(cat.json.defaults.sampling);
  const status = await request("/api/status");
  assert.equal(status.json.llamaServer.available, false);
  assert.equal((await request("/api/history?limit=abc")).json.items.length, 1);
  assert.equal((await request("/api/history/run-1")).status, 200);
  assert.equal((await request("/api/history/nope")).status, 404);
});

test("job lifecycle: submit, poll, result saved to history", async () => {
  const sub = await post("/api/run", { backend: "llamacpp", modelId: GGUF, techniqueId: "delirium", doseMg: 300, prompt: "hi", sampling: { temperature: 0 } });
  assert.equal(sub.status, 202);
  assert.equal(sub.json.controller, undefined);
  let job;
  for (let i = 0; i < 50; i += 1) {
    job = (await request(`/api/jobs/${sub.json.id}`)).json;
    if (job.status === "done" || job.status === "error") break;
    await new Promise((r) => setTimeout(r, 20));
  }
  assert.equal(job.status, "done", job.error);
  assert.equal(job.result.type, "run");
  assert.ok(job.result.metrics.impairment > 0);
  assert.equal(appended.at(-1).id, job.result.id);
  assert.equal((await post(`/api/jobs/${sub.json.id}/cancel`, {})).status, 200);
  assert.equal((await request(`/api/jobs/${sub.json.id}/cancel`)).status, 405);
  assert.equal((await request("/api/jobs/job-unknown")).status, 404);
});

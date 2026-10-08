#!/usr/bin/env node
// Starts the lab web UI + API on 127.0.0.1 (loopback only).
import { loadConfig } from "../lib/config.js";
import { Lab } from "../lib/lab.js";
import { createApp } from "../lib/server.js";

const config = loadConfig();
const lab = new Lab(config);
const { server } = createApp(lab, config);

const shutdown = () => {
  lab.shutdown();
  server.close();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("exit", () => lab.shutdown());

const models = await lab.registry.refresh();
server.listen(config.port, config.host, () => {
  console.log(`LLM Injection Runtime Lab: http://localhost:${server.address().port}`);
  console.log(`  Ollama: ${config.ollamaUrl} (${models.ollamaError ? `unavailable: ${models.ollamaError}` : `${models.ollama.length} models`})`);
  console.log(`  llama-server: ${config.llamaServerBin}`);
  console.log(`  GGUF models: ${models.gguf.length} (from ${config.modelsDirs.join(", ")} and Ollama blobs)`);
});
server.on("error", (error) => {
  console.error(error.code === "EADDRINUSE" ? `Port ${config.port} is in use. Set PORT=<n>.` : error.message);
  process.exit(1);
});

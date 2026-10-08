import { createServer } from "node:http";
import { appendFile, mkdir, readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { extname, join, normalize } from "node:path";
import {
  DEFAULT_DOSES,
  aggregateDoseRows,
  buildEvaluationMessages,
  createDrugProfile,
  createRuntimePerturbationOptions,
  evaluateRun,
  fingerprintMessages,
  optionDiff,
  parseDoseList
} from "../src/drugs.js";

const root = join(process.cwd(), "src");
const dataDir = join(process.cwd(), "data");
const historyPath = join(dataDir, "runs.jsonl");
const preferredPort = Number.parseInt(process.env.PORT || "4173", 10);
const ollamaHost = (process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8"
};

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host}`);
    if (url.pathname.startsWith("/api/")) {
      await routeApi(request, response, url);
      return;
    }

    const requestedPath = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    const normalized = normalize(requestedPath);
    if (normalized.startsWith("..")) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    const filePath = join(root, normalized);
    const body = await readFile(filePath);
    response.writeHead(200, { "content-type": types[extname(filePath)] || "text/plain; charset=utf-8" });
    response.end(body);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
});

await listenOnAvailablePort(server, preferredPort);

async function routeApi(request, response, url) {
  if (request.method === "GET" && url.pathname === "/api/models") {
    await handleModels(response);
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/history") {
    await handleHistory(response, url);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/run") {
    await handleRun(request, response);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/dose-response") {
    await handleDoseResponse(request, response);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/chaos-agent") {
    await handleChaosAgent(request, response);
    return;
  }

  sendJson(response, 404, { error: "Unknown API route." });
}

async function handleModels(response) {
  try {
    const result = await fetch(`${ollamaHost}/api/tags`);
    if (!result.ok) {
      throw new Error(`Ollama returned ${result.status}: ${await result.text()}`);
    }
    const data = await result.json();
    sendJson(response, 200, {
      host: ollamaHost,
      models: Array.isArray(data.models)
        ? data.models.map((model) => ({
            name: model.name,
            size: model.size,
            modified_at: model.modified_at
          }))
        : []
    });
  } catch (error) {
    sendJson(response, 503, {
      host: ollamaHost,
      error: `Could not reach Ollama. Start it with "ollama serve" or set OLLAMA_HOST. ${error.message}`
    });
  }
}

async function handleRun(request, response) {
  try {
    const input = await readJson(request);
    validateRunInput(input);
    const record = await runRuntimeTrial(input);
    await appendHistory({ type: "run", ...record });
    sendJson(response, 200, record);
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
}

async function handleDoseResponse(request, response) {
  try {
    const input = await readJson(request);
    validateRunInput(input);
    const doses = parseDoseList(input.doses || DEFAULT_DOSES).slice(0, 10);
    const trials = clampInteger(input.trials || 1, 1, 5);
    const rows = [];

    for (const doseMg of doses) {
      for (let trial = 0; trial < trials; trial += 1) {
        const row = await runRuntimeTrial({
          ...input,
          doseMg,
          seed: `${input.seed || "llm-drugs"}:dose-${doseMg}:trial-${trial}`
        });
        rows.push({
          doseMg,
          trial,
          response: row.response,
          metrics: row.metrics,
          audit: row.audit
        });
      }
    }

    const result = {
      id: createId("dose"),
      timestamp: new Date().toISOString(),
      type: "dose-response",
      model: input.model,
      drugId: input.drugId,
      prompt: input.prompt,
      expected: input.expected || "",
      trials,
      rows,
      summary: aggregateDoseRows(rows)
    };
    await appendHistory(result);
    sendJson(response, 200, result);
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
}

async function handleChaosAgent(request, response) {
  try {
    const input = await readJson(request);
    if (normalizeBackend(input.backend) === "llamacpp") {
      if (!String(input.llamaCliPath || "").trim() || !String(input.llamaModelPath || "").trim()) {
        throw new Error("Set patched llama-cli.exe and GGUF model paths before running llama.cpp agent chaos.");
      }
    } else {
      validateModel(input.model);
    }
    const objective = String(input.objective || input.prompt || "").trim();
    if (!objective) {
      throw new Error("Enter an agent objective.");
    }
    const steps = clampInteger(input.steps || 4, 1, 8);
    const profile = createDrugProfile(input);
    const impairedOptions = createRuntimePerturbationOptions(profile, "impaired");
    const transcript = [];

    for (let step = 0; step < steps; step += 1) {
      const messages = buildEvaluationMessages({
        prompt: [
          `Agent objective:\n${objective}`,
          transcript.length ? `Previous steps:\n${transcript.map((item) => `Step ${item.step}: ${item.output}`).join("\n\n")}` : "",
          "Return the next concrete step, any tool/check you would use, and the current risk."
        ].filter(Boolean).join("\n\n"),
        memory: input.memory || ""
      });
      const output = await callBackend({
        input,
        messages,
        options: {
          ...impairedOptions,
          seed: createRuntimePerturbationOptions(createDrugProfile({ ...input, seed: `${input.seed || "agent"}:${step}` }), "impaired").seed
        },
        profile: createDrugProfile({ ...input, seed: `${input.seed || "agent"}:${step}` }),
        label: `agent-step-${step}`,
        drugEnabled: true
      });
      transcript.push({
        step: step + 1,
        messageFingerprint: fingerprintMessages(messages),
        output,
        metrics: evaluateRun({
          prompt: objective,
          baseline: transcript[step - 1]?.output || "",
          impaired: output,
          profile,
          expected: input.expected || ""
        })
      });
    }

    const result = {
      id: createId("agent"),
      timestamp: new Date().toISOString(),
      type: "agent-chaos",
      model: normalizeBackend(input.backend) === "llamacpp" ? input.llamaModelPath : input.model,
      backend: normalizeBackend(input.backend),
      profile,
      objective,
      steps: transcript,
      audit: {
        backend: normalizeBackend(input.backend) === "llamacpp" ? "llamacpp-cli-logit-hook" : "ollama-runtime",
        options: impairedOptions
      }
    };
    await appendHistory(result);
    sendJson(response, 200, result);
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
}

async function handleHistory(response, url) {
  try {
    const limit = clampInteger(url.searchParams.get("limit") || 25, 1, 200);
    const raw = await readFile(historyPath, "utf8").catch(() => "");
    const items = raw
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .slice(-limit)
      .reverse();
    sendJson(response, 200, { items });
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
}

async function runRuntimeTrial(input) {
  const profile = createDrugProfile(input);
  const messages = buildEvaluationMessages(input);
  const baselineOptions = createRuntimePerturbationOptions(profile, "baseline");
  const impairedOptions = createRuntimePerturbationOptions(profile, "impaired");
  const messageFingerprint = fingerprintMessages(messages);
  const backend = normalizeBackend(input.backend);
  const baseline = await callBackend({
    input,
    messages,
    options: baselineOptions,
    profile: createDrugProfile({ ...input, doseMg: 0 }),
    label: "baseline",
    drugEnabled: false
  });
  const response = profile.intensity === 0
    ? baseline
    : await callBackend({
        input,
        messages,
        options: impairedOptions,
        profile,
        label: "runtime-drug",
        drugEnabled: true
      });
  const metrics = evaluateRun({
    prompt: input.prompt,
    expected: input.expected || "",
    baseline,
    impaired: response,
    profile
  });
  const judge = input.judgeEnabled ? await judgeRun({ input, profile, baseline, response, metrics }) : null;

  return {
    id: createId("run"),
    timestamp: new Date().toISOString(),
    type: "run",
    model: backend === "llamacpp" ? input.llamaModelPath : input.model,
    backend,
    profile,
    prompt: input.prompt,
    expected: input.expected || "",
    baseline: { answer: baseline },
    response,
    metrics,
    judge,
    audit: {
      backend: backend === "llamacpp" ? "llamacpp-cli-logit-hook" : "ollama-runtime",
      messageFingerprint,
      messagesIdentical: true,
      baselineOptions,
      impairedOptions,
      llamaCliPath: backend === "llamacpp" ? input.llamaCliPath : undefined,
      llamaModelPath: backend === "llamacpp" ? input.llamaModelPath : undefined,
      optionDiff: optionDiff(baselineOptions, impairedOptions)
    }
  };
}

async function judgeRun({ input, profile, baseline, response, metrics }) {
  if (normalizeBackend(input.backend) === "llamacpp" && !input.model) {
    return {
      error: "Evaluator skipped for llama.cpp backend because no Ollama judge model is selected."
    };
  }
  const judgeModel = input.judgeModel || input.model;
  const prompt = [
    "Score this runtime-perturbed LLM response as JSON only.",
    "Fields: impairment_detected boolean, factual_survival 0-100, coherence 0-100, notes short string.",
    `Drug: ${profile.drug.name}`,
    `Dose: ${profile.doseMg} mg`,
    `Expected anchor: ${input.expected || "(none)"}`,
    `Heuristic metrics: ${JSON.stringify(metrics)}`,
    `Baseline:\n${baseline}`,
    `Runtime-perturbed response:\n${response}`
  ].join("\n\n");

  try {
    const content = await callOllamaChat({
      model: judgeModel,
      messages: buildEvaluationMessages({ prompt }),
      options: {
        temperature: 0.1,
        top_p: 0.85,
        top_k: 20,
        repeat_penalty: 1.1,
        seed: 90125,
        num_ctx: 4096,
        num_predict: 320
      },
      label: "judge"
    });
    return {
      model: judgeModel,
      raw: content,
      parsed: parseJudgeJson(content)
    };
  } catch (error) {
    return {
      model: judgeModel,
      error: error.message
    };
  }
}

async function callOllamaChat({ model, messages, options, label }) {
  const result = await fetch(`${ollamaHost}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options
    })
  });

  if (!result.ok) {
    throw new Error(`Ollama ${label} call failed with ${result.status}: ${await result.text()}`);
  }

  const data = await result.json();
  const content = data?.message?.content || data?.response || "";
  if (!content.trim()) {
    throw new Error(`Ollama ${label} call returned an empty response.`);
  }
  return content.trim();
}

async function callBackend({ input, messages, options, profile, label, drugEnabled }) {
  if (normalizeBackend(input.backend) === "llamacpp") {
    return callLlamaCppCli({ input, messages, profile, label, drugEnabled });
  }
  return callOllamaChat({
    model: input.model,
    messages,
    options,
    label
  });
}

async function callLlamaCppCli({ input, messages, profile, label, drugEnabled }) {
  const llamaCliPath = String(input.llamaCliPath || "").trim();
  const llamaModelPath = String(input.llamaModelPath || "").trim();
  if (!llamaCliPath) {
    throw new Error("Set the patched llama-cli.exe path before using the llama.cpp backend.");
  }
  if (!llamaModelPath) {
    throw new Error("Set the GGUF model path before using the llama.cpp backend.");
  }

  const prompt = flattenMessagesForCli(messages);
  const maxTokens = clampInteger(input.llamaMaxTokens || 160, 16, 2048);
  const timeoutMs = clampInteger(input.llamaTimeoutSeconds || 900, 60, 3600) * 1000;
  const args = [
    "-m",
    llamaModelPath,
    "-p",
    prompt,
    "-n",
    String(maxTokens),
    "--no-display-prompt"
  ];
  const env = {
    ...process.env,
    LLM_DRUG_KIND: profile.drugId,
    LLM_DRUG_DOSE_MG: drugEnabled ? String(profile.doseMg) : "0",
    LLM_DRUG_SEED: String(seedFromString(`${profile.seed}:${label}`))
  };

  const result = await runProcess(llamaCliPath, args, env, timeoutMs);
  const output = stripLlamaNoise(result.stdout);
  if (!output.trim()) {
    throw new Error(`llama.cpp ${label} call returned no text. stderr: ${result.stderr.slice(0, 1000)}`);
  }
  return output.trim();
}

function flattenMessagesForCli(messages) {
  return messages
    .map((message) => `${message.role.toUpperCase()}:\n${message.content}`)
    .join("\n\n");
}

function stripLlamaNoise(value) {
  return String(value || "")
    .split(/\r?\n/)
    .filter((line) => !/^llama_/.test(line) && !/^ggml_/.test(line) && !/^main:/.test(line))
    .join("\n")
    .trim();
}

function runProcess(command, args, env, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env,
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error([
        `Command timed out after ${Math.round(timeoutMs / 1000)}s: ${command}`,
        "For large 30B models, increase the llama.cpp timeout in the UI or lower max tokens.",
        stderr ? `stderr before timeout:\n${stderr.slice(-2000)}` : "",
        stdout ? `stdout before timeout:\n${stdout.slice(-2000)}` : ""
      ].filter(Boolean).join("\n\n")));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Command failed with exit code ${code}: ${stderr || stdout}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

async function appendHistory(record) {
  await mkdir(dataDir, { recursive: true });
  await appendFile(historyPath, `${JSON.stringify(record)}\n`, "utf8");
}

async function readJson(request) {
  let raw = "";
  for await (const chunk of request) {
    raw += chunk;
    if (raw.length > 1_000_000) {
      throw new Error("Request body is too large.");
    }
  }
  return raw ? JSON.parse(raw) : {};
}

function validateRunInput(input) {
  if (normalizeBackend(input.backend) === "llamacpp") {
    if (!String(input.llamaCliPath || "").trim()) {
      throw new Error("Set the patched llama-cli.exe path.");
    }
    if (!String(input.llamaModelPath || "").trim()) {
      throw new Error("Set the GGUF model path.");
    }
  } else {
    validateModel(input.model);
  }
  if (!String(input.prompt || "").trim()) {
    throw new Error("Enter a prompt to test.");
  }
}

function validateModel(model) {
  if (!model || typeof model !== "string") {
    throw new Error("Select an Ollama model before running the test.");
  }
}

function parseJudgeJson(content) {
  const match = String(content).match(/\{[\s\S]*\}/);
  if (!match) {
    return null;
  }
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

function clampInteger(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.min(max, Math.max(min, parsed));
}

function normalizeBackend(backend) {
  return backend === "llamacpp" ? "llamacpp" : "ollama";
}

function seedFromString(value) {
  let hash = 0;
  for (const char of String(value)) {
    hash = Math.imul(31, hash) + char.charCodeAt(0);
    hash |= 0;
  }
  return Math.abs(hash || 1);
}

function createId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function sendJson(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

function listenOnAvailablePort(server, startPort) {
  const maxAttempts = 20;

  return new Promise((resolve, reject) => {
    let attempt = 0;

    const tryListen = (port) => {
      const onError = (error) => {
        server.off("listening", onListening);
        if (error.code === "EADDRINUSE" && attempt < maxAttempts) {
          attempt += 1;
          const nextPort = port + 1;
          console.warn(`Port ${port} is in use; trying ${nextPort}.`);
          tryListen(nextPort);
          return;
        }
        reject(error);
      };

      const onListening = () => {
        server.off("error", onError);
        console.log(`LLM Drugs runtime lab running at http://localhost:${port}`);
        console.log(`Ollama API target: ${ollamaHost}`);
        resolve();
      };

      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(port);
    };

    tryListen(startPort);
  });
}

import { DRUGS, createDrugProfile, describeDose } from "./drugs.js";

const elements = {
  backend: document.querySelector("#backend"),
  ollamaFields: document.querySelector("#ollama-fields"),
  llamacppFields: document.querySelector("#llamacpp-fields"),
  model: document.querySelector("#model"),
  llamaCliPath: document.querySelector("#llama-cli-path"),
  llamaModelPath: document.querySelector("#llama-model-path"),
  llamaTimeoutSeconds: document.querySelector("#llama-timeout-seconds"),
  llamaMaxTokens: document.querySelector("#llama-max-tokens"),
  refreshModels: document.querySelector("#refresh-models"),
  status: document.querySelector("#ollama-status"),
  drug: document.querySelector("#drug"),
  drugSummary: document.querySelector("#drug-summary"),
  dose: document.querySelector("#dose"),
  doseNumber: document.querySelector("#dose-number"),
  doseLabel: document.querySelector("#dose-label"),
  seed: document.querySelector("#seed"),
  judgeEnabled: document.querySelector("#judge-enabled"),
  prompt: document.querySelector("#prompt"),
  expected: document.querySelector("#expected"),
  memory: document.querySelector("#memory"),
  doses: document.querySelector("#doses"),
  trials: document.querySelector("#trials"),
  objective: document.querySelector("#objective"),
  run: document.querySelector("#run"),
  doseResponse: document.querySelector("#dose-response"),
  chaosAgent: document.querySelector("#chaos-agent"),
  baselineMeta: document.querySelector("#baseline-meta"),
  impairedMeta: document.querySelector("#impaired-meta"),
  copyBaseline: document.querySelector("#copy-baseline"),
  copyImpaired: document.querySelector("#copy-impaired"),
  baselineOutput: document.querySelector("#baseline-output"),
  impairedOutput: document.querySelector("#impaired-output"),
  metrics: document.querySelector("#metrics"),
  auditLog: document.querySelector("#audit-log"),
  doseSummary: document.querySelector("#dose-summary"),
  doseChart: document.querySelector("#dose-chart"),
  doseBody: document.querySelector("#dose-body"),
  doseTrials: document.querySelector("#dose-trials"),
  agentTrace: document.querySelector("#agent-trace"),
  refreshHistory: document.querySelector("#refresh-history"),
  historyBody: document.querySelector("#history-body")
};

function init() {
  elements.drug.innerHTML = Object.entries(DRUGS)
    .map(([id, drug]) => `<option value="${id}">${drug.name}</option>`)
    .join("");
  bindEvents();
  updateDrugPreview();
  updateBackendFields();
  void loadModels();
  void loadHistory();
}

function bindEvents() {
  elements.backend.addEventListener("input", updateBackendFields);
  elements.refreshModels.addEventListener("click", loadModels);
  elements.run.addEventListener("click", runSingle);
  elements.doseResponse.addEventListener("click", runDoseResponse);
  elements.chaosAgent.addEventListener("click", runAgentChaos);
  elements.copyBaseline.addEventListener("click", () => copyText(elements.baselineOutput.textContent, "Baseline copied."));
  elements.copyImpaired.addEventListener("click", () => copyText(elements.impairedOutput.textContent, "Runtime-drugged response copied."));
  elements.refreshHistory.addEventListener("click", loadHistory);
  elements.drug.addEventListener("input", updateDrugPreview);
  elements.dose.addEventListener("input", () => {
    elements.doseNumber.value = elements.dose.value;
    updateDrugPreview();
  });
  elements.doseNumber.addEventListener("input", () => {
    elements.dose.value = elements.doseNumber.value;
    updateDrugPreview();
  });
}

async function loadModels() {
  if (elements.backend.value === "llamacpp") {
    setStatus("llama.cpp backend selected. Set paths, then run from the UI.", "pending");
    return;
  }
  setStatus("Checking Ollama...", "pending");
  elements.refreshModels.disabled = true;
  try {
    const data = await getJson("/api/models");
    if (!data.models.length) {
      elements.model.innerHTML = "<option value=\"\">No Ollama models found</option>";
      setStatus(`Connected to ${data.host}, but no models are installed.`, "error");
      return;
    }
    elements.model.innerHTML = data.models
      .map((model) => `<option value="${escapeHtml(model.name)}">${escapeHtml(model.name)}</option>`)
      .join("");
    setStatus(`Connected to ${data.host}. ${data.models.length} model${data.models.length === 1 ? "" : "s"} available.`, "ok");
  } catch (error) {
    elements.model.innerHTML = "<option value=\"\">Ollama unavailable</option>";
    setStatus(error.message, "error");
  } finally {
    elements.refreshModels.disabled = false;
  }
}

async function runSingle() {
  setBusy(true, "Running baseline and runtime-drugged calls...");
  elements.baselineOutput.textContent = "Running baseline...";
  elements.impairedOutput.textContent = "Running runtime perturbation...";
  elements.baselineMeta.textContent = "Calling selected model with 0 mg runtime settings...";
  elements.impairedMeta.textContent = "Calling selected model with runtime drug settings...";

  try {
    const result = await postJson("/api/run", currentInput());
    elements.baselineOutput.textContent = result.baseline.answer;
    elements.impairedOutput.textContent = result.response;
    elements.baselineMeta.textContent = `${labelModel(result)} · 0 mg · ${result.audit.messageFingerprint}`;
    elements.impairedMeta.textContent = `${result.profile.drug.name} ${result.profile.doseMg} mg · survival ${result.metrics.survivalScore}% · impairment ${result.metrics.impairmentScore}%`;
    renderMetrics(result.metrics, result.judge);
    renderAudit(result.audit);
    setStatus(`Run complete. Messages identical: ${result.audit.messagesIdentical ? "yes" : "no"}.`, "ok");
    await loadHistory();
  } catch (error) {
    setStatus(error.message, "error");
    renderAudit({ error: error.message });
  } finally {
    setBusy(false);
  }
}

async function runDoseResponse() {
  setBusy(true, "Running dose-response batch...");
  elements.doseSummary.textContent = "Running...";
  elements.doseBody.innerHTML = "<tr><td colspan=\"6\">Batch in progress. Large local models can take a while.</td></tr>";
  elements.doseTrials.innerHTML = "";

  try {
    const result = await postJson("/api/dose-response", currentInput());
    renderDoseResponse(result);
    setStatus(`Dose-response complete: ${result.rows.length} real calls stored.`, "ok");
    await loadHistory();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function runAgentChaos() {
  setBusy(true, "Running impaired agent loop...");
  elements.agentTrace.textContent = "Agent chaos run in progress...";

  try {
    const result = await postJson("/api/chaos-agent", {
      ...currentInput(),
      objective: elements.objective.value,
      steps: 4
    });
    elements.agentTrace.innerHTML = result.steps
      .map((step) => `
        <section class="trace-step">
          <strong>Step ${step.step}</strong>
          <p>Fingerprint: ${escapeHtml(step.messageFingerprint)}</p>
          <pre>${escapeHtml(step.output)}</pre>
          <p>Coherence ${step.metrics.coherence}% · Impairment ${step.metrics.impairmentScore}% · Survival ${step.metrics.survivalScore}%</p>
        </section>
      `)
      .join("");
    setStatus("Agent chaos run complete.", "ok");
    await loadHistory();
  } catch (error) {
    setStatus(error.message, "error");
    elements.agentTrace.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function loadHistory() {
  try {
    const data = await getJson("/api/history?limit=25");
    elements.historyBody.innerHTML = data.items.length
      ? data.items.map(renderHistoryRow).join("")
      : "<tr><td colspan=\"6\">No persisted runs yet.</td></tr>";
  } catch (error) {
    elements.historyBody.innerHTML = `<tr><td colspan="6">${escapeHtml(error.message)}</td></tr>`;
  }
}

function currentInput() {
  return {
    model: elements.model.value,
    backend: elements.backend.value,
    llamaCliPath: elements.llamaCliPath.value,
    llamaModelPath: elements.llamaModelPath.value,
    llamaTimeoutSeconds: elements.llamaTimeoutSeconds.value,
    llamaMaxTokens: elements.llamaMaxTokens.value,
    drugId: elements.drug.value,
    doseMg: elements.doseNumber.value,
    seed: elements.seed.value,
    prompt: elements.prompt.value,
    expected: elements.expected.value,
    memory: elements.memory.value,
    doses: elements.doses.value,
    trials: elements.trials.value,
    judgeEnabled: elements.judgeEnabled.checked
  };
}

function updateBackendFields() {
  const isLlamaCpp = elements.backend.value === "llamacpp";
  elements.ollamaFields.classList.toggle("hidden", isLlamaCpp);
  elements.llamacppFields.classList.toggle("hidden", !isLlamaCpp);
  elements.judgeEnabled.disabled = isLlamaCpp;
  if (isLlamaCpp) {
    elements.judgeEnabled.checked = false;
    setStatus("llama.cpp backend selected. It runs patched llama-cli.exe directly from the UI.", "pending");
  } else {
    void loadModels();
  }
}

function updateDrugPreview() {
  const profile = createDrugProfile({
    drugId: elements.drug.value,
    doseMg: elements.doseNumber.value,
    seed: elements.seed.value
  });
  elements.drugSummary.textContent = profile.drug.focus;
  elements.doseLabel.textContent = `${profile.doseMg} mg - ${describeDose(profile.doseMg)}`;
}

function labelModel(result) {
  if (result.backend === "llamacpp") {
    return `llama.cpp · ${result.audit.llamaModelPath || result.model}`;
  }
  return result.model;
}

function renderMetrics(metrics, judge) {
  const labels = {
    accuracyEstimate: "Accuracy estimate",
    survivalScore: "Survival",
    impairmentScore: "Impairment",
    divergence: "Divergence",
    coherence: "Coherence",
    hallucinationRisk: "Hallucination risk",
    memoryRisk: "Memory risk",
    reasoningRisk: "Reasoning risk"
  };
  elements.metrics.classList.remove("empty");
  elements.metrics.innerHTML = Object.entries(labels)
    .map(([key, label]) => {
      const value = metrics[key];
      const riskMetric = /risk|impairment|divergence/i.test(key);
      const tone = riskMetric
        ? value > 70 ? "danger" : value > 35 ? "warn" : "safe"
        : value < 35 ? "danger" : value < 70 ? "warn" : "safe";
      return `
        <div class="metric">
          <div class="metric-row">
            <span>${label}</span>
            <strong>${value}%</strong>
          </div>
          <div class="bar"><span class="${tone}" style="width: ${value}%"></span></div>
        </div>
      `;
    })
    .join("") + (judge ? `<pre class="judge">${escapeHtml(JSON.stringify(judge.parsed || judge.raw || judge.error, null, 2))}</pre>` : "");
}

function renderAudit(audit) {
  if (audit.error) {
    elements.auditLog.innerHTML = `<li>${escapeHtml(audit.error)}</li>`;
    return;
  }
  elements.auditLog.innerHTML = [
    `Backend: ${audit.backend}.`,
    `Messages identical: ${audit.messagesIdentical ? "yes" : "no"}.`,
    `Message fingerprint: ${audit.messageFingerprint}.`,
    `Changed runtime options: ${audit.optionDiff.map((item) => `${item.key} ${item.baseline} -> ${item.impaired}`).join(", ") || "none"}.`
  ].map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderDoseResponse(result) {
  elements.doseSummary.textContent = `${result.rows.length} trials · ${result.model}`;
  elements.doseChart.innerHTML = result.summary
    .map((row) => `
      <div class="dose-bar">
        <span>${row.doseMg} mg</span>
        <div><b style="width:${row.survivalScore}%"></b></div>
        <strong>${row.survivalScore}%</strong>
      </div>
    `)
    .join("");
  elements.doseBody.innerHTML = result.summary
    .map((row) => `
      <tr>
        <td>${row.doseMg} mg</td>
        <td>${row.trials}</td>
        <td>${row.passed}/${row.trials}</td>
        <td>${row.survivalScore}%</td>
        <td>${row.impairmentScore}%</td>
        <td>${row.hallucinationRisk}%</td>
      </tr>
    `)
    .join("");
  elements.doseTrials.innerHTML = result.rows
    .map((row) => `
      <details class="response-detail">
        <summary>
          <strong>${row.doseMg} mg · trial ${row.trial + 1}</strong>
          <span>Survival ${row.metrics.survivalScore}% · Impairment ${row.metrics.impairmentScore}% · ${row.metrics.passed ? "passed" : "failed"}</span>
        </summary>
        <pre>${escapeHtml(row.response)}</pre>
      </details>
    `)
    .join("");
}

function renderHistoryRow(item) {
  const profile = item.profile || {};
  const result = item.metrics
    ? `${item.metrics.survivalScore}% survival`
    : item.summary
      ? `${item.summary.length} doses`
      : item.steps
        ? `${item.steps.length} steps`
        : "";
  const output = getHistoryOutput(item);
  return `
    <tr>
      <td>${escapeHtml(new Date(item.timestamp).toLocaleString())}</td>
      <td>${escapeHtml(item.type)}</td>
      <td>${escapeHtml(item.model || "")}</td>
      <td>${escapeHtml(profile.drug?.name || item.drugId || "")} ${profile.doseMg ?? ""}${profile.doseMg !== undefined ? " mg" : ""}</td>
      <td>${escapeHtml(result)}</td>
      <td class="history-output">${renderOutputDetails(output)}</td>
    </tr>
  `;
}

function getHistoryOutput(item) {
  if (item.response) {
    return {
      label: "Runtime-drugged response",
      text: item.response
    };
  }
  if (item.rows?.length) {
    return {
      label: `${item.rows.length} dose trial responses`,
      text: item.rows.map((row) => `${row.doseMg} mg trial ${row.trial + 1}:\n${row.response}`).join("\n\n---\n\n")
    };
  }
  if (item.steps?.length) {
    return {
      label: `${item.steps.length} agent step outputs`,
      text: item.steps.map((step) => `Step ${step.step}:\n${step.output}`).join("\n\n---\n\n")
    };
  }
  return {
    label: "No output",
    text: ""
  };
}

function renderOutputDetails(output) {
  if (!output.text) {
    return "<span class=\"muted\">No output captured.</span>";
  }
  return `
    <details class="inline-detail">
      <summary>${escapeHtml(output.label)} · ${escapeHtml(previewText(output.text, 90))}</summary>
      <pre>${escapeHtml(output.text)}</pre>
    </details>
  `;
}

async function getJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed with ${response.status}.`);
  }
  return data;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed with ${response.status}.`);
  }
  return data;
}

function setBusy(isBusy, message = "") {
  for (const button of [elements.run, elements.doseResponse, elements.chaosAgent, elements.refreshModels]) {
    button.disabled = isBusy;
  }
  if (message) {
    setStatus(message, "pending");
  }
}

function setStatus(message, tone) {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
}

async function copyText(value, message) {
  try {
    await navigator.clipboard.writeText(value || "");
    setStatus(message, "ok");
  } catch {
    setStatus("Copy failed; select the text manually.", "error");
  }
}

function previewText(value, maxLength) {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 1)}…`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();

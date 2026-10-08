import { DOSE_MAX_MG, TECHNIQUES, TECHNIQUE_IDS, describeDose, describeTechnique, hill, parseDoseList } from "./catalog.js";

const $ = (id) => document.getElementById(id);
const state = { catalog: null, models: { gguf: [], ollama: [] }, technique: "hallucinogen", job: null, record: null, status: null };

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (v, d = 2) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? "–" : Number(v).toFixed(d));
const ci = (x, d = 2) => (!x || x.mean === null ? "–" : x.lo === null ? fmt(x.mean, d) : `${fmt(x.mean, d)} [${fmt(x.lo, d)}, ${fmt(x.hi, d)}]`);

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function message(text, tone = "") {
  $("message").textContent = text;
  $("message").className = `message ${tone}`;
}

// ---------------------------------------------------------------- setup panel

async function loadStatus() {
  try {
    state.status = await api("/api/status");
    const llama = state.status.llamaServer.available;
    $("status").innerHTML = `<span class="pill ${llama ? "ok" : "bad"}">llama-server ${llama ? "ready" : "missing"}</span>` +
      `<span class="pill ${state.models.ollamaError ? "bad" : "ok"}">Ollama ${state.models.ollamaError ? "offline" : "online"}</span>`;
  } catch (error) {
    $("status").innerHTML = `<span class="pill bad">API offline</span>`;
  }
}

async function loadModels() {
  try {
    state.models = await api("/api/models");
  } catch (error) {
    message(`Could not list models: ${error.message}`, "error");
  }
  renderModels();
  const judge = $("judge");
  const current = judge.value;
  judge.innerHTML = `<option value="">None</option>` + state.models.ollama.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("");
  judge.value = current;
  loadStatus();
}

function renderModels() {
  const backend = $("backend").value;
  const list = backend === "llamacpp" ? state.models.gguf : state.models.ollama;
  const previous = $("model").value;
  $("model").innerHTML = list.length
    ? list.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}${m.size ? ` · ${(m.size / 1e9).toFixed(1)} GB` : ""}</option>`).join("")
    : `<option value="">(no models found)</option>`;
  if (list.some((m) => m.id === previous)) $("model").value = previous;
  $("backendHint").textContent =
    backend === "llamacpp"
      ? "Patched llama-server. Every technique acts inside the forward pass or on the logits. GGUF files come from ./models and from Ollama's blob store."
      : state.models.ollamaError
        ? `Ollama is not reachable: ${state.models.ollamaError}`
        : "Ollama exposes only sampling options. Techniques are approximated there, and some are unavailable.";
  renderTechniques();
}

function renderTechniques() {
  const backend = $("backend").value;
  $("techniques").innerHTML = TECHNIQUE_IDS.map((id) => {
    const t = TECHNIQUES[id];
    const support = backend === "llamacpp" ? "full" : t.ollamaSupport;
    return `<button type="button" role="radio" aria-checked="${id === state.technique}" class="technique ${id === state.technique ? "selected" : ""} support-${support}" data-id="${id}" ${support === "none" ? "disabled" : ""}>
      <strong>${esc(t.name)}</strong><span>${esc(t.category)}</span>${support !== "full" ? `<em>${support === "none" ? "llama.cpp only" : "approx."}</em>` : ""}</button>`;
  }).join("");
  if (backend === "ollama" && TECHNIQUES[state.technique].ollamaSupport === "none") selectTechnique("hallucinogen");
  else renderTechniqueInfo();
}

function selectTechnique(id) {
  state.technique = id;
  const t = TECHNIQUES[id];
  $("theme").value = (t.theme || []).join(", ");
  $("themeWrap").classList.toggle("hidden", !t.theme);
  renderTechniques();
}

function renderTechniqueInfo() {
  const d = describeTechnique(state.technique);
  const backend = $("backend").value;
  const support = backend === "llamacpp" ? "full" : d.support.ollama;
  const sites = d.sites.length ? d.sites.map((s) => `<span class="tag">${esc(s)}</span>`).join("") : `<span class="tag">none</span>`;
  $("techniqueInfo").innerHTML = `
    <p><strong>${esc(d.name)}</strong> — ${esc(d.summary)}</p>
    <p class="hint">Analogy: ${esc(d.analogy)}</p>
    <ul>${d.mechanism.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>
    <p class="hint">Sites: ${sites} · Hill EC50 ${d.curve.ec50} mg, n = ${d.curve.n} · E = effect intensity (0–1)</p>
    ${support === "approximate" ? `<p class="warn">On Ollama this technique is approximated with sampler options only (temperature, top-p/top-k, mirostat, context size). It is not the same mechanism.</p>` : ""}`;
  renderDose();
}

function renderDose() {
  const dose = Number($("dose").value);
  const curve = TECHNIQUES[state.technique].curve;
  const e = hill(dose, curve);
  $("doseLabel").textContent = `${dose} mg → intensity E = ${e.toFixed(3)} (${describeDose(state.technique, dose)})`;
  const W = 320, H = 120, P = 18;
  const x = (d) => P + (d / DOSE_MAX_MG) * (W - 2 * P);
  const y = (v) => H - P - v * (H - 2 * P);
  let path = "";
  for (let d = 0; d <= DOSE_MAX_MG; d += 5) path += `${d ? "L" : "M"}${x(d).toFixed(1)},${y(hill(d, curve)).toFixed(1)}`;
  $("curve").innerHTML = `
    <line x1="${P}" y1="${y(0)}" x2="${W - P}" y2="${y(0)}" class="axis"/>
    <line x1="${P}" y1="${y(0)}" x2="${P}" y2="${y(1)}" class="axis"/>
    <line x1="${x(curve.ec50)}" y1="${y(0)}" x2="${x(curve.ec50)}" y2="${y(0.5)}" class="guide"/>
    <path d="${path}" class="line"/>
    <circle cx="${x(dose)}" cy="${y(e)}" r="4" class="dot"/>
    <text x="${P}" y="${H - 3}" class="label">0</text><text x="${W - P - 24}" y="${H - 3}" class="label">500 mg</text>
    <text x="${x(curve.ec50) + 3}" y="${y(0.5) - 3}" class="label">EC50</text><text x="2" y="${y(1) + 4}" class="label">1</text>`;
}

function setDose(value) {
  const v = Math.min(DOSE_MAX_MG, Math.max(0, Number(value) || 0));
  $("dose").value = v;
  $("doseNumber").value = v;
  renderDose();
}

function currentInput() {
  const num = (id) => Number($(id).value);
  return {
    backend: $("backend").value,
    modelId: $("model").value,
    techniqueId: state.technique,
    doseMg: num("dose"),
    prompt: $("prompt").value,
    memory: $("memory").value,
    expected: $("expected").value,
    system: $("system").value,
    theme: $("theme").value,
    seed: num("seed"),
    noiseFloor: $("noiseFloor").checked,
    judgeModelId: $("judge").value,
    sampling: Object.fromEntries(["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "max_tokens", "num_ctx"].map((k) => [k, num(k)])),
    doses: parseDoseList($("doses").value),
    trials: num("trials"),
    reseedInjection: $("reseed").checked,
    steps: num("steps")
  };
}

// ---------------------------------------------------------------- jobs

async function startJob(kind) {
  if (state.job) return;
  const input = currentInput();
  if (!input.modelId) return message("Select a model first.", "error");
  try {
    const job = await api(`/api/${kind}`, input);
    state.job = job.id;
    setBusy(true);
    message("");
    pollJob();
  } catch (error) {
    message(error.message, "error");
  }
}

async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api(`/api/jobs/${state.job}`);
    const p = job.progress || { done: 0, total: 1 };
    $("progressFill").style.width = `${Math.round((100 * p.done) / Math.max(1, p.total))}%`;
    $("progressText").textContent = `${job.status}: ${p.message || ""} (${p.done}/${p.total})`;
    if (job.status === "queued" || job.status === "running") return void setTimeout(pollJob, 700);
    state.job = null;
    setBusy(false);
    if (job.status === "done") {
      showRecord(job.result);
      loadHistory();
      message("Done.", "ok");
    } else {
      message(job.status === "cancelled" ? "Cancelled." : `Error: ${job.error}`, job.status === "cancelled" ? "" : "error");
    }
  } catch (error) {
    state.job = null;
    setBusy(false);
    message(error.message, "error");
  }
}

function setBusy(busy) {
  $("progress").classList.toggle("hidden", !busy);
  for (const id of ["runTrial", "runSweep", "runAgent"]) $(id).disabled = busy;
}

// ---------------------------------------------------------------- results

function metricsTable(m) {
  if (!m) return "";
  const rows = [
    ["Impairment (heuristic, 0–100)", fmt(m.impairment, 1), ""],
    ["Divergence from baseline", fmt(m.divergence), m.noiseFloor === null ? "" : `noise floor ${fmt(m.noiseFloor)} → excess ${fmt(m.excessDivergence)}`],
    ["Garble rate", fmt(m.treated.garble), `baseline ${fmt(m.baseline.garble)}`],
    ["Repetition (repeated 3-grams)", fmt(m.treated.repetition), `baseline ${fmt(m.baseline.repetition)}`],
    ["Script switches / 100 letters", fmt(m.treated.scriptSwitch), `baseline ${fmt(m.baseline.scriptSwitch)}`],
    ["Length (words)", m.treated.words, `baseline ${m.baseline.words}`]
  ];
  if (m.anchor) rows.push(["Expected anchor present", m.anchor.treated ? "yes" : "no", `baseline ${m.anchor.baseline ? "yes" : "no"}`]);
  const it = m.internal?.treated;
  const ib = m.internal?.baseline;
  if (it) {
    rows.push(["Top-5 entropy (nats)", fmt(it.entropy), ib ? `baseline ${fmt(ib.entropy)}` : ""]);
    rows.push(["Surprisal of chosen tokens", fmt(it.surprisal), ib ? `baseline ${fmt(ib.surprisal)}` : ""]);
  }
  return `<table class="metrics"><tbody>${rows.map((r) => `<tr><th>${esc(r[0])}</th><td>${esc(r[1])}</td><td class="hint">${esc(r[2])}</td></tr>`).join("")}</tbody></table>`;
}

function auditBlock(treatment, engine) {
  const parts = [];
  if (treatment?.kind === "llamacpp") {
    parts.push(`<pre>${esc(Object.entries(treatment.env || {}).map(([k, v]) => `${k}=${v}`).join("\n") || "(no LLM_INJ_* variables)")}</pre>`);
    if (engine) {
      parts.push(`<p>Engine: <code>${esc(engine.active || "inactive (no perturbation)")}</code></p>`);
      parts.push(`<p>Sites fired: ${engine.sitesFired?.length ? engine.sitesFired.map((s) => `<span class="tag">${esc(s)}</span>`).join("") : "none"}</p>`);
      if (engine.warnings?.length) parts.push(`<p class="warn">${esc(engine.warnings.join("; "))}</p>`);
    }
  } else if (treatment?.kind === "ollama") {
    parts.push(treatment.changes?.length
      ? `<table class="metrics"><tbody>${treatment.changes.map((c) => `<tr><th>${esc(c.key)}</th><td>${esc(c.baseline)} → ${esc(c.treated)}</td></tr>`).join("")}</tbody></table>`
      : "<p>No option changes (placebo or zero dose).</p>");
  }
  return `<details class="audit"><summary>Treatment audit (intensity ${fmt(treatment?.intensity, 3)})</summary>${parts.join("")}</details>`;
}

function judgeBlock(judge) {
  if (!judge) return "";
  if (judge.error) return `<p class="warn">Judge (${esc(judge.model)}) failed: ${esc(judge.error)}</p>`;
  const row = (name, s) => `<tr><th>${name}</th><td>${s?.coherence ?? "–"}</td><td>${s?.on_task ?? "–"}</td><td>${s?.factual ?? "–"}</td></tr>`;
  return `<h3>Blind judge: ${esc(judge.model)}</h3><table class="metrics"><thead><tr><th></th><th>Coherence</th><th>On task</th><th>Factual</th></tr></thead>
    <tbody>${row("Baseline", judge.baseline)}${row("Treated", judge.treated)}</tbody></table><p class="hint">${esc(judge.notes)}</p>`;
}

function arms(cols) {
  return `<div class="arms">${cols.map(([title, text, note]) => `<article class="arm"><header><h3>${esc(title)}</h3><span class="hint">${esc(note || "")}</span></header><pre class="output">${esc(text)}</pre></article>`).join("")}</div>`;
}

function header(record) {
  const t = TECHNIQUES[record.techniqueId];
  const dose = record.doseMg !== undefined && record.doseMg !== null ? ` · ${record.doseMg} mg` : "";
  return `<p class="record-head"><strong>${esc(t ? t.name : record.techniqueId)}</strong>${esc(dose)} · ${esc(record.backend)} · ${esc(record.model)} · <span class="hint">${esc(new Date(record.timestamp).toLocaleString())}</span>${record.legacy ? ` <span class="tag warn">legacy record</span>` : ""}</p>`;
}

function renderRun(r) {
  const cols = [["Baseline", r.arms.baseline.content, `seed ${r.arms.baseline.seed}`]];
  if (r.arms.noise) cols.push(["Noise floor (untreated, other seed)", r.arms.noise.content, `seed ${r.arms.noise.seed}`]);
  cols.push(["Treated", r.arms.treated.content, `seed ${r.arms.treated.seed} · ${r.arms.treated.ms} ms`]);
  return header(r) + arms(cols) + metricsTable(r.metrics) + judgeBlock(r.judge) + auditBlock(r.treatment, r.arms.treated.engine);
}

function doseChart(summary) {
  const W = 560, H = 220, P = 36;
  const x = (d) => P + (d / DOSE_MAX_MG) * (W - 2 * P);
  const y = (v) => H - P - Math.max(0, Math.min(1, v)) * (H - 2 * P);
  const series = [
    ["impairment", (s) => s.impairment, 100, "s1"],
    ["excess divergence", (s) => s.excessDivergence, 1, "s2"]
  ];
  let svg = `<line x1="${P}" y1="${y(0)}" x2="${W - P}" y2="${y(0)}" class="axis"/><line x1="${P}" y1="${y(0)}" x2="${P}" y2="${y(1)}" class="axis"/>`;
  for (const d of [0, 100, 200, 300, 400, 500]) svg += `<text x="${x(d) - 8}" y="${H - P + 14}" class="label">${d}</text>`;
  svg += `<text x="${W - P - 20}" y="${H - 4}" class="label">mg</text><text x="4" y="${y(1) + 4}" class="label">max</text>`;
  for (const [name, get, scale, cls] of series) {
    const pts = summary.map((s) => ({ d: s.doseMg, v: get(s) })).filter((p) => p.v && p.v.mean !== null);
    svg += `<path class="line ${cls}" d="${pts.map((p, i) => `${i ? "L" : "M"}${x(p.d).toFixed(1)},${y(p.v.mean / scale).toFixed(1)}`).join("")}"/>`;
    for (const p of pts) {
      if (p.v.lo !== null) svg += `<line class="err ${cls}" x1="${x(p.d)}" x2="${x(p.d)}" y1="${y(p.v.lo / scale)}" y2="${y(p.v.hi / scale)}"/>`;
      svg += `<circle class="dot ${cls}" cx="${x(p.d)}" cy="${y(p.v.mean / scale)}" r="3.5"/>`;
    }
  }
  const legend = series.map(([name, , scale, cls]) => `<span class="legend ${cls}">${esc(name)}${scale === 100 ? " (/100)" : ""}</span>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Dose-response chart">${svg}</svg><div>${legend} <span class="hint">error bars: bootstrap 95% CI over trials</span></div>`;
}

function renderSweep(r) {
  const table = `<table class="metrics"><thead><tr><th>Dose</th><th>n</th><th>Impairment</th><th>Excess divergence</th><th>Garble</th><th>Repetition</th><th>Anchor kept</th><th>Entropy</th></tr></thead><tbody>
    ${r.summary.map((s) => `<tr><th>${s.doseMg} mg</th><td>${s.n}</td><td>${ci(s.impairment, 1)}</td><td>${ci(s.excessDivergence)}</td><td>${ci(s.garble)}</td><td>${ci(s.repetition)}</td><td>${ci(s.anchorTreated)}</td><td>${ci(s.entropy)}</td></tr>`).join("")}
  </tbody></table>`;
  const samples = r.rows.filter((row) => row.trial === 0).map((row) => [`${row.doseMg} mg (E = ${fmt(row.intensity, 2)})`, row.treated.content, `trial 1 · impairment ${fmt(row.metrics.impairment, 1)}`]);
  return header(r) + doseChart(r.summary) + table +
    `<details open><summary>Trial 1 outputs</summary>${arms([["Baseline", r.baselines[0].baseline.content, `seed ${r.baselines[0].seed}`], ...samples])}</details>` +
    auditBlock(r.rows[r.rows.length - 1]?.treatment, r.rows[r.rows.length - 1]?.treated.engine);
}

function renderAgent(r) {
  return header(r) + r.steps.map((s) => `<h3>Step ${s.step} <span class="hint">divergence ${fmt(s.metrics.divergence)} · impairment ${fmt(s.metrics.impairment, 1)}</span></h3>` +
    arms([["Baseline trajectory", s.baseline.content], ["Treated trajectory", s.treated.content]])).join("") +
    auditBlock(r.treatment, r.steps[r.steps.length - 1]?.treated.engine);
}

function renderLegacy(r) {
  const text = typeof r.response === "string" ? r.response : JSON.stringify(r.response ?? r.rows ?? r.steps, null, 2);
  return header(r) + `<p class="warn">Recorded by the pre-1.0 prototype. Its metrics used a different, partly dose-derived formula and are not comparable.</p>` +
    arms([["Baseline", r.baseline?.answer ?? ""], ["Response", text]]);
}

function showRecord(record) {
  state.record = record;
  const view = record.legacy ? renderLegacy : record.type === "run" ? renderRun : record.type === "dose-response" ? renderSweep : renderAgent;
  $("results").innerHTML = view(record);
  $("exportJson").disabled = false;
  $("exportCsv").disabled = record.legacy;
}

// ---------------------------------------------------------------- history & export

async function loadHistory() {
  try {
    const { items } = await api("/api/history?limit=60");
    $("history").innerHTML = items.length
      ? items.map((item, i) => `<button type="button" class="history-item" data-index="${i}">
          <span>${esc(item.type)}</span><strong>${esc(TECHNIQUES[item.techniqueId]?.name || item.techniqueId || "–")}</strong>
          <span>${item.doseMg !== undefined && item.doseMg !== null ? `${item.doseMg} mg` : item.input?.doses ? `${item.input.doses.join("/")} mg` : ""}</span>
          <span class="hint">${esc(item.model)}</span><span class="hint">${esc(new Date(item.timestamp).toLocaleString())}</span>${item.legacy ? `<span class="tag warn">legacy</span>` : ""}</button>`).join("")
      : `<p class="empty">No runs yet.</p>`;
    state.history = items;
  } catch (error) {
    $("history").innerHTML = `<p class="warn">${esc(error.message)}</p>`;
  }
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function recordToCsv(r) {
  const head = ["id", "type", "technique", "backend", "model", "dose_mg", "trial", "seed", "intensity", "impairment", "divergence", "noise_floor", "excess_divergence", "garble", "repetition", "script_switch", "anchor", "entropy", "surprisal", "words"];
  const line = (dose, trial, seed, intensity, m) => [r.id, r.type, r.techniqueId, r.backend, JSON.stringify(r.model), dose, trial, seed, intensity, m.impairment, m.divergence, m.noiseFloor ?? "", m.excessDivergence,
    m.treated.garble, m.treated.repetition, m.treated.scriptSwitch, m.anchor ? m.anchor.treated : "", m.internal?.treated?.entropy ?? "", m.internal?.treated?.surprisal ?? "", m.treated.words].join(",");
  const rows = r.type === "run" ? [line(r.doseMg, 0, r.arms.treated.seed, r.intensity, r.metrics)]
    : r.type === "dose-response" ? r.rows.map((row) => line(row.doseMg, row.trial, row.seed, row.intensity, row.metrics))
    : r.steps.map((s) => line(r.doseMg, s.step, s.treated.seed, r.intensity, s.metrics));
  return `${head.join(",")}\n${rows.join("\n")}\n`;
}

// ---------------------------------------------------------------- wiring

function bind() {
  $("backend").addEventListener("change", renderModels);
  $("refreshModels").addEventListener("click", loadModels);
  $("techniques").addEventListener("click", (event) => {
    const button = event.target.closest(".technique");
    if (button && !button.disabled) selectTechnique(button.dataset.id);
  });
  $("dose").addEventListener("input", () => setDose($("dose").value));
  $("doseNumber").addEventListener("input", () => setDose($("doseNumber").value));
  $("runTrial").addEventListener("click", () => startJob("run"));
  $("runSweep").addEventListener("click", () => startJob("dose-response"));
  $("runAgent").addEventListener("click", () => startJob("agent"));
  $("cancel").addEventListener("click", () => state.job && api(`/api/jobs/${state.job}/cancel`, {}).catch(() => {}));
  $("refreshHistory").addEventListener("click", loadHistory);
  $("history").addEventListener("click", async (event) => {
    const item = event.target.closest(".history-item");
    if (item) showRecord(state.history[Number(item.dataset.index)]);
  });
  $("exportJson").addEventListener("click", () => state.record && download(`${state.record.id}.json`, JSON.stringify(state.record, null, 2), "application/json"));
  $("exportCsv").addEventListener("click", () => state.record && download(`${state.record.id}.csv`, recordToCsv(state.record), "text/csv"));
}

async function init() {
  bind();
  try {
    state.catalog = await api("/api/catalog");
    $("system").value = state.catalog.defaults.system;
  } catch {
    // static preview without the API
  }
  selectTechnique(state.technique);
  setDose(150);
  await loadModels();
  await loadHistory();
}

init();

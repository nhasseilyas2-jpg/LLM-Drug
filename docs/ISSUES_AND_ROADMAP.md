# Issues found and roadmap

This document records the audit of the pre-1.0 prototype ("LLM Drugs"), what was fixed for 1.0,
the issues that remain open, and the research roadmap.

## Audit of the prototype and fixes

| # | Severity | Issue in the prototype | Evidence | Fix in 1.0 |
|---|---|---|---|---|
| 1 | Critical | **The llama.cpp hook never ran.** The patch hooked `llama_sampler_sample()`, which `llama-cli` and `llama-server` do not call (they use the common sampler chain). | Greedy output byte-identical at 0 mg and 500 mg. | New engine hooks the sampler chain itself plus 5 graph/KV sites; e2e tests assert every technique changes greedy output at 500 mg and placebo does not. |
| 2 | Critical | **Only logits could be touched.** The "drugs" were sampling tweaks; nothing reached attention, hidden states or memory. | Code review. | Six injection sites: logits, attention scale, residual noise, layer gain, FFN dropout, KV forgetting ([engine README](../llamacpp-injection/README.md)). |
| 3 | Critical | **Remote code execution / file read.** The browser sent the llama binary path and model path; the server spawned whatever it was given, and it listened on all interfaces. | Code review. | Server binds 127.0.0.1 only, checks `Host` (DNS rebinding) and `Origin`, requires `application/json`, body limit 256 KB. Clients send opaque model ids; paths come only from server env. Tests cover each case. |
| 4 | High | **Circular metrics.** Part of the "impairment" score was computed from the dose and profile, so higher doses scored worse by construction. | Code review. | All metrics are output-only ([METHODOLOGY](METHODOLOGY.md#metrics)); a test asserts identical arms score 0. |
| 5 | High | **No controls.** No placebo, no matched seeds, no measure of normal sampling variability. | Code review. | Baseline, placebo, and a noise-floor arm with a different seed; excess divergence subtracts the noise floor. |
| 6 | High | **`llama-cli` hang.** Newer `llama-cli` enters interactive mode; the Node spawn waited forever. | Reproduced. | Uses `llama-server` (HTTP, one process per configuration, LRU pool, health checks, timeouts). |
| 7 | High | **Stale build directory.** The CMake cache pointed at a different source tree, so rebuilding did not pick up the patch. | `CMakeCache.txt`. | Setup script pins the llama.cpp commit, applies a `git apply` patch, verifies it with a reverse check, rebuilds in place. |
| 8 | Medium | Engine parameters were silently clamped and there was no proof a hook ran. | — | Strict parsing with warnings; stderr audit lines (`ACTIVE`, `site fired`) stored with each run. |
| 9 | Medium | Single prompt, single run: no repeated trials or uncertainty. | — | Dose-response sweeps with trials, shared baselines, bootstrap CIs, CSV/JSON export, headless runner. |
| 10 | Medium | Ollama "techniques" presented as equivalent to internal ones. | 30B model unaffected at 400 mg. | Per-technique support level (full / approximate / none); unsupported techniques are rejected with 400. |
| 11 | Medium | UI used inline scripts/styles, no CSP; history was an unbounded JSON array rewritten on every run. | — | Strict CSP, no inline code; append-only JSONL with rotation, corrupt-line tolerance and legacy-record mapping. |
| 12 | Low | No tests for the actual behaviour (tests only checked string formatting). | — | 132 C++ engine checks, 49 Node unit/integration tests, 24 e2e tests against the real engine, 5 Python client tests. |
| 13 | Low | Mis-encoded text (double UTF-8) in technique descriptions. | `Ã‚Â·` in the UI. | Fixed; a test now rejects encoding damage. |

## Known open issues

Status: ✅ resolved · 🟡 partly resolved · ⏳ open (with the reason).

| Issue | Status |
|---|---|
| **Graph-site noise is activation-derived** (`sin(K·x + φ)`). | ✅ `LLM_INJ_NOISE_MODE=hash` uses a counter-hash ggml custom op: an independent draw per (seed, layer, channel). The default stays `sin`, because custom ops run on the CPU backend. |
| **One process per configuration** (1–3 s restart per dose). | ✅ Shared mode: one `llama-server` per model re-reads `LLM_INJ_CONFIG_FILE` at each request (`config gen=N`). Used by default; `LLAMA_SHARED=0` restores the old behaviour. |
| **Calibration on one small model.** | 🟡 Benchmark curves for qwen2.5-0.5b (13 techniques) and a reduced run on qwen3-coder:30b (MoE) are in [METHODOLOGY](METHODOLOGY.md#benchmark-robustness-curves). Dense 7B–70B models and per-model EC50 fitting are still open: no such GGUF was available here, and CPU-only runs of large dense models are slow. |
| **Logprobs exclude logit-site perturbations** (llama-server reports pre-sampler logprobs). | 🟡 Clean-model surprisal now scores treated text with the untreated model, so it covers logit-site techniques too. The raw logprob caveat remains, since changing it would mean changing llama-server's reporting. |
| **About 31 architectures do not call `build_cvec`**, so residual and steering sites never fire there. | ⏳ The audit line ("site fired") makes this visible. A generic hook needs a patch in every model builder, which is a large maintenance surface for a pinned patch. Attention, head, FFN, KV and logit sites work everywhere. |
| **No KV forgetting for recurrent/hybrid models.** | ⏳ Mamba/RWKV have no KV mask to edit. A state-decay analogue would be needed. There is no such model to test with here. |
| **CPU-first.** GPU builds compile but are not benchmarked. | 🟡 The release workflow adds a Vulkan compile check and a Metal build on macOS. GPU runtime benchmarks need GPU runners or hardware, which were not available. Hash-noise mode is CPU-bound by design. |
| **Impairment weights are ad hoc.** | 🟡 Still a coarse summary, but analysis no longer depends on it: the benchmark measures task accuracy directly, and clean surprisal gives a model-internal measure. |
| **Patch maintenance** (pinned to one llama.cpp commit). | ⏳ Inherent. CI applies the patch to the pinned commit on every push; rebasing is manual. |
| Ollama sampler mapping is approximate. | ⏳ By design: the Ollama API exposes no logits or activations. Documented as "sampling stress". |

## Roadmap

Ordered roughly by research value.

| # | Item | Status |
|---|---|---|
| 1 | **Clean-model surprisal**: perplexity of the treated output under the untreated model. | ✅ Grammar-forced teacher forcing against the clean server, for every arm and in sweeps and agent loops. In the CSV and Python exports. |
| 2 | **Activation steering techniques** (receptor-specific "drugs"). | ✅ `steer` engine site (GGUF control vectors, raw-vector scaling), `npm run vectors` (persona-pair prompt sets → `llama-cvector-generator`), `euphoria` / `dysphoria` techniques, and steering-aware combinations. More vectors only need a new `steering/<name>.json`. |
| 3 | **Head- and neuron-level targeting** (lesions). | ✅ `HEAD_LESION` / `HEAD_IDS` / `HEAD_GAIN` / `HEAD_LAYERS` and `FFN_LESION`. Techniques `neurotoxin` (diffuse) and `stroke` (focal band). |
| 4 | **Per-request injection configuration**, and dose changes inside one generation. | ✅ Shared mode (reloadable config file) and the in-generation PK time course. |
| 5 | **Pharmacokinetics**: onset/half-life over tokens, tolerance across agent steps, combinations. | ✅ `PK_ONSET` / `PK_HALFLIFE`, agent `tolerance` (dose·(1−tol)^step), and co-administration with per-knob merge rules. All in the UI and API. |
| 6 | **Benchmarks under perturbation.** | 🟡 `npm run bench`: an exact-answer set with Wilson CIs and D50 per technique, JSON+Markdown output. Standard sets (GSM8K, MMLU) and tool-use tasks are not bundled, because of licensing and size. The task file format (`{id, q, a[]}`) accepts them directly via `--tasks`. |
| 7 | **Recalibration on larger models** and matched-impairment comparison. | 🟡 See "Calibration on one small model" above. D50 per technique is the matched-impairment measure; comparing models means comparing their D50 tables. |
| 8 | **True RNG for graph sites** and a generic residual hook. | 🟡 True RNG is done (hash mode). The generic residual hook is open (see above). |
| 9 | **GPU CI** and prebuilt binaries for Windows/Linux/macOS. | 🟡 `.github/workflows/release.yml` builds and packages `llama-server` + `llama-cvector-generator` for all three OSes on `v*` tags, plus a Vulkan compile job. It has not been executed yet: it needs a push to GitHub, and it cannot run locally. GPU *runtime* CI needs self-hosted GPU runners. |
| 10 | **Notebook / Python client.** | ✅ `tools/python` (stdlib-only, pandas optional): `load_history`, `flatten` / `to_dataframe` with the same columns as the CSV export, and `LabClient` for the HTTP API (submit, wait, cancel). Tested in CI. |

### Next ideas

- More steering vectors (anxiety/calm, confidence/doubt, honesty) and a UI to choose a vector.
- Sparse-autoencoder feature clamping as a more selective "receptor" site.
- Withdrawal and rebound: an opposite steering pulse after the half-life.
- Per-token traces of the PK factor and fired sites, visualized along the response.

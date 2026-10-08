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
| 12 | Low | No tests for the actual behaviour (tests only checked string formatting). | — | 60 C++ unit tests, 38 Node unit/integration tests, 10 e2e tests against the real engine. |
| 13 | Low | Mis-encoded text (double UTF-8) in technique descriptions. | `Ã‚Â·` in the UI. | Fixed; a test now rejects encoding damage. |

## Known open issues

- **Graph-site noise is activation-derived.** Residual noise and FFN dropout use `sin(K·x + φ)`
  instead of true RNG (ggml has no per-element RNG op). It is reproducible and well spread, but
  correlated with the input. A custom ggml op or a precomputed noise tensor would remove this.
- **Logprobs exclude logit-site perturbations** (llama-server reports pre-sampler logprobs).
- **About 31 architectures do not call `build_cvec`**, so residual sites never fire there. The audit
  line makes this visible, but a generic hook would be better.
- **No KV forgetting for recurrent/hybrid models.**
- **CPU-first.** GPU builds (`-Cuda`, `-Vulkan`, `--metal`) compile but were not benchmarked here.
- **One process per configuration.** Each new dose starts a new `llama-server` (about 1–3 s for
  small models, longer for large ones). A per-request configuration API inside llama-server would
  remove this cost.
- **Calibration on one small model.** Constants and EC50 values were tuned on qwen2.5-0.5b-instruct.
- **Impairment weights are ad hoc.** Use individual metrics and the judge for analysis.
- **Patch maintenance.** The patch is pinned to one llama.cpp commit and needs rebasing for newer versions.

## Roadmap

Ordered roughly by research value.

1. **Clean-model surprisal.** Score the treated text with the *untreated* model
   (perplexity of the treated output under the clean model). This is a model-internal, dose-blind
   measure of how "unlike itself" the output is, and works for logit-site techniques too.
2. **Activation steering techniques.** Load control vectors (llama.cpp already supports
   `--control-vector`) and expose them as dose-scaled techniques, e.g. mood or persona shifts. This
   is the closest analogue to receptor-specific drugs.
3. **Head- and neuron-level targeting.** Per-head attention ablation/scaling and targeted FFN
   neuron dropout, for localized "lesions" and interpretability experiments.
4. **Per-request injection configuration** in llama-server, removing the process-per-dose cost and
   enabling dose changes inside one generation ("onset", "wear-off", "half-life" curves).
5. **Pharmacokinetics.** Dose that changes over generated tokens (absorption/elimination curves),
   tolerance across agent steps, and combinations of two techniques (interaction effects).
6. **Benchmarks under perturbation.** Run standard task sets (GSM8K, MMLU subsets, tool-use tasks)
   at several doses and report robustness curves per model.
7. **Recalibration on larger models** (7B–70B), plus matched-impairment comparisons across models.
8. **True RNG for graph sites** (custom ggml op) and a generic residual hook for all architectures.
9. **GPU CI** and prebuilt patched binaries for Windows/Linux/macOS.
10. **Notebook / Python client** for analysis of exported JSONL/CSV.

# LLM Injection Runtime Lab

**Dose-controlled runtime perturbation of local language models, with matched controls and metrics.**

The lab "injects" a technique into a model while it generates text, at a dose from 0 to 500 mg,
and compares the result with an untreated run of the same prompt. Techniques are named after drug
classes (hallucinogen, depressant, amnesic, stimulant, dissociative, delirium, delusion,
paranoia, ...) because each one disturbs a different part of the model's computation, in the way
those drug classes disturb different parts of cognition:

| Technique | What actually changes inside the model |
|---|---|
| Hallucinogen | noise in the residual stream + improbable tokens boosted |
| Depressant | flattened attention + damped late layers |
| Amnesic | random earlier tokens hidden from attention (KV memory) |
| Stimulant | sharpened attention + amplified layers + over-confident sampling |
| Dissociative | a band of middle layers progressively switched off |
| Delirium | feed-forward dropout + vetoed top token |
| Delusion / Paranoia | persistent bias toward a theme vocabulary |
| Creativity | flatter distribution + light late-layer noise |
| Placebo | nothing (control) |

The prompt is never changed and the output is never edited. Model weights are never modified.
The analogy is a mnemonic only; every run records the exact low-level parameters that were applied.

> Built for robustness, interpretability and AI-safety research: how gracefully does a model
> degrade, which internal component matters for which behaviour, and how do errors compound in
> agent loops?

## How it works

```
             same messages, same sampling seed
 ┌──────────┐   ┌───────────────────────┐   ┌──────────────┐
 │ Baseline │   │ Noise floor (seed+Δ)  │   │   Treated    │  ← LLM_INJ_* env from technique × dose
 └────┬─────┘   └──────────┬────────────┘   └──────┬───────┘
      └─────── untreated llama-server ──┘   patched llama-server (logits, attention,
                                             residual stream, FFN, KV memory)
                         ↓ metrics (output-only) · blind judge · history
```

- **Patched llama.cpp backend (full).** A small engine compiled into llama.cpp perturbs
  [six injection sites](llamacpp-injection/README.md). Dose maps to intensity with a Hill curve.
- **Ollama backend (approximate).** Only sampling options can be changed (temperature, top-p,
  top-k, mirostat, context length). Useful as "sampling stress"; it cannot reach model internals.
- **Controls.** Baseline, placebo, and a noise-floor arm measure ordinary sampling variability.
- **Metrics.** Divergence beyond the noise floor, garble, repetition, script switching, expected-
  answer hit, logprob entropy/confidence, a composite impairment score and an optional blind
  LLM judge. Nothing is derived from the dose. See [METHODOLOGY](docs/METHODOLOGY.md).
- **Modes.** Single trial, dose-response sweep with bootstrap CIs, multi-step agent loop.
- **Reproducible.** Seeds everywhere, full parameter audit, JSONL history, CSV/JSON export,
  headless experiment runner.

## Quick start

Requirements: Node.js ≥ 20. For the full backend: Git, CMake ≥ 3.21 and a C++17 compiler
(Visual Studio 2022 Build Tools on Windows). Optional: [Ollama](https://ollama.com).

```powershell
# 1. Build the patched llama-server (clones llama.cpp at the pinned commit, applies the patch)
powershell -ExecutionPolicy Bypass -File scripts\setup-llamacpp-injection.ps1 -Test
#    Linux/macOS: bash scripts/setup-llamacpp-injection.sh --test   (add --cuda / --vulkan / --metal)

# 2. Put one or more .gguf models in models\  (models already pulled with Ollama are found automatically)

# 3. Start the lab
npm start        # → http://localhost:4173
```

A good first model is a small instruct model such as `qwen2.5-0.5b-instruct-q4_k_m.gguf`: it is
fast on CPU and the techniques were calibrated on it.

No dependencies are installed: the lab uses only Node built-ins.

## Using the lab

1. Pick a backend and model, a technique and a dose. The curve shows dose → intensity.
2. Enter a prompt. Optionally add *context* (needed for the amnesic technique), an *expected
   answer* (substring, `a|b` for alternatives) and a judge model.
3. **Run trial** shows baseline, noise floor and treated output side by side, with metrics, the
   applied engine parameters and the sites that actually fired.
4. **Dose sweep** runs several doses × trials and plots impairment with 95 % CIs.
5. **Agent loop** runs a multi-step objective on both arms and scores each step.

Every result is saved to `data/runs.jsonl` and can be exported as CSV or JSON.

### Headless experiments

```powershell
npm run experiment -- experiments\calibration-llamacpp.json
npm run experiment -- experiments\ollama-sampler.json --model qwen3-coder
```

Results are written to `data/experiments/` as JSON and CSV. The spec format is in
`experiments/*.json` (backend, model, techniques, doses, trials, sampling, seed, prompts).

### Using the engine without the lab

```powershell
npm run inject-env -- delirium 300 --format ps     # prints $env:LLM_INJ_... lines
vendor\llama.cpp\build\bin\Release\llama-server.exe -m models\model.gguf
```

## Configuration

| Variable | Default | |
|---|---|---|
| `PORT` | 4173 | UI/API port (always bound to 127.0.0.1) |
| `LLAMA_SERVER_BIN` | `vendor/llama.cpp/build/bin/[Release/]llama-server` | patched binary |
| `LLAMA_MODELS_DIR` | `models` | GGUF search path (`;`/`:` separated) |
| `OLLAMA_URL` / `OLLAMA_MODELS` | `http://127.0.0.1:11434` / `~/.ollama/models` | Ollama API and blob store |
| `LLAMA_THREADS`, `LLAMA_CTX`, `LLAMA_GPU_LAYERS` | auto, 4096, — | llama-server settings |
| `LLAMA_MAX_SERVERS` | 2 | llama-server processes kept warm |
| `LAB_DATA_DIR` | `data` | history and experiment output |

## API

All endpoints are loopback-only; writes require `Content-Type: application/json` and same-origin.

| Method | Path | |
|---|---|---|
| GET | `/api/status`, `/api/catalog`, `/api/models` | backend status, techniques, models |
| POST | `/api/run`, `/api/dose-response`, `/api/agent` | start a job → `202 {id}` |
| GET | `/api/jobs/:id` | job status, progress and result |
| POST | `/api/jobs/:id/cancel` | cancel a job |
| GET | `/api/history?limit=n`, `/api/history/:id` | saved runs |

Models are referenced by id (`gguf:<hash>` or `ollama:<name>`), never by path.

## Development

```powershell
npm test            # 38 unit + integration tests (no model needed)
npm run test:e2e    # 10 tests against the real patched engine (skipped if not built)
npm run build       # static bundle in dist/
node scripts/gen-techniques-doc.mjs   # regenerate docs/TECHNIQUES.md from the catalog
```

Engine unit tests: see [llamacpp-injection/README.md](llamacpp-injection/README.md#unit-tests).

## Documentation

- [Technique reference](docs/TECHNIQUES.md): every technique, its interventions and parameters per dose
- [Methodology](docs/METHODOLOGY.md): arms, metrics, impairment score, CIs, limitations
- [Engine](llamacpp-injection/README.md): injection sites, exact math, caveats
- [Issues and roadmap](docs/ISSUES_AND_ROADMAP.md): audit of the prototype, open issues, research roadmap
- [Contributing](CONTRIBUTING.md)

## Scope and responsible use

The lab perturbs local open-weight models on your own machine. It cannot reach hosted APIs
(OpenAI, Anthropic, ...) because they expose a few sampling options but no access to logits or
internals during generation. Drug names describe *computational* analogies; nothing here is
information about real substances.

## License

MIT, see [LICENSE](LICENSE). llama.cpp is MIT-licensed by its authors.

# Contributing

Thanks for helping. Issues and pull requests are welcome, especially new techniques, new
injection sites, calibration data on other models, and better metrics.

## Layout

| Path | What |
|---|---|
| `llamacpp-injection/` | C++ engine, llama.cpp patch, engine unit tests |
| `src/catalog.js` | technique catalog: the single source of truth (dose curves, parameters, Ollama mapping) |
| `src/metrics.js` | output-only metrics, impairment score, bootstrap |
| `src/app.js`, `index.html`, `styles.css` | UI (no framework, no inline code: CSP is strict) |
| `lib/` | server, job queue, experiment runner, llama-server pool, model registry, history |
| `scripts/` | serve, build, setup, headless experiments, `inject-env`, doc generator |
| `test/`, `test/e2e/` | Node tests (unit/integration, and against the real engine) |

The project has **no npm dependencies**. Please keep it that way unless there is a strong reason.

## Before you open a PR

```powershell
npm test
npm run build
npm run test:e2e                        # if you touched the engine, catalog or llama.cpp code
node scripts/gen-techniques-doc.mjs     # if you touched src/catalog.js; commit docs/TECHNIQUES.md
```

Engine changes: rebuild with `scripts/setup-llamacpp-injection.ps1 -Test` (or `.sh --test`).

## Adding a technique

1. Add an entry to `TECHNIQUES` in `src/catalog.js`: `analogy`, `summary`, a `mechanism` list
   that states **exactly** what runs, `sites`, a Hill `curve`, `llamacpp(e)` returning `LLM_INJ_*`
   values linear in `e`, and `ollama(e, base)` (or `() => null` with `ollamaSupport: "none"`).
2. Run the catalog tests (parameters must stay inside engine ranges and grow with dose).
3. Run `npm run test:e2e`: the technique must fire its sites and change greedy output at 500 mg.
4. Calibrate: run a sweep on a small model and check that 500 mg is clearly impaired but 50 mg is
   close to baseline. Report the numbers in the PR.
5. Regenerate `docs/TECHNIQUES.md`.

## Adding an injection site

1. Add the knob to `llm_inj_config`, parse it in `llm_inj_parse_env` (strict range, warning on
   invalid input), include it in `llm_inj_describe` and `graph_active`/`logits_active`.
2. Hook it in llama.cpp with a minimal change marked `// llm-injection:` and call
   `llm_inj_mark_fired(site)` when it runs. Regenerate the patch from `vendor/llama.cpp`:
   `git -C vendor/llama.cpp diff -- src > llamacpp-injection/patches/llama.cpp.patch` (LF endings).
3. With no env set the hook must be a strict no-op. The e2e placebo test checks this.
4. Document the exact math in `llamacpp-injection/README.md`.

## Style

- Plain modern JavaScript (ES modules, Node ≥ 20), 2-space indent, double quotes.
- C++17 matching llama.cpp style, 4-space indent.
- Metrics must never use the dose or technique parameters.
- Nothing from an HTTP client may become a file path or command-line argument.

## Responsible use

This project studies model robustness. Please do not add features aimed at evading safety
measures of hosted services or at producing harmful content.

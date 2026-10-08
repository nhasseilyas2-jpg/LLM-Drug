# LLM Drugs Runtime Lab

LLM Drugs is a local AI reliability lab for testing how real local models behave under runtime-level perturbation. The default path does not ask the model to hallucinate, does not inject drug prompts, and does not rewrite the output.

The baseline and impaired calls use identical chat messages. The drug layer changes only inference/runtime controls exposed by Ollama, then records metrics, dose curves, evaluator output, and agent-chaos traces.

## Current backend

- **Ollama runtime perturbation** through the local Node server.
- **Patched llama.cpp CLI backend** through the same UI when `llama-cli.exe` and a `.gguf` model path are configured.
- Same prompt/messages for baseline and impaired calls.
- Message fingerprint auditing for every run.
- Runtime option perturbation for `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `num_ctx`, `num_keep`, and `mirostat`.
- Persistent JSONL experiment history in `data\runs.jsonl`.
- Dose-response batches with trial aggregation.
- Optional evaluator model pass after a single run.
- Agent chaos mode that runs a multi-step loop under impaired runtime settings.

## Lower-level backend artifact

`llamacpp-drug-backend\` contains a concrete C++ sampler/KV-cache perturbation module, Windows build notes, and an auto-patcher script for a patched `llama.cpp` backend. That is the next level below Ollama: direct logit perturbation and KV-cache perturbation inside the inference engine.

Ollama does not expose arbitrary logit or activation hooks. The Ollama backend is therefore the strongest runtime perturbation available through Ollama itself; the `llamacpp-drug-backend` artifact documents and implements the internal hook needed for true logit/KV-cache drugs.

To stage the llama.cpp hook:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-llamacpp-drug-backend.ps1 -Clone -Configure -Build
```

See `llamacpp-drug-backend\BUILD_WINDOWS.md` for tool installation, existing-checkout usage, and runtime environment variables.

## Run locally

Keep Ollama running:

```powershell
ollama serve
```

Start the lab:

```powershell
npm run dev
```

Open the printed URL. If the default port is busy, the server automatically tries the next port.

## Validate

```powershell
npm test
npm run build
```

The build outputs static assets to `dist\`. Real model runs require the Node server because browser requests go through local API endpoints.

## Running patched llama.cpp from the UI

1. Build/apply the patched llama.cpp backend with `scripts\setup-llamacpp-drug-backend.ps1`.
2. Open the lab UI.
3. Set **Backend** to **Patched llama.cpp CLI logit hook**.
4. Enter the path to `llama-cli.exe`.
5. Enter the path to your `.gguf` model.
6. Type a prompt and click **Run runtime drug**.

The server launches `llama-cli.exe` for the baseline run with `LLM_DRUG_DOSE_MG=0`, then launches it again for the drugged run with `LLM_DRUG_KIND`, `LLM_DRUG_DOSE_MG`, and `LLM_DRUG_SEED` set automatically.

## API endpoints

- `GET /api/models` lists installed Ollama models.
- `POST /api/run` runs one baseline/impaired runtime trial.
- `POST /api/dose-response` runs a dose curve with repeated trials.
- `POST /api/chaos-agent` runs a multi-step impaired agent loop.
- `GET /api/history` returns persisted experiment history.

## Design boundary

This app now removes the old prompt-simulation path. Any impaired behavior in real runs comes from runtime options or, in the provided lower-level backend artifact, from inference-engine logit/KV-cache perturbation.

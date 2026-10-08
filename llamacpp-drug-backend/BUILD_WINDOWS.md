# Build and apply the llama.cpp drug backend on Windows

This is the practical path for compiling a patched `llama.cpp` runtime with real logit perturbation.

## 1. Install build tools

This machine currently has Git, but this shell did not find `cmake`, `ninja`, `cl`, `g++`, or `clang++`.

Install CMake:

```powershell
winget install Kitware.CMake
```

Install Visual Studio Build Tools with the C++ toolchain:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Restart the terminal after installing. If `cl` still is not in PATH, open **x64 Native Tools Command Prompt for VS 2022** and run the commands from there.

## 2. Clone, patch, configure, and build

From the LLM Drugs project directory:

```powershell
cd C:\Users\NHASSE\.copilot\chats\2026-09-24\supreme-succotash-6d0e4541
powershell -ExecutionPolicy Bypass -File .\scripts\setup-llamacpp-drug-backend.ps1 -Clone -Configure -Build
```

If you already have a llama.cpp checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-llamacpp-drug-backend.ps1 -LlamaCppDir C:\src\llama.cpp -Configure -Build
```

The script does three source changes:

1. Copies `llm_drug_sampler.h` and `llm_drug_sampler.cpp` into `llama.cpp\src`.
2. Adds `llm_drug_sampler.cpp` to `src\CMakeLists.txt`.
3. Inserts a logit perturbation hook in `src\llama-sampler.cpp` immediately before `llama_sampler_apply(smpl, &cur_p);`.

## 3. Run with a drug

Use a local GGUF model path:

```powershell
$env:LLM_DRUG_KIND = "hallucinogen"
$env:LLM_DRUG_DOSE_MG = "500"
$env:LLM_DRUG_SEED = "1234"
.\vendor\llama.cpp\build\bin\Release\llama-cli.exe -m C:\models\your-model.gguf -p "Who invented the telephone?"
```

To turn the drug off:

```powershell
Remove-Item Env:\LLM_DRUG_DOSE_MG
```

Supported `LLM_DRUG_KIND` values:

- `hallucinogen`
- `amnesia`
- `delusion`
- `ego`
- `confusion`
- `creativity`
- `paranoia`

## 4. Notes

- The auto-patcher implements the logit hook. That is the most stable internal hook across current llama.cpp revisions.
- KV-cache amnesia hooks are more revision-sensitive because llama.cpp has several cache implementations. The module already includes `llm_drugs_should_drop_kv(...)`; add it near the KV write path for the specific cache implementation you are using.
- If CMake generates binaries under `build\bin\` instead of `build\bin\Release\`, use that path instead.

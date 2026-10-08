param(
    [string]$LlamaCppDir = (Join-Path (Get-Location) "vendor\llama.cpp"),
    [switch]$Clone,
    [switch]$Configure,
    [switch]$Build,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Require-Command($Name, $InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is not available. $InstallHint"
    }
}

function Insert-Once($Path, $Needle, $Insert, $Description) {
    $content = Get-Content $Path -Raw
    if ($content.Contains($Insert.Trim())) {
        Write-Output "$Description already present."
        return
    }
    if (-not $content.Contains($Needle)) {
        throw "Could not find patch point in $Path for: $Needle"
    }
    $content = $content.Replace($Needle, "$Needle`r`n$Insert")
    Set-Content -Path $Path -Value $content -NoNewline
    Write-Output "Patched $Description."
}

function Replace-Once($Path, $Needle, $Replacement, $Description) {
    $content = Get-Content $Path -Raw
    if ($content.Contains("LLM_DRUGS_LOGIT_HOOK_BEGIN")) {
        Write-Output "$Description already present."
        return
    }
    if (-not $content.Contains($Needle)) {
        throw "Could not find patch point in $Path for: $Needle"
    }
    $content = $content.Replace($Needle, $Replacement)
    Set-Content -Path $Path -Value $content -NoNewline
    Write-Output "Patched $Description."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "llamacpp-drug-backend"
$srcDir = Join-Path $LlamaCppDir "src"

if ($Clone -and -not (Test-Path $LlamaCppDir)) {
    Require-Command git "Install Git or clone llama.cpp yourself."
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LlamaCppDir) | Out-Null
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaCppDir
}

if (-not (Test-Path $srcDir)) {
    throw "llama.cpp src directory was not found: $srcDir. Pass -Clone or set -LlamaCppDir to an existing llama.cpp checkout."
}

$samplerCpp = Join-Path $srcDir "llama-sampler.cpp"
$cmakeFile = Join-Path $srcDir "CMakeLists.txt"
if (-not (Test-Path $samplerCpp)) {
    throw "Expected modern llama.cpp sampler file not found: $samplerCpp"
}
if (-not (Test-Path $cmakeFile)) {
    throw "Expected llama.cpp CMake file not found: $cmakeFile"
}

Copy-Item (Join-Path $backendDir "llm_drug_sampler.h") (Join-Path $srcDir "llm_drug_sampler.h") -Force
Copy-Item (Join-Path $backendDir "llm_drug_sampler.cpp") (Join-Path $srcDir "llm_drug_sampler.cpp") -Force
Write-Output "Copied drug sampler module into $srcDir."

Insert-Once `
    -Path $cmakeFile `
    -Needle "    llama-sampler.cpp" `
    -Insert "    llm_drug_sampler.cpp" `
    -Description "CMake source registration"

Insert-Once `
    -Path $samplerCpp `
    -Needle '#include "llama-grammar.h"' `
    -Insert '#include "llm_drug_sampler.h"' `
    -Description "drug sampler include"

$needle = "    llama_sampler_apply(smpl, &cur_p);"
$hook = @'
    // LLM_DRUGS_LOGIT_HOOK_BEGIN
    {
        llm_drug_config drug_cfg = llm_drugs_make_config_from_env();
        if (drug_cfg.dose_mg > 0.0f) {
            std::vector<float> drug_logits(cur_p.size);
            for (size_t i = 0; i < cur_p.size; ++i) {
                drug_logits[i] = cur_p.data[i].logit;
            }
            llm_drugs_apply_logits(drug_logits.data(), static_cast<int>(drug_logits.size()), drug_cfg);
            for (size_t i = 0; i < cur_p.size; ++i) {
                cur_p.data[i].logit = drug_logits[i];
            }
            cur_p.sorted = false;
        }
    }
    // LLM_DRUGS_LOGIT_HOOK_END

    llama_sampler_apply(smpl, &cur_p);
'@
Replace-Once -Path $samplerCpp -Needle $needle -Replacement $hook -Description "logit perturbation hook"

if ($Configure -or $Build) {
    Require-Command cmake "Install with: winget install Kitware.CMake"
    $buildDir = Join-Path $LlamaCppDir "build"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    cmake -S $LlamaCppDir -B $buildDir -DGGML_NATIVE=OFF
}

if ($Build) {
    cmake --build (Join-Path $LlamaCppDir "build") --config Release
}

Write-Output ""
Write-Output "llama.cpp drug backend is staged."
Write-Output "Activate at runtime with environment variables, for example:"
Write-Output '  $env:LLM_DRUG_KIND = "hallucinogen"'
Write-Output '  $env:LLM_DRUG_DOSE_MG = "500"'
Write-Output '  $env:LLM_DRUG_SEED = "1234"'
Write-Output '  .\vendor\llama.cpp\build\bin\Release\llama-cli.exe -m C:\models\model.gguf -p "Who invented the telephone?"'

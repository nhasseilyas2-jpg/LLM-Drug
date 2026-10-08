<#
.SYNOPSIS
  Builds the patched llama-server used by the LLM Injection Runtime Lab.

.DESCRIPTION
  1. Clones llama.cpp into vendor\llama.cpp (or reuses it) and checks out the pinned commit.
  2. Applies llamacpp-injection\patches\llama.cpp.patch (skipped if already applied).
  3. Copies the injection engine sources into llama.cpp\src.
  4. Configures and builds llama-server with CMake.
  Optionally builds and runs the engine unit tests (-Test).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-llamacpp-injection.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup-llamacpp-injection.ps1 -Cuda -Test
#>
param(
    [string]$LlamaCppDir = "",
    [string]$Commit = "",
    [switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Test,
    [switch]$SkipCheckout,
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 1)
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$inj = Join-Path $root "llamacpp-injection"
if (-not $LlamaCppDir) { $LlamaCppDir = Join-Path $root "vendor\llama.cpp" }
if (-not $Commit) { $Commit = (Get-Content (Join-Path $inj "patches\LLAMA_CPP_COMMIT") -Raw).Trim() }
$patch = Join-Path $inj "patches\llama.cpp.patch"

function Need($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "$name not found. $hint" }
}
function Invoke-Git { & git -C $LlamaCppDir @args; if ($LASTEXITCODE -ne 0) { throw "git $args failed" } }

Need git "Install Git: https://git-scm.com/"
Need cmake "Install CMake 3.21+ and a C++17 compiler (Visual Studio 2022 Build Tools on Windows)."

if (-not (Test-Path (Join-Path $LlamaCppDir ".git"))) {
    Write-Host "Cloning llama.cpp into $LlamaCppDir"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LlamaCppDir) | Out-Null
    & git clone https://github.com/ggml-org/llama.cpp.git $LlamaCppDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$ErrorActionPreference = "Continue"  # native stderr must not abort probes on Windows PowerShell
& git -C $LlamaCppDir apply --check --reverse $patch 2>$null
$alreadyPatched = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"

if (-not $alreadyPatched) {
    if (-not $SkipCheckout) {
        $dirty = & git -C $LlamaCppDir status --porcelain --untracked-files=no
        if ($dirty) { throw "$LlamaCppDir has local changes. Commit/stash them or pass -LlamaCppDir to a clean checkout." }
        & git -C $LlamaCppDir cat-file -e "$Commit^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { Invoke-Git fetch origin $Commit }
        Write-Host "Checking out pinned llama.cpp commit $Commit"
        Invoke-Git -c advice.detachedHead=false checkout $Commit
    }
    Write-Host "Applying injection patch"
    Invoke-Git apply --whitespace=nowarn $patch
} else {
    Write-Host "Injection patch already applied."
}

foreach ($f in "llm-injection.h", "llm-injection.cpp", "llm-injection-graph.cpp") {
    Copy-Item (Join-Path $inj "src\$f") (Join-Path $LlamaCppDir "src\$f") -Force
}
Write-Host "Engine sources copied."

$build = Join-Path $LlamaCppDir "build"
$cfg = @("-S", $LlamaCppDir, "-B", $build, "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON")
if ($Cuda) { $cfg += "-DGGML_CUDA=ON" }
if ($Vulkan) { $cfg += "-DGGML_VULKAN=ON" }
& cmake @cfg
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
& cmake --build $build --config Release --target llama-server -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$exe = Get-ChildItem -Path (Join-Path $build "bin") -Recurse -Filter "llama-server*" | Where-Object { $_.Extension -in ".exe", "" } | Select-Object -First 1
Write-Host "Built: $($exe.FullName)"

if ($Test) {
    $tb = Join-Path $root "build\engine-tests"
    & cmake -S (Join-Path $inj "test") -B $tb
    if ($LASTEXITCODE -ne 0) { throw "Engine test configure failed" }
    & cmake --build $tb --config Release -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "Engine test build failed" }
    $t = Get-ChildItem -Path $tb -Recurse -Filter "test-llm-injection*" | Where-Object { $_.Extension -in ".exe", "" } | Select-Object -First 1
    & $t.FullName
    if ($LASTEXITCODE -ne 0) { throw "Engine unit tests failed" }
}

Write-Host ""
Write-Host "Done. Start the lab with: npm start   (put .gguf models in models\ or set LLAMA_MODELS_DIR)"

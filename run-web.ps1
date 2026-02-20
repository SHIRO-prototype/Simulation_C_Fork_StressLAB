# Start StressLAB dashboard (Vite) from repo root.
# Uses repo-local Node/npm when available.

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$webDir = Join-Path $PSScriptRoot "web"

$candidateNodeDirs = @(
    (Join-Path $repoRoot ".tools\node"),
    (Join-Path $repoRoot "tools\node"),
    (Join-Path $repoRoot ".node"),
    (Join-Path $repoRoot "node")
)

foreach ($dir in $candidateNodeDirs) {
    $nodeExe = Join-Path $dir "node.exe"
    if (Test-Path $nodeExe) {
        $env:PATH = "$dir;$env:PATH"
        break
    }
}

$npmCmd = "npm"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    foreach ($dir in $candidateNodeDirs) {
        $candidateNpm = Join-Path $dir "npm.cmd"
        if (Test-Path $candidateNpm) {
            $npmCmd = $candidateNpm
            break
        }
    }
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Node.js runtime not found in this repo shell." -ForegroundColor Yellow
    Write-Host "  - Use the repo-local Node/npm toolchain configured for this project." -ForegroundColor Gray
    Write-Host "  - If you keep Node in-repo, place it under .tools/node, tools/node, .node, or node." -ForegroundColor Gray
    Write-Host "  - See web/README.md for repo-local run instructions." -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Set-Location $webDir
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies (npm install)..." -ForegroundColor Cyan
    & $npmCmd install
}
Write-Host "Starting Vite dev server..." -ForegroundColor Cyan
& $npmCmd run dev

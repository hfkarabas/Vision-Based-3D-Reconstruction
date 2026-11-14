<#
.SYNOPSIS
  Re-run everything after ODM on existing smoke-suite outputs (ODM products reused), so the
  suite reflects the current pipeline without repeating hours of photogrammetry.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\refresh_smoke.ps1 -Threads 10
#>
param([int]$Threads = 10, [string]$Out = "outputs\smoke")
$ErrorActionPreference = "Continue"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$registry = Get-Content (Join-Path $Repo "configs\dataset_registry.json") -Raw | ConvertFrom-Json
foreach ($s in $registry.datasets | Where-Object { $_.role -eq "smoke" -and $_.local_path }) {
    $dest = Join-Path $Out $s.id
    if (-not (Test-Path (Join-Path $dest "odm\project\odm_dem\dsm.tif"))) { continue }
    Remove-Item (Join-Path $dest "buildings"), (Join-Path $dest "terrain"), (Join-Path $dest "detect") -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $($s.id) ..."
    & $Python -m uav3d run $s.local_path --out $dest --reuse-odm --threads $Threads
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $($s.id) exit=$LASTEXITCODE"
}
& $Python (Join-Path $Repo "bench\summarize_smoke.py") $Out

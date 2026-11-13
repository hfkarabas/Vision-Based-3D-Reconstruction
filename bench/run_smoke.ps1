<#
.SYNOPSIS
  Generality (smoke) suite: run `uav3d run` unchanged on every local "smoke" dataset of
  configs/dataset_registry.json (DATASET_PROTOCOL.md section 3), one after another.
.DESCRIPTION
  - No dataset-specific options: every dataset gets the identical command.
  - A failing dataset does not stop the suite; its run.json records the stage and reason.
  - Keeps Windows awake while running (no sleep), restores the normal state at the end.
  - Writes outputs\smoke\summary.md and summary.csv via bench\summarize_smoke.py.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_smoke.ps1 -Threads 10
#>
param(
    [int]$Threads = 0,
    [string[]]$Exclude = @("odm_ziegeleipark"),
    [string]$Out = "outputs\smoke"
)
$ErrorActionPreference = "Continue"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

try {
    $registry = Get-Content (Join-Path $Repo "configs\dataset_registry.json") -Raw | ConvertFrom-Json
    $sets = $registry.datasets | Where-Object {
        $_.role -eq "smoke" -and $_.local_path -and ($Exclude -notcontains $_.id) -and (Test-Path (Join-Path $Repo $_.local_path))
    }
    New-Item -ItemType Directory -Force $Out | Out-Null
    $started = Get-Date
    Write-Host "Smoke suite: $($sets.Count) datasets, threads=$(if ($Threads) { $Threads } else { 'all' }), started $started"
    foreach ($s in $sets) {
        $dest = Join-Path $Out $s.id
        $cmdArgs = @("-m", "uav3d", "run", $s.local_path, "--out", $dest)
        if ($Threads -gt 0) { $cmdArgs += @("--threads", "$Threads") }
        Write-Host ""
        Write-Host "[$(Get-Date -Format HH:mm:ss)] $($s.id) ($($s.images) images) ..."
        & $Python @cmdArgs
        Write-Host "[$(Get-Date -Format HH:mm:ss)] $($s.id) exit=$LASTEXITCODE"
        & $Python (Join-Path $Repo "bench\summarize_smoke.py") $Out | Out-Null
    }
    & $Python (Join-Path $Repo "bench\summarize_smoke.py") $Out
    Write-Host ""
    Write-Host "Done in $([math]::Round(((Get-Date) - $started).TotalHours, 2)) h. Summary: $Out\summary.md"
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}

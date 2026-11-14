<#
.SYNOPSIS
  After the hold-out chain (run_holdout_adaptive.ps1): real subset validation on UseGeo-1 at budget 0.5 with
  the frozen gain model, then the replay of all methods on the hold-out blocks with the same frozen model.
  Runs one job at a time (RAM).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_after_holdout.ps1 -Threads 10
#>
param([int]$Threads = 10)
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$env:PYTHONPATH = "src"
Add-Type -Namespace Win32 -Name Power2 -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Win32.Power2]::SetThreadExecutionState([uint32]"0x80000001")
function Say($m) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $m" }
try {
    Say "waiting for run_holdout_adaptive.ps1 ..."
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -like '*run_holdout_adaptive.ps1*' }) {
        Start-Sleep 60
    }
    Say "UseGeo-1 subset validation (budget 0.5) ..."
    & $Python bench\validate_subsets.py outputs\tune\usegeo_1 datasets\usegeo\dataset-1\LiDAR_dataset1.las `
        --budget 0.5 --model configs\gain_model.json --threads $Threads
    Say "validation exit=$LASTEXITCODE"
    foreach ($n in 2, 3) {
        $out = "outputs\holdout\usegeo_${n}_unsplit"
        if (Test-Path "$out\tile_gains.json") {
            Say "UseGeo-$n replay with the frozen model ..."
            & $Python bench\adaptive_replay.py $out --model configs\gain_model.json
            Say "replay exit=$LASTEXITCODE"
        } else { Say "UseGeo-${n}: no tile_gains.json, skipped" }
    }
}
finally {
    [void][Win32.Power2]::SetThreadExecutionState([uint32]"0x80000000")
}

<#
.SYNOPSIS
  AirZoo urban and rural (oblique capture) with the oblique mode, after the hold-out chain and its replays
  (run_holdout_adaptive.ps1, run_after_holdout.ps1) have finished (RAM). Fresh output folders (_oblique);
  the earlier runs stay for comparison.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_oblique_after.ps1 -Threads 10
#>
param([int]$Threads = 10)
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$env:PYTHONPATH = "src"
Add-Type -Namespace Win32 -Name Power3 -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Win32.Power3]::SetThreadExecutionState([uint32]"0x80000001")
function Say($m) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $m" }
try {
    # one job at a time: two scoring jobs in parallel ended in a system crash (2026-11-15 15:00)
    Say "waiting for the hold-out chain and its replays (RAM) ..."
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -like '*run_holdout_adaptive.ps1*' -or $_.CommandLine -like '*run_after_holdout.ps1*' }) {
        Start-Sleep 60
    }
    # per-tile scores and frozen-model replays of the hold-out blocks, where missing (UseGeo-2 failed in the
    # chain for lack of ODM's overlap.tif, fixed in bench/tile_gains.py; UseGeo-3 was cut by the crash)
    foreach ($n in 3, 2) {
        $u = "outputs\holdout\usegeo_$($n)_unsplit"
        if (-not (Test-Path "$u\tile_gains.json")) {
            Say "UseGeo-$n per-tile scores ..."
            & $Python bench\tile_gains.py $u "datasets\usegeo\dataset-$n\LiDAR_dataset$n.las" *> "$u\tile_gains.log"
            Say "tile_gains exit=$LASTEXITCODE"
        }
        if ((Test-Path "$u\tile_gains.json") -and -not (Test-Path "$u\adaptive_replay.json")) {
            Say "UseGeo-$n replay with the frozen model ..."
            & $Python bench\adaptive_replay.py $u --model configs\gain_model.json *> "$u\adaptive_replay.log"
            Say "replay exit=$LASTEXITCODE"
        }
    }
    foreach ($s in @(@("urban", "Scene1"), @("rural", "Scene2"))) {
        $out = "outputs\tune\airzoo_$($s[0])_oblique"
        Say "AirZoo $($s[0]) (oblique mode) -> $out"
        & $Python -m uav3d run "datasets\airzoo\$($s[0])\$($s[1])\query" --out $out --threads $Threads
        Say "exit=$LASTEXITCODE"
    }
}
finally {
    [void][Win32.Power3]::SetThreadExecutionState([uint32]"0x80000000")
}

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
    Say "waiting for the hold-out chain and its replays ..."
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -like '*run_holdout_adaptive.ps1*' -or $_.CommandLine -like '*run_after_holdout.ps1*' }) {
        Start-Sleep 60
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

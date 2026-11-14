<#
.SYNOPSIS
  Overnight: reconstruct UseGeo hold-out strips (2, 3) keeping ODM intermediates (needed to
  re-densify regions in the adaptive experiment), then evaluate each against its LiDAR.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_usegeo_night.ps1 -Threads 10
#>
param([int]$Threads = 10, [int[]]$Strips = @(2, 3), [int]$KeepMinFreeGb = 90)
$ErrorActionPreference = "Continue"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")
try {
    foreach ($n in $Strips) {
        $out = "outputs\holdout\usegeo_$n"  # hold-out: kept apart from outputs\tune (protocol)
        # intermediates (~40 GB per 250 images) only while the disk can take them
        $free = [math]::Round((Get-PSDrive C).Free / 1GB)
        $keep = @(); if ($free -ge $KeepMinFreeGb) { $keep = @("--keep-odm-intermediates") }
        Write-Host "[$(Get-Date -Format HH:mm:ss)] UseGeo-$n ODM (free $free GB, keep=$($keep.Count -gt 0)) ..."
        & $Python -m uav3d run "datasets\usegeo\dataset-$n\uav3d_input" --out $out --threads $Threads `
            --skip-buildings @keep
        $code = $LASTEXITCODE
        Write-Host "[$(Get-Date -Format HH:mm:ss)] UseGeo-$n exit=$code"
        if ($code -eq 0) {
            & $Python -m uav3d.evaluate.geometry $out "datasets\usegeo\dataset-$n\LiDAR_dataset$n.las"
        }
        Write-Host "[$(Get-Date -Format HH:mm:ss)] UseGeo-$n done; free $([math]::Round((Get-PSDrive C).Free/1GB)) GB"
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000000")
}

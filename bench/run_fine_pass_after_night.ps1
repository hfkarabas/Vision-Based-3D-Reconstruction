<#
.SYNOPSIS
  Wait for the UseGeo hold-out night script to finish, then run the UseGeo-1 fine pass (tuning data).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_fine_pass_after_night.ps1 -Threads 10
#>
param([int]$Threads = 10)
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")
try {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] waiting for run_usegeo_night.ps1 ..."
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -like '*run_usegeo_night.ps1*' }) {
        Start-Sleep 60
    }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] UseGeo-1 fine pass (level 3, 3x3 tiles) ..."
    & $Python bench\fine_pass.py outputs\tune\usegeo_1\odm\project outputs\tune\usegeo_1\fine_pass `
        --grid 3 --level 3 --k 20 --threads $Threads
    Write-Host "[$(Get-Date -Format HH:mm:ss)] fine pass exit=$LASTEXITCODE; free $([math]::Round((Get-PSDrive C).Free/1GB)) GB"
}
finally {
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000000")
}

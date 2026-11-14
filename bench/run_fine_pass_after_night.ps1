<#
.SYNOPSIS
  Wait for the UseGeo hold-out night script to finish, then run the UseGeo-1 fine pass (tuning data)
  and the per-tile gain analysis against the UseGeo-1 LiDAR.
.NOTES
  ODM's "medium" pass on 42 MP UseGeo images is OpenMVS level 3 (994 px) because of --min-resolution 640;
  the fine pass is therefore level 2 (1988 px). fine_pass.py refuses a level that is not finer.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_fine_pass_after_night.ps1 -Threads 10 -Level 2
#>
param([int]$Threads = 10, [int]$Level = 2, [int]$SubSceneArea = 400000)
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
    Write-Host "[$(Get-Date -Format HH:mm:ss)] UseGeo-1 fine pass (level ${Level}: depth maps, sub-scenes, fuse) ..."
    & $Python bench\fine_pass.py outputs\tune\usegeo_1\odm\project outputs\tune\usegeo_1\fine_pass `
        --level $Level --threads $Threads --sub-scene-area $SubSceneArea
    $code = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format HH:mm:ss)] fine pass exit=$code; free $([math]::Round((Get-PSDrive C).Free/1GB)) GB"
    if ($code -eq 0) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] per-tile gains vs LiDAR ..."
        & $Python bench\tile_gains.py outputs\tune\usegeo_1 datasets\usegeo\dataset-1\LiDAR_dataset1.las
        Write-Host "[$(Get-Date -Format HH:mm:ss)] tile gains exit=$LASTEXITCODE"
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000000")
}

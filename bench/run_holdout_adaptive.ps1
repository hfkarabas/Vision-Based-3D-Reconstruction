<#
.SYNOPSIS
  Hold-out data for the adaptive experiment (UseGeo-2, UseGeo-3): unsplit SfM, coarse pass (level 3, ODM's
  "medium"), fine pass (level 2), tile image sets, per-tile scores against LiDAR.
  Waits for the UseGeo-1 chain (run_fine_pass_after_night.ps1) to finish first (RAM).
.NOTES
  Protocol: the gain model is fitted on UseGeo-1 only and frozen BEFORE the hold-out tile_gains outputs are
  read; this script therefore does not run the replay on the hold-out blocks.
  Disk: after each block the regenerable OpenMVS intermediates (depth maps, .mvs, .ply) are deleted; the
  clouds (coarse_pass/fine.laz, fine_pass/fine.laz), records and the SfM (with undistorted images) stay.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File bench\run_holdout_adaptive.ps1 -Threads 10
#>
param([int]$Threads = 10, [int[]]$Sets = @(2, 3))
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")
function Say($m) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $m" }
function FreeGb { [math]::Round((Get-PSDrive C).Free / 1GB) }
try {
    Say "waiting for run_fine_pass_after_night.ps1 ..."
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -like '*run_fine_pass_after_night.ps1*' }) {
        Start-Sleep 60
    }
    foreach ($n in $Sets) {
        $out = "outputs\holdout\usegeo_${n}_unsplit"
        $proj = "$out\odm\project"
        if ((FreeGb) -lt 90 -and (Test-Path "outputs\holdout\usegeo_$n\odm\project\submodels")) {
            Say "free $(FreeGb) GB: pruning the split run's ODM intermediates (standard odm.prune)"
            & $Python -c "from pathlib import Path; from uav3d import odm; print(odm.prune(Path(r'outputs\holdout\usegeo_$n\odm\project')))"
        }
        Say "UseGeo-$n SfM (unsplit) ... free $(FreeGb) GB"
        if (-not (Test-Path "$out\sfm.json")) {
            & $Python bench\holdout_sfm.py "datasets\usegeo\dataset-$n\uav3d_input" $out --threads $Threads
            if ($LASTEXITCODE -ne 0) { Say "SfM failed ($LASTEXITCODE)"; continue }
        }
        Say "UseGeo-$n coarse pass (level 3) ..."
        & $Python bench\fine_pass.py $proj "$out\coarse_pass" --level 3 --coarse --threads $Threads --sub-scene-area 660000
        if ($LASTEXITCODE -ne 0) { Say "coarse pass failed ($LASTEXITCODE)"; continue }
        Say "UseGeo-$n fine pass (level 2) ..."
        & $Python bench\fine_pass.py $proj "$out\fine_pass" --level 2 --threads $Threads --sub-scene-area 400000
        if ($LASTEXITCODE -ne 0) { Say "fine pass failed ($LASTEXITCODE)"; continue }
        Say "UseGeo-$n tile image sets ..."
        & $Python bench\tile_shots.py $out --k 50 200
        Say "UseGeo-$n per-tile scores (output not read before the model is frozen) ..."
        & $Python bench\tile_gains.py $out "datasets\usegeo\dataset-$n\LiDAR_dataset$n.las" *> "$out\tile_gains.log"
        Say "tile_gains exit=$LASTEXITCODE"
        foreach ($d in "coarse_pass", "fine_pass") {
            Remove-Item -Recurse -Force "$out\$d\depthmaps" -ErrorAction SilentlyContinue
            Get-ChildItem "$out\$d" -File | Where-Object { $_.Extension -in ".mvs", ".ply", ".dmap" } |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
        Say "UseGeo-$n done; free $(FreeGb) GB"
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000000")
}

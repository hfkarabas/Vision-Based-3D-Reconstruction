[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Project,
    [Parameter(Mandatory)][string]$Output,
    [string]$Blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
)
$ErrorActionPreference = 'Stop'
$env:OMEGA_PROJECT_ROOT = (Resolve-Path -LiteralPath $Project).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$baseline = Join-Path $env:OMEGA_PROJECT_ROOT 'outputs/mission-omega/building-recovery'
$python = Join-Path $baseline 'runtime/venv/Scripts/python.exe'
$outputPath = [IO.Path]::GetFullPath($Output, (Get-Location).Path)
if (Test-Path -LiteralPath $outputPath) { throw "Use a fresh output directory: $outputPath" }
$env:BLENDER_USER_RESOURCES = Join-Path $outputPath 'blender-user'

function Run-Stage {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code $LASTEXITCODE." }
}

Run-Stage $python @("$PSScriptRoot/recover.py", $baseline, $outputPath)
Run-Stage $Blender @('--background', '--factory-startup', '--python-exit-code', '2',
    '--python', "$PSScriptRoot/build.py", '--', '--output', $outputPath,
    '--height-grid', "$outputPath/terrain.npz", '--name', 'urban_white_lod1_expanded', '--max-burial', '12')
Run-Stage $Blender @('--background', '--factory-startup', '--python-exit-code', '2',
    '--python', "$env:OMEGA_PROJECT_ROOT/scripts/omega_building_reopen.py", '--',
    '--scene', "$outputPath/urban_white_lod1_expanded.blend",
    '--validation', "$outputPath/reports/blender_glb_reimport.json",
    '--render', "$outputPath/preview/oblique_reopened.png",
    '--report', "$outputPath/reports/blender_reopen.json")
Run-Stage $python @("$PSScriptRoot/publish.py", $env:OMEGA_PROJECT_ROOT, $outputPath)

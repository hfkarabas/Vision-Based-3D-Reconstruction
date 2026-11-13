param(
    [Parameter(Mandatory)][string]$PilotOutput
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pilot = (Resolve-Path -LiteralPath $PilotOutput).Path
$inputs = Get-Content -LiteralPath (Join-Path $pilot 'inputs.json') -Raw | ConvertFrom-Json
$previous = Get-Content -LiteralPath (Join-Path $pilot 'reconstruction.json') -Raw | ConvertFrom-Json
if ($null -eq $previous.returncode -or $previous.returncode -ne 0 -or
    !(Test-Path -LiteralPath (Join-Path $pilot 'colmap/dense/fused.ply'))) {
    throw 'A completed COLMAP reconstruction in this pilot directory is required.'
}
if (!$inputs.images -or !$inputs.crs -or $inputs.extent.Count -ne 4) {
    throw 'The pilot input manifest is incomplete.'
}

$runtime = Join-Path $root 'outputs/mission-omega/building-recovery/runtime'
$tools = Join-Path $root 'outputs/airzoo-benchmark/geometric-root-cause-recovery'
$python = Join-Path $runtime 'venv/Scripts/python.exe'
$images = Split-Path -Parent $inputs.images[0].path

# Pass the original image order without modifying its saved input manifest.
$selection = [IO.Path]::GetTempFileName()
try {
    @{images = @($inputs.images | ForEach-Object { Split-Path -Leaf $_.path })} |
        ConvertTo-Json | Set-Content -LiteralPath $selection -Encoding utf8
    $arguments = @((Join-Path $PSScriptRoot 'omega_uav_pipeline.py'),
        '--images', $images, '--selection', $selection, '--crs', $inputs.crs, '--extent') +
        @($inputs.extent) + @(
            '--settings', (Join-Path $root 'configs/omega_reusable_uav.json'),
            '--colmap', (Join-Path $tools 'colmap-cuda/bin/colmap.exe'),
            '--pycolmap', (Join-Path $tools 'colmap-python'),
            '--pdal', (Join-Path $root 'outputs/mission-omega/tools/gis/Library/bin/pdal.exe'),
            '--blender', 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe',
            '--hub-cache', (Join-Path $runtime 'hub'), '--output', $pilot, '--through', 'orthophoto'
        )
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Pilot orthophoto stage failed with exit code $LASTEXITCODE." }
} finally {
    Remove-Item -LiteralPath $selection
}

[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Output,
    [string]$Blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe',
    [string]$Regions = 'configs/omega_terrain_urban_regions.json',
    [string]$Views = 'configs/omega_terrain_urban_views.json'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = [IO.Path]::GetFullPath($Output, (Get-Location).Path)
if (Test-Path -LiteralPath $out) { throw 'Use a new output directory.' }

function Run-Stage([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code $LASTEXITCODE." }
}

Push-Location $root
try {
    $omega = 'outputs/mission-omega'
    $base = "$omega/building-recovery-expanded"
    $python = "$omega/building-recovery/runtime/venv/Scripts/python.exe"
    $pdal = "$omega/tools/gis/Library/bin/pdal.exe"
    $dsm = "$omega/building-recovery/survey-ground/dsm_on_terrain_grid.tif"
    $env:OMEGA_PROJECT_ROOT = $root
    $env:BLENDER_USER_RESOURCES = Join-Path $out 'validation/blender-user'
    $script = 'scripts/omega_terrain_cleanup.py'
    $blenderArgs = @('--background','--factory-startup','--python-exit-code','2')
    Run-Stage $python @($script,'classify','--dsm',$dsm,'--pdal',$pdal,'--output',"$out/validation/classification")
    Run-Stage $python @($script,'clean','--dsm',$dsm,'--base',"$base/terrain.npz",'--filters',"$out/validation/classification",
        '--role','reference-assisted','--regions',$Regions,'--footprints',"$base/building_footprints_projected.json",'--output',"$out/raster")
    Run-Stage $python @($script,'prepare-asset','--baseline',$base,'--terrain',"$out/raster",'--dsm',$dsm,
        '--regions',$Regions,'--output',"$out/asset")
    $asset = "$out/asset/urban_terrain_cleaned_white_buildings"
    $env:OMEGA_TERRAIN_URBAN = '1'
    try {
        Run-Stage $Blender ($blenderArgs + @('--python','scripts/omega_terrain_blender.py','--',
            '--output',"$out/asset",'--height-grid',"$out/raster/terrain.npz",
            '--name','urban_terrain_cleaned_white_buildings','--max-burial','12'))
    } finally {
        $env:OMEGA_TERRAIN_URBAN = '0'
    }
    Run-Stage $Blender ($blenderArgs + @('--python','scripts/omega_building_reopen.py','--','--scene',"$asset.blend",
        '--validation',"$out/asset/reports/blender_glb_reimport.json",'--render',"$out/asset/preview/oblique_reopened.png",
        '--report',"$out/validation/urban_blend_reopen.json"))
    Run-Stage $Blender ($blenderArgs + @('--python','scripts/omega_terrain_blender.py','--','compare',
        '--before',"$base/urban_white_lod1_expanded.glb",'--after',"$asset.glb",'--views',$Views,'--output',"$out/validation/matched-urban"))
} finally {
    Pop-Location
}

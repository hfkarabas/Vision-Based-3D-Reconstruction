param(
    [Parameter(Mandatory)][string]$Cloud,
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$Pdal,
    [Parameter(Mandatory)][string]$Blender,
    [Parameter(Mandatory)][string]$HubCache,
    [Parameter(Mandatory)][string]$Settings,
    [string]$Frame,
    [string]$Projection,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Name = 'uav_pilot'
)
$ErrorActionPreference = 'Stop'
$scripts = $PSScriptRoot
if ([bool]$Frame -eq [bool]$Projection) { throw 'Provide either -Frame metadata or a documented metre-based -Projection.' }
if (Test-Path -LiteralPath $Output) { throw 'Use a fresh output directory.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
$surface = Join-Path $Output 'raster'
$filters = Join-Path $Output 'classification'
$terrain = Join-Path $Output 'terrain'
$asset = Join-Path $Output 'asset'

function Invoke-Stage([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code $LASTEXITCODE for $($Arguments[0])" }
}

if ($Frame) {
    $frameData = Get-Content -LiteralPath $Frame -Raw | ConvertFrom-Json
    $frameArgs = @('--crs', $frameData.crs, '--origin') + @($frameData.origin_projected_m)
    $frameArgs += @('--transform', $frameData.transform)
} else {
    $frameArgs = @('--crs', $Projection)
}
Invoke-Stage $Python (@("$scripts/omega_pilot_surface.py", 'rasterize', '--cloud', $Cloud,
    '--pdal', $Pdal, '--settings', $Settings, '--output', $surface) + $frameArgs)
Invoke-Stage $Python @("$scripts/omega_terrain_cleanup.py", 'classify', '--dsm', "$surface/dsm.tif",
    '--pdal', $Pdal, '--output', $filters)
Invoke-Stage $Python @("$scripts/omega_terrain_cleanup.py", 'clean', '--dsm', "$surface/dsm.tif",
    '--base', "$filters/csf.tif", '--filters', $filters, '--output', $terrain, '--role', 'uav-only')
Invoke-Stage $Python @("$scripts/omega_building_rasters.py", 'prepare', '--rgb', "$surface/rgb.tif",
    '--dsm', "$surface/dsm.tif", '--dtm', "$terrain/terrain_cleaned_production.tif",
    '--support', "$terrain/masks/ground_anchor.tif", '--output', $asset,
    '--rgb-role', 'reconstruction', '--dsm-role', 'reconstruction', '--dtm-role', 'reconstruction')

$config = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json
$model = $config.building_model
if ($model.tile_px -ne 512 -or $model.overlap_px -ne 256) {
    throw 'The pinned GeoAI inference command uses 512 pixel tiles and 256 pixel overlap.'
}
Invoke-Stage $Python @("$scripts/omega_building_inference.py", '--rgb', "$asset/inference_rgb.tif",
    '--output', $asset, '--hub-cache', $HubCache, '--model', $model.repo,
    '--batch-size', $model.batch_size, '--threshold', $model.probability_threshold,
    '--minimum-area', $model.minimum_polygon_area_m2)
$actual = Get-Content "$asset/inference.json" -Raw | ConvertFrom-Json
if ($actual.model_revision -ne $model.revision -or $actual.model_checkpoint_sha256 -ne $model.sha256) {
    throw 'Inference checkpoint differs from the frozen candidate.'
}
Invoke-Stage $Python @("$scripts/omega_building_rasters.py", 'assess', '--output', $asset,
    '--polygons', "$asset/geoai_polygons.gpkg")
Invoke-Stage $Python @("$scripts/omega_building_review.py", 'overlay', '--rgb', "$asset/inference_rgb.tif",
    '--primary', "$asset/geoai_polygons.gpkg", '--projected-assessment', "$asset/building_footprints_projected.json",
    '--output', "$asset/building_detection_overlay.png")

$grid = Get-Content "$asset/grid.json" -Raw | ConvertFrom-Json -AsHashtable
$grid.texture_label = 'Top-down projection of reconstructed UAV RGB points'
$grid | ConvertTo-Json -Depth 15 | Set-Content "$asset/grid.json" -Encoding utf8
Invoke-Stage $Blender @('-b', '-t', '4', '--python-exit-code', '1', '--python',
    "$scripts/omega_building_blender.py", '--', '--height-grid', "$terrain/terrain.npz",
    '--output', $asset, '--name', $Name, '--allow-terrain-only')
Invoke-Stage $Blender @('-b', '-t', '4', '--python-exit-code', '1', '--python',
    "$scripts/omega_building_reopen.py", '--', '--scene', "$asset/$Name.blend",
    '--validation', "$asset/reports/blender_glb_reimport.json",
    '--render', "$asset/preview/native_reopen.png", '--report', "$asset/reports/blend_reopen.json")

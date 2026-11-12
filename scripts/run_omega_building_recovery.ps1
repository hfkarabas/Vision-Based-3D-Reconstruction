[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Config,
    [Parameter(Mandatory)][string]$Output
)

$ErrorActionPreference = 'Stop'
$configPath = (Resolve-Path -LiteralPath $Config).Path
$outputPath = [IO.Path]::GetFullPath($Output, (Get-Location).Path)
if (Test-Path -LiteralPath $outputPath) {
    throw "Output already exists. Use a fresh output directory: $outputPath"
}
$settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$repo = Split-Path $PSScriptRoot -Parent

function Invoke-Stage {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed ($LASTEXITCODE): $Executable $($Arguments[0])"
    }
}

function Prepare-Grid {
    param([string]$Dtm, [string]$Support, [string]$Observed, [string]$DtmRole, [string]$Destination)
    $argv = @(
        'scripts/omega_building_rasters.py', 'prepare',
        '--rgb', $settings.rgb, '--dsm', $settings.dsm, '--dtm', $Dtm,
        '--support', $Support, '--output', $Destination,
        '--rgb-role', $settings.rgb_role, '--dsm-role', $settings.dsm_role, '--dtm-role', $DtmRole
    )
    if ($Observed) { $argv += @('--observed-support', $Observed) }
    if ($settings.srs_metadata) { $argv += @('--srs-metadata', $settings.srs_metadata) }
    if ($settings.rgb_crs) { $argv += @('--rgb-crs', $settings.rgb_crs) }
    if ($settings.dsm_crs) { $argv += @('--dsm-crs', $settings.dsm_crs) }
    if ($settings.gsd) { $argv += @('--gsd', [string]$settings.gsd) }
    Invoke-Stage $settings.python $argv
}

Push-Location $repo
try {
    if ($settings.mode -notin @('existing_terrain', 'survey_assisted')) {
        throw 'Set mode to existing_terrain or survey_assisted.'
    }
    if ($settings.mode -eq 'survey_assisted' -and $settings.dsm_role -ne 'reference') {
        throw 'Survey-assisted mode requires an explicitly declared reference DSM.'
    }
    if ($settings.require_reconstruction_only -and (
        $settings.mode -ne 'existing_terrain' -or
        $settings.rgb_role -ne 'reconstruction' -or
        $settings.dsm_role -ne 'reconstruction' -or
        $settings.dtm_role -ne 'reconstruction'
    )) {
        throw 'Reconstruction-only execution requires all three source roles to be reconstruction.'
    }
    New-Item -ItemType Directory -Path $outputPath | Out-Null
    Copy-Item -LiteralPath $configPath -Destination (Join-Path $outputPath 'run_config.json')
    $segmented = Join-Path $outputPath 'segmentation'
    Prepare-Grid $settings.dtm $settings.support $settings.observed_support $settings.dtm_role $segmented
    $infer = @(
        'scripts/omega_building_inference.py', '--rgb', (Join-Path $segmented 'inference_rgb.tif'),
        '--output', $segmented
    )
    if ($settings.reuse_inference) { $infer += @('--reuse', $settings.reuse_inference) }
    else { $infer += @('--hub-cache', $settings.hub_cache) }
    if ($settings.device) { $infer += @('--device', $settings.device) }
    if ($settings.model) { $infer += @('--model', $settings.model) }
    Invoke-Stage $settings.python $infer

    $asset = $segmented
    if ($settings.mode -eq 'survey_assisted') {
        $ground = Join-Path $outputPath 'survey-ground'
        Invoke-Stage $settings.python @(
            'scripts/omega_building_survey_ground.py', '--base', $segmented, '--output', $ground,
            '--pdal', $settings.pdal
        )
        $asset = Join-Path $outputPath 'delivery'
        $groundDtm = Join-Path $ground 'dtm.tif'
        $groundSupport = Join-Path $ground 'support.tif'
        $observed = Join-Path $ground 'observed_ground.tif'
        Prepare-Grid $groundDtm $groundSupport $observed 'reference' $asset
        Invoke-Stage $settings.python @(
            'scripts/omega_building_inference.py', '--rgb', (Join-Path $asset 'inference_rgb.tif'),
            '--output', $asset, '--reuse', $segmented
        )
        $terrain = @('--height-grid', (Join-Path $ground 'blender_height_grid.npz'))
    } else {
        $terrain = @('--terrain', $settings.terrain_glb, '--terrain-validation', $settings.terrain_validation)
    }
    Invoke-Stage $settings.python @(
        'scripts/omega_building_rasters.py', 'assess', '--output', $asset,
        '--polygons', (Join-Path $asset 'geoai_polygons.gpkg')
    )
    $review = @()
    if ($settings.visual_review) { $review = @('--visual-review', $settings.visual_review) }
    $build = @(
        '--background', '--factory-startup', '--python-exit-code', '2',
        '--python', 'scripts/omega_building_blender.py', '--', '--output', $asset,
        '--name', $settings.asset_name
    ) + $terrain + $review
    try {
        Invoke-Stage $settings.blender $build
    } finally {
        Invoke-Stage $settings.python (@('scripts/omega_building_report.py', '--output', $asset) + $review)
    }
    Write-Output "Asset directory $asset"
} finally {
    Pop-Location
}

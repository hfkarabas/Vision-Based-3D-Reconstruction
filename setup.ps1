<#
.SYNOPSIS
  One-command, reproducible setup for the uav3d pipeline (Windows, PowerShell 5.1+).
.DESCRIPTION
  1. Python environment via uv (.venv, extras: gis, dev).
  2. Pinned external tools from tools.lock.json into $env:UAV3D_TOOLS (default: <repo>\.tools).
     Every download is verified against its SHA-256 before it is used.
  3. Runs `uav3d doctor`.
  Nothing is read from or written to outputs\.
.PARAMETER OdmFrom
  Copy an existing ODM install of the pinned version instead of downloading the installer.
.PARAMETER SkipTools
  Only set up the Python environment.
#>
param(
    [string]$OdmFrom = "",
    [switch]$SkipTools
)
$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
$Tools = if ($env:UAV3D_TOOLS) { $env:UAV3D_TOOLS } else { Join-Path $Repo ".tools" }
$Lock = Get-Content (Join-Path $Repo "tools.lock.json") -Raw | ConvertFrom-Json

function Repair-OdmPaths($odmDir) {
    # Same edits as ODM's winpostinstall.bat: the install embeds absolute paths, so a copied
    # or moved install must have them rewritten (done here without relying on cmd's cwd lookup).
    $venv = Join-Path $odmDir "venv"
    $sbbin = Join-Path $odmDir "SuperBuild\install\bin"
    $cv2 = Join-Path $venv "Lib\site-packages\cv2"
    Set-Content (Join-Path $venv "pyvenv.cfg") "home = $venv\Scripts`r`ninclude-system-site-packages = false" -Encoding ascii
    Set-Content (Join-Path $cv2 "config.py") "BINARIES_PATHS = [r`"$sbbin`"] + BINARIES_PATHS" -Encoding ascii
    Set-Content (Join-Path $cv2 "config-3.12.py") "PYTHON_EXTENSIONS_PATHS = [r'''$venv\lib\site-packages\cv2\python-3.12'''] + PYTHON_EXTENSIONS_PATHS" -Encoding ascii
}

function Get-Verified($name, $entry) {
    $cache = Join-Path $Tools "_downloads"
    New-Item -ItemType Directory -Force $cache | Out-Null
    $file = Join-Path $cache ([IO.Path]::GetFileName($entry.url))
    $ok = (Test-Path $file) -and ((Get-FileHash $file -Algorithm SHA256).Hash.ToLower() -eq $entry.sha256)
    if (-not $ok) {
        Write-Host "Downloading $name $($entry.version) ..."
        curl.exe -fL --retry 5 -C - -o $file $entry.url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($entry.url)" }
        $hash = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $entry.sha256) {
            Remove-Item $file -Force
            throw "SHA-256 mismatch for ${name}: got $hash, expected $($entry.sha256)"
        }
    }
    return $file
}

# 1. Python environment
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it from https://docs.astral.sh/uv/ and rerun."
}
Push-Location $Repo
try { uv sync --extra gis --extra dev; if ($LASTEXITCODE -ne 0) { throw "uv sync failed" } }
finally { Pop-Location }

if (-not $SkipTools) {
    New-Item -ItemType Directory -Force $Tools | Out-Null

    # 2a. ODM
    $odmDir = Join-Path $Tools "odm"
    $odmVersionFile = Join-Path $odmDir "VERSION"
    $haveOdm = (Test-Path $odmVersionFile) -and ((Get-Content $odmVersionFile -Raw).Trim() -eq $Lock.odm.version)
    if (-not $haveOdm) {
        if ($OdmFrom) {
            $srcVersion = (Get-Content (Join-Path $OdmFrom "VERSION") -Raw).Trim()
            if ($srcVersion -ne $Lock.odm.version) { throw "ODM at $OdmFrom is $srcVersion, lock requires $($Lock.odm.version)" }
            Write-Host "Copying ODM $srcVersion from $OdmFrom ..."
            robocopy $OdmFrom $odmDir /E /NFL /NDL /NJH /NJS /NP | Out-Null
            if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
        } else {
            $installer = Get-Verified "ODM" $Lock.odm
            Write-Host "Installing ODM into $odmDir ..."
            $p = Start-Process $installer -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/DIR=`"$odmDir`"" -Wait -PassThru
            if ($p.ExitCode -ne 0) { throw "ODM installer exited with $($p.ExitCode)" }
        }
    }
    Repair-OdmPaths $odmDir

    # 2b. Roofer
    $rooferDir = Join-Path $Tools "roofer"
    $rooferStamp = Join-Path $rooferDir ".version"
    $haveRoofer = (Test-Path $rooferStamp) -and ((Get-Content $rooferStamp -Raw).Trim() -eq $Lock.roofer.version)
    if (-not $haveRoofer) {
        $zip = Get-Verified "Roofer" $Lock.roofer
        if (Test-Path $rooferDir) { Remove-Item $rooferDir -Recurse -Force }
        Expand-Archive $zip -DestinationPath $rooferDir
        Set-Content $rooferStamp $Lock.roofer.version -Encoding ascii
    }
}

# 3. Verify
& (Join-Path $Repo ".venv\Scripts\python.exe") -m uav3d doctor
exit $LASTEXITCODE

$repo = "C:\projects\Vision-Based-3D-Reconstruction"
$prompt = Join-Path $repo "MISSION0_PROMPT.txt"
$logDir = Join-Path $repo "outputs\codex-night"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "mission0.log"

Set-Location $repo

$codex = (Get-Command codex.cmd -ErrorAction Stop).Source

Get-Content -Raw -Encoding UTF8 $prompt |
    & $codex exec --sandbox workspace-write -C $repo - *> $log

exit $LASTEXITCODE

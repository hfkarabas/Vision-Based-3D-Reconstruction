# Mission Omega — no time target and no time limit. Single Codex invocation.
# Run under your ordinary Windows account, NOT Administrator.
param(
    [string]$ProjectRoot = 'C:\projects\Vision-Based-3D-Reconstruction'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$PromptFile = Join-Path $ProjectRoot 'MISSION_OMEGA_PROMPT.txt'
$MainOutput = Join-Path $ProjectRoot 'outputs\mission-omega'
$RunLogRoot = Join-Path $MainOutput 'runner'

if (!(Test-Path -PathType Container $ProjectRoot)) { throw "Missing project: $ProjectRoot" }
if (!(Test-Path -PathType Leaf $PromptFile)) { throw "Missing prompt: $PromptFile" }

$Codex = (Get-Command codex.cmd -ErrorAction Stop).Source
& $Codex login status
if ($LASTEXITCODE -ne 0) { throw 'Codex CLI login is not working.' }

New-Item -ItemType Directory -Force -Path $RunLogRoot | Out-Null
$RunTag = (Get-Date).ToString('yyyyMMdd-HHmmss')
$OutLog = Join-Path $RunLogRoot "$RunTag.stdout.jsonl"
$ErrLog = Join-Path $RunLogRoot "$RunTag.stderr.log"
$LastMessage = Join-Path $RunLogRoot "$RunTag.last-response.txt"

$ArgsForCodex = 'exec --approve-for-me -C "{0}" --json --output-last-message "{1}" -' -f $ProjectRoot, $LastMessage
$CmdLine = '""{0}" {1}"' -f $Codex, $ArgsForCodex

Write-Host 'Mission Omega started with NO time limit and NO duration target.'
Write-Host "Project: $ProjectRoot"
Write-Host "Logs: $RunLogRoot"

$Process = Start-Process -FilePath $env:ComSpec `
    -ArgumentList @('/d', '/s', '/c', $CmdLine) `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardInput $PromptFile `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

# Wait until Codex exits normally. NO deadline check and NO taskkill.
$Process.WaitForExit()
$Process.Refresh()
$FinalExitCode = $Process.ExitCode
Write-Host "Codex exited with code $FinalExitCode."
Write-Host "Output: $MainOutput"
Write-Host "Last response: $LastMessage"
Write-Host "Read outputs\mission-omega\mission_status.json and README_OPEN_FIRST.md if created."
if ($FinalExitCode -ne 0) {
    Write-Warning "Codex failed with exit code $FinalExitCode; see $ErrLog"
}

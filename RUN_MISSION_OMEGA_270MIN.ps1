# Mission Omega: one user-provided mission prompt, unattended Codex execution window.
# Requires Codex CLI already authenticated in this same Windows user account.
# Run this script from a normal PowerShell session. Never run as Administrator by default.

param(
    [string]$ProjectRoot = 'C:\projects\Vision-Based-3D-Reconstruction',
    [int]$MaxMinutes = 270,
    [int]$MaxRounds = 4
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$PromptFile = Join-Path $ProjectRoot 'MISSION_OMEGA_PROMPT.txt'
$MainOutput = Join-Path $ProjectRoot 'outputs\mission-omega'
$RunLogRoot = Join-Path $MainOutput 'runner'
$MissionStatus = Join-Path $MainOutput 'mission_status.json'

if (!(Test-Path -PathType Container $ProjectRoot)) { throw "Missing project: $ProjectRoot" }
if (!(Test-Path -PathType Leaf $PromptFile)) { throw "Copy CODEX_MISSION_OMEGA_270MIN.txt to $PromptFile before starting." }
if ($MaxMinutes -lt 30 -or $MaxMinutes -gt 720) { throw 'MaxMinutes must be between 30 and 720.' }
if ($MaxRounds -lt 1 -or $MaxRounds -gt 12) { throw 'MaxRounds must be between 1 and 12.' }

$Codex = (Get-Command codex.cmd -ErrorAction Stop).Source
& $Codex login status
if ($LASTEXITCODE -ne 0) { throw 'Codex CLI login must be working before unattended execution.' }

New-Item -ItemType Directory -Force -Path $RunLogRoot | Out-Null
$StartedAt = Get-Date
$Deadline = $StartedAt.AddMinutes($MaxMinutes)
$RunTag = $StartedAt.ToString('yyyyMMdd-HHmmss')
$ContinuationFile = Join-Path $RunLogRoot "continue-$RunTag.txt"
$ContinuationText = @'
CONTINUE THE ORIGINAL MISSION OMEGA FROM THE EXISTING WORKSPACE STATE. Read outputs/mission-omega/mission_status.json and the previous round logs. DO NOT restart or duplicate expensive completed stages, DO NOT delete old results, and DO NOT stop at a plan. Continue real asset production, verification, and the scientific budget-aware regional experiment while honoring the ORIGINAL prompt and its safety/license/data-provenance rules. If all feasible gates are done and assets are ready for human visual inspection, update mission_status.json outcome to ready_for_human_review and finish honestly. Otherwise use your remaining execution window on the highest-value unfinished gate and save measured evidence.
'@
[System.IO.File]::WriteAllText($ContinuationFile, $ContinuationText, [System.Text.UTF8Encoding]::new($false))

Write-Host "Mission Omega started: $StartedAt"
Write-Host "Deadline (maximum window): $Deadline"
Write-Host "Project: $ProjectRoot"
Write-Host "Codex executable: $Codex"
Write-Host "Logs: $RunLogRoot"
Write-Host 'NOTE: 270 minutes is a MAXIMUM RUNNING WINDOW, not guaranteed compute time or model-quality success.'

Set-Location $ProjectRoot
$FinalExitCode = 0
for ($Round = 1; $Round -le $MaxRounds; $Round++) {
    if ((Get-Date) -ge $Deadline) { Write-Warning 'Mission deadline reached before starting another round.'; break }

    $RoundLabel = '{0}-round-{1:D2}' -f $RunTag, $Round
    $OutLog = Join-Path $RunLogRoot "$RoundLabel.stdout.jsonl"
    $ErrLog = Join-Path $RunLogRoot "$RoundLabel.stderr.log"
    $LastMessage = Join-Path $RunLogRoot "$RoundLabel.last-response.txt"
    $IsInitial = ($Round -eq 1)
    $StdinFile = if ($IsInitial) { $PromptFile } else { $ContinuationFile }

    if ($IsInitial) {
        $ArgsForCodex = 'exec --approve-for-me -C "{0}" --json --output-last-message "{1}" -' -f $ProjectRoot, $LastMessage
    } else {
        # Resume the PREVIOUS exec session; this is not a new research assignment.
        # Codex resumes the most recent session under the same working directory.
        $ArgsForCodex = 'exec resume --last -'
    }
    # cmd.exe handles codex.cmd correctly; stdin is redirected from a UTF-8 file.
    # The outer double quotes are required by cmd /s /c when the program path is quoted.
    $CmdLine = '""{0}" {1}"' -f $Codex, $ArgsForCodex

    Write-Host "Starting Codex round $Round / $MaxRounds at $(Get-Date)."
    $Process = Start-Process -FilePath $env:ComSpec `
        -ArgumentList @('/d', '/s', '/c', $CmdLine) `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardInput $StdinFile `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -PassThru

    $HitDeadline = $false
    while (-not $Process.WaitForExit(15000)) {
        if ((Get-Date) -ge $Deadline) {
            $HitDeadline = $true
            Write-Warning "Hard deadline reached; stopping process tree PID $($Process.Id). Cached work is kept."
            & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null
            break
        }
    }
    try { $Process.Refresh() } catch { }
    if (-not $HitDeadline -and $Process.HasExited) {
        $FinalExitCode = $Process.ExitCode
        Write-Host "Round $Round exited with code $FinalExitCode at $(Get-Date)."
    } else {
        $FinalExitCode = 124
    }
    if ($HitDeadline) { break }

    $Outcome = $null
    if (Test-Path -PathType Leaf $MissionStatus) {
        try {
            $StatusObject = (Get-Content -Raw -Encoding UTF8 $MissionStatus | ConvertFrom-Json)
            $Outcome = [string]$StatusObject.outcome
        } catch { Write-Warning "Status file is not valid JSON yet: $($_.Exception.Message)" }
    }
    if ($Outcome -eq 'ready_for_human_review') {
        Write-Host 'Codex reports artifacts are ready for HUMAN VISUAL REVIEW; stopping normally.'
        break
    }
    if ($Outcome -eq 'blocked') {
        Write-Warning 'Codex recorded a hard blocker. Inspect mission_status.json and stderr before retrying.'
        break
    }
    $Tail = if (Test-Path $ErrLog) { (Get-Content $ErrLog -Tail 25 -ErrorAction SilentlyContinue) -join "`n" } else { '' }
    if ($Tail -match '(?i)insufficient.quota|authentication.failed|unauthorized|rate.limit.exceeded|usage.limit') {
        Write-Warning 'Provider authorization/quota limit appears in stderr. Avoiding blind retry.'
        break
    }
    if ((Get-Date).AddMinutes(5) -ge $Deadline) {
        Write-Host 'Less than five minutes remain; no additional Codex round started.'
        break
    }
    if ($Round -lt $MaxRounds) {
        Write-Host 'Mission not yet ready for human review; resuming saved Codex session without asking for new user input.'
    }
}

Write-Host "Mission Omega runner ended at $(Get-Date); elapsed $([math]::Round(((Get-Date)-$StartedAt).TotalMinutes,1)) minutes."
Write-Host "Outputs: $MainOutput"
Write-Host "Logs: $RunLogRoot"
Write-Host 'Read outputs\mission-omega\README_OPEN_FIRST.md and mission_status.json if created.'
if ($FinalExitCode -ne 0) { Write-Warning "Last process exit code: $FinalExitCode (review logs)." }

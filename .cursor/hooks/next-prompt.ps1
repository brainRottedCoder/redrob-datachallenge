$ErrorActionPreference = "Stop"

# Read hook input from stdin (Cursor passes agent stop context)
$null = [Console]::In.ReadToEnd()

$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$queuePath = Join-Path $projectRoot ".cursor/prompt-queue.json"
$statePath = Join-Path $projectRoot ".cursor/prompt-state.json"

if (-not (Test-Path $queuePath)) {
    Write-Output '{}'
    exit 0
}

if (-not (Test-Path $statePath)) {
    @{ index = 1 } | ConvertTo-Json | Set-Content $statePath -Encoding UTF8
}

$queue = Get-Content $queuePath -Raw | ConvertFrom-Json
$state = Get-Content $statePath -Raw | ConvertFrom-Json

$idx = [int]$state.index
$total = $queue.prompts.Count

if ($idx -ge $total) {
    Write-Output '{}'
    exit 0
}

$nextPrompt = [string]$queue.prompts[$idx]
$state.index = $idx + 1
$state | ConvertTo-Json | Set-Content $statePath -Encoding UTF8

$response = @{
    followup_message = @"
[FitRank queue $($idx + 1)/$total] AUTO-CONTINUE — do not wait for user input.

$nextPrompt

MANDATORY after this phase:
1. Mark phase [x] in .cursor/tasks.md
2. Run: pytest tests/test_phase$($idx + 1)_*.py -v (or the matching phase test file)
3. git add -A; git commit -m `"Phase $($idx + 1): <short description>`"; git push origin main
4. Stop — the stop hook will inject the next phase automatically.
"@
}

$response | ConvertTo-Json -Compress
exit 0

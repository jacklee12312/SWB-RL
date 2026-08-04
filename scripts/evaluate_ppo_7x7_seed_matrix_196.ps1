Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\python.exe"
$checkpointRoot = Join-Path $projectRoot "data\checkpoints\ppo_7x7_scaling_20260801"
$reportRoot = Join-Path $projectRoot "data\reports\ppo_7x7_seed_matrix_20260804"
$statusPath = Join-Path $reportRoot "launcher_status.json"
$launcherLog = Join-Path $reportRoot "launcher.log"
$summaryPath = Join-Path $reportRoot "summary.json"
$oneMSeeds = @("20260903", "20260904", "20260905", "20260906", "20260907", "20260908")
$threeMSeeds = @("20260831", "20260901", "20260902")
$masterSeed = 20260804
$pairs = [System.Collections.Generic.List[object]]::new()
$completedRows = [System.Collections.Generic.List[object]]::new()
$activePair = ""

function Checkpoint-Path {
    param(
        [Parameter(Mandatory = $true)][string]$Seed,
        [Parameter(Mandatory = $true)][ValidateSet("1m", "3m")][string]$Tier
    )

    $filename = if ($Tier -eq "3m") { "final_3m.pt" } else { "final.pt" }
    return Join-Path $checkpointRoot "seed_$Seed\$filename"
}

for ($left = 0; $left -lt $oneMSeeds.Count; $left++) {
    for ($right = $left + 1; $right -lt $oneMSeeds.Count; $right++) {
        $learnerSeed = $oneMSeeds[$left]
        $opponentSeed = $oneMSeeds[$right]
        $pairs.Add([pscustomobject][ordered]@{
            pair_id = "1m_${learnerSeed}_vs_1m_${opponentSeed}"
            group = "one_m_internal"
            learner_seed = $learnerSeed
            learner_tier = "1m"
            opponent_seed = $opponentSeed
            opponent_tier = "1m"
            learner_checkpoint = Checkpoint-Path -Seed $learnerSeed -Tier "1m"
            opponent_checkpoint = Checkpoint-Path -Seed $opponentSeed -Tier "1m"
        })
    }
}

foreach ($learnerSeed in $threeMSeeds) {
    foreach ($opponentSeed in $oneMSeeds) {
        $pairs.Add([pscustomobject][ordered]@{
            pair_id = "3m_${learnerSeed}_vs_1m_${opponentSeed}"
            group = "three_m_vs_one_m"
            learner_seed = $learnerSeed
            learner_tier = "3m"
            opponent_seed = $opponentSeed
            opponent_tier = "1m"
            learner_checkpoint = Checkpoint-Path -Seed $learnerSeed -Tier "3m"
            opponent_checkpoint = Checkpoint-Path -Seed $opponentSeed -Tier "1m"
        })
    }
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Set-Location -LiteralPath $projectRoot

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding utf8
}

function Write-LauncherStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$ErrorMessage = ""
    )

    $completedIds = @($completedRows | ForEach-Object { $_.pair_id })
    [ordered]@{
        schema_version = 1
        state = $State
        launcher_pid = $PID
        active_pair = $activePair
        completed_pair_count = $completedRows.Count
        total_pair_count = $pairs.Count
        completed_game_count = 196 * $completedRows.Count
        total_game_count = 196 * $pairs.Count
        queued_pairs = @(
            $pairs | Where-Object {
                $_.pair_id -notin $completedIds -and $_.pair_id -ne $activePair
            } | ForEach-Object { $_.pair_id }
        )
        master_seed = $masterSeed
        games_per_pair = 196
        updated_at = [DateTimeOffset]::Now.ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Write-Summary {
    [ordered]@{
        schema_version = 1
        purpose = "pairwise 196-game full 7x7 class evaluation for six 1M seeds and three 3M-vs-six-1M cross pairs"
        master_seed = $masterSeed
        games_per_pair = 196
        expected_pair_counts = [ordered]@{
            one_m_internal = 15
            three_m_vs_one_m = 18
            total = 33
        }
        completed_pair_count = $completedRows.Count
        completed_game_count = 196 * $completedRows.Count
        pairs = @($completedRows)
        updated_at = [DateTimeOffset]::Now.ToString("o")
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
}

try {
    foreach ($pair in $pairs) {
        foreach ($checkpoint in @($pair.learner_checkpoint, $pair.opponent_checkpoint)) {
            if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                throw "Checkpoint not found: $checkpoint"
            }
        }
        $output = Join-Path $reportRoot "$($pair.pair_id).json"
        if (Test-Path -LiteralPath $output) {
            throw "Refusing to overwrite existing evaluation: $output"
        }
    }

    Write-Summary
    Write-LauncherStatus -State "starting"
    Write-LauncherLog -Message "Starting 33-pair, 6468-game evaluation queue."

    foreach ($pair in $pairs) {
        $activePair = $pair.pair_id
        $output = Join-Path $reportRoot "$($pair.pair_id).json"
        $runLog = Join-Path $reportRoot "$($pair.pair_id).log"
        Write-LauncherStatus -State "running"
        Write-LauncherLog -Message "Starting $activePair."
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $arguments = @(
            "-m", "scripts.evaluate_ppo",
            $pair.learner_checkpoint,
            "--seed-count", "2",
            "--max-agent-steps", "512",
            "--master-seed", "$masterSeed",
            "--device", "cuda",
            "--classes", "1", "2", "3", "4", "5", "6", "7",
            "--full-matchup-matrix",
            "--opponent", "historical",
            "--opponent-checkpoint", $pair.opponent_checkpoint,
            "--output", $output
        )
        & $python @arguments *> $runLog
        $exitCode = $LASTEXITCODE
        $stopwatch.Stop()
        if ($exitCode -ne 0) {
            throw "$activePair exited with code $exitCode; see $runLog"
        }

        $report = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
        if ([int]$report.metrics.games -ne 196) {
            throw "$activePair produced $($report.metrics.games) games instead of 196"
        }
        $completedRows.Add([pscustomobject][ordered]@{
            pair_id = $pair.pair_id
            group = $pair.group
            learner_seed = $pair.learner_seed
            learner_tier = $pair.learner_tier
            opponent_seed = $pair.opponent_seed
            opponent_tier = $pair.opponent_tier
            games = [int]$report.metrics.games
            learner_win_rate = [double]$report.metrics.win_rate
            confidence_interval_95 = @($report.metrics.confidence_interval_95)
            relative_elo = [double]$report.metrics.elo_relative
            terminated = [int]$report.metrics.terminated
            truncated = [int]$report.metrics.truncated
            illegal_actions = [int]$report.metrics.illegal_actions
            action_mask_mismatches = [int]$report.metrics.action_mask_mismatches
            elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
            report = $output
        })
        Write-Summary
        Write-LauncherLog -Message (
            "Completed {0}: win_rate={1:P2}, truncated={2}, elapsed={3:N1}s." -f
            $activePair,
            [double]$report.metrics.win_rate,
            [int]$report.metrics.truncated,
            $stopwatch.Elapsed.TotalSeconds
        )
        $activePair = ""
        Write-LauncherStatus -State "between_pairs"
    }

    Write-Summary
    Write-LauncherStatus -State "completed"
    Write-LauncherLog -Message "All 33 pairs and 6468 games completed."
}
catch {
    Write-Summary
    Write-LauncherStatus -State "failed" -ErrorMessage $_.Exception.Message
    Write-LauncherLog -Message "FAILED: $($_.Exception.Message)"
    exit 1
}

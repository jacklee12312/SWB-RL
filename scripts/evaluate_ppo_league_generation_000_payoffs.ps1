Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\python.exe"
$generationPath = Join-Path $projectRoot "data\reports\league_training\generation_000_manifest.json"
$protocolPath = Join-Path $projectRoot "data\reports\league_training\evaluation_protocol.json"
$reportRoot = Join-Path $projectRoot "data\reports\league_training\generation_000_payoff_evaluations"
$snapshotPath = Join-Path $projectRoot "data\reports\league_training\generation_000_training_payoff_snapshot.json"
$statusPath = Join-Path $reportRoot "launcher_status.json"
$summaryPath = Join-Path $reportRoot "summary.json"
$launcherLog = Join-Path $reportRoot "launcher.log"
$focalPolicyId = "seed_20260903_1m"
$masterSeed = 20261001
$seedCount = 2
$gamesPerPair = 196
$activePair = ""
$completedRows = [System.Collections.Generic.List[object]]::new()

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Set-Location -LiteralPath $projectRoot
& $python -m scripts.report_ppo_league_training_payoff --plan-only --check
if ($LASTEXITCODE -ne 0) {
    throw "Frozen PFSP payoff evaluation plan is missing or stale."
}

$generation = Get-Content -Raw -LiteralPath $generationPath | ConvertFrom-Json
$protocol = Get-Content -Raw -LiteralPath $protocolPath | ConvertFrom-Json
$tuningSeeds = @($protocol.seed_partitions.pfsp_tuning_match_master_seeds)
$finalSeeds = @($protocol.seed_partitions.final_evaluation_match_master_seeds)
if ($masterSeed -notin $tuningSeeds -or $masterSeed -in $finalSeeds) {
    throw "PFSP payoff master seed violates the frozen evaluator partition."
}
$focal = @($generation.entries | Where-Object {
    $_.opponent_id -eq $focalPolicyId -and $_.training_eligible
})
if ($focal.Count -ne 1) {
    throw "Expected exactly one trainable focal policy $focalPolicyId."
}
$opponents = @($generation.entries | Where-Object {
    $_.training_eligible
} | Sort-Object opponent_id)
if ($opponents.Count -ne 24) {
    throw "Generation 0 must contain exactly 24 trainable opponents."
}
$focalCheckpoint = Join-Path $projectRoot $focal[0].checkpoint_path
if (-not (Test-Path -LiteralPath $focalCheckpoint -PathType Leaf)) {
    throw "Focal checkpoint not found: $focalCheckpoint"
}

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = "{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding utf8
}

function Write-Summary {
    [ordered]@{
        schema_version = 1
        purpose = "PFSP training-only Generation 0 payoff snapshot inputs"
        data_partition = "pfsp_tuning"
        source_generation = 0
        target_generation = 1
        focal_policy_ids = @($focalPolicyId)
        match_master_seeds = @($masterSeed)
        seed_count = $seedCount
        games_per_pair = $gamesPerPair
        expected_pair_count = $opponents.Count
        completed_pair_count = $completedRows.Count
        expected_game_count = $opponents.Count * $gamesPerPair
        completed_game_count = $completedRows.Count * $gamesPerPair
        pairs = @($completedRows)
        updated_at = [DateTimeOffset]::Now.ToString("o")
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
}

function Write-LauncherStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$ErrorMessage = ""
    )
    [ordered]@{
        schema_version = 1
        state = $State
        launcher_pid = $PID
        active_pair = $activePair
        completed_pair_count = $completedRows.Count
        total_pair_count = $opponents.Count
        completed_game_count = $completedRows.Count * $gamesPerPair
        total_game_count = $opponents.Count * $gamesPerPair
        focal_policy_id = $focalPolicyId
        match_master_seed = $masterSeed
        data_partition = "pfsp_tuning"
        updated_at = [DateTimeOffset]::Now.ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Validate-CompletedReport {
    param(
        [Parameter(Mandatory = $true)]$Opponent,
        [Parameter(Mandatory = $true)][string]$Output
    )
    $report = Get-Content -Raw -LiteralPath $Output | ConvertFrom-Json
    if ([int]$report.metrics.games -ne $gamesPerPair) {
        throw "$Output does not contain $gamesPerPair games."
    }
    if (
        [int]$report.metrics.terminated -ne $gamesPerPair -or
        [int]$report.metrics.truncated -ne 0 -or
        [int]$report.metrics.illegal_actions -ne 0 -or
        [int]$report.metrics.action_mask_mismatches -ne 0
    ) {
        throw "$Output failed safety validation."
    }
    if ($report.checkpoint.sha256 -ne $focal[0].checkpoint_sha256) {
        throw "$Output uses the wrong focal checkpoint."
    }
    if (
        $report.configuration.opponent_checkpoint_sha256 -ne
        $Opponent.checkpoint_sha256
    ) {
        throw "$Output uses the wrong opponent checkpoint."
    }
    if (
        [int]$report.configuration.master_seed -ne $masterSeed -or
        [int]$report.configuration.seed_count -ne $seedCount
    ) {
        throw "$Output uses the wrong PFSP tuning seed contract."
    }
    return $report
}

try {
    Write-LauncherStatus -State "starting"
    Write-LauncherLog -Message (
        "Starting Generation 0 PFSP payoff queue: focal={0}, pairs={1}, games={2}." -f
        $focalPolicyId,
        $opponents.Count,
        ($opponents.Count * $gamesPerPair)
    )
    foreach ($opponent in $opponents) {
        $pairId = "${focalPolicyId}__vs__$($opponent.opponent_id)"
        $activePair = $pairId
        $output = Join-Path $reportRoot "$pairId.json"
        $runLog = Join-Path $reportRoot "$pairId.log"
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        if (Test-Path -LiteralPath $output -PathType Leaf) {
            $report = Validate-CompletedReport -Opponent $opponent -Output $output
            $state = "reused_valid_existing_report"
        }
        else {
            $opponentCheckpoint = Join-Path $projectRoot $opponent.checkpoint_path
            if (-not (Test-Path -LiteralPath $opponentCheckpoint -PathType Leaf)) {
                throw "Opponent checkpoint not found: $opponentCheckpoint"
            }
            Write-LauncherStatus -State "running"
            Write-LauncherLog -Message "Starting $pairId."
            $arguments = @(
                "-m", "scripts.evaluate_ppo",
                $focalCheckpoint,
                "--seed-count", "$seedCount",
                "--max-agent-steps", "512",
                "--master-seed", "$masterSeed",
                "--device", "cuda",
                "--classes", "1", "2", "3", "4", "5", "6", "7",
                "--full-matchup-matrix",
                "--opponent", "historical",
                "--opponent-checkpoint", $opponentCheckpoint,
                "--output", $output
            )
            & $python @arguments *> $runLog
            if ($LASTEXITCODE -ne 0) {
                throw "$pairId exited with code $LASTEXITCODE; see $runLog"
            }
            $report = Validate-CompletedReport -Opponent $opponent -Output $output
            $state = "completed"
        }
        $stopwatch.Stop()
        $completedRows.Add([pscustomobject][ordered]@{
            pair_id = $pairId
            focal_policy_id = $focalPolicyId
            opponent_id = $opponent.opponent_id
            games = [int]$report.metrics.games
            score_rate = [double]$report.metrics.win_rate
            confidence_interval_95 = @($report.metrics.confidence_interval_95)
            truncated = [int]$report.metrics.truncated
            elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
            state = $state
            report = $output
        })
        Write-Summary
        Write-LauncherLog -Message (
            "Completed {0}: score={1:P2}, elapsed={2:N1}s, state={3}." -f
            $pairId,
            [double]$report.metrics.win_rate,
            $stopwatch.Elapsed.TotalSeconds,
            $state
        )
        $activePair = ""
        Write-LauncherStatus -State "between_pairs"
    }
    Write-Summary
    if (Test-Path -LiteralPath $snapshotPath -PathType Leaf) {
        & $python -m scripts.report_ppo_league_training_payoff --check *>> $launcherLog
    }
    else {
        & $python -m scripts.report_ppo_league_training_payoff *>> $launcherLog
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Training payoff snapshot aggregation failed."
    }
    Write-LauncherStatus -State "completed"
    Write-LauncherLog -Message "Generation 0 PFSP payoff snapshot completed."
}
catch {
    Write-Summary
    Write-LauncherStatus -State "failed" -ErrorMessage $_.Exception.Message
    Write-LauncherLog -Message "FAILED: $($_.Exception.Message)"
    exit 1
}

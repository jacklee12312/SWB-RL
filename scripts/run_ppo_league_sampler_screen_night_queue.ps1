param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\python.exe"
$reportRoot = Join-Path $projectRoot "data\reports\league_training\sampler_screen_20260804"
$planPath = Join-Path $reportRoot "night_queue_plan.json"
$statusPath = Join-Path $reportRoot "launcher_status.json"
$summaryPath = Join-Path $reportRoot "night_queue_summary.json"
$launcherLog = Join-Path $reportRoot "launcher.log"
$activeStage = ""
$activeJob = ""
$completedTraining = [System.Collections.Generic.List[string]]::new()
$completedCandidateEvaluations = [System.Collections.Generic.List[string]]::new()
$completedActiveMatrix = [System.Collections.Generic.List[string]]::new()
$completedArchiveBaseline = [System.Collections.Generic.List[string]]::new()

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Set-Location -LiteralPath $projectRoot

& $python -m scripts.report_ppo_league_sampler_screen_plan --check
if ($LASTEXITCODE -ne 0) {
    throw "Frozen sampler screen night queue plan is missing or stale."
}
$plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
if (-not [bool]$plan.immutable) {
    throw "Sampler screen plan must be immutable."
}

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

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
    [ordered]@{
        schema_version = 1
        state = $State
        launcher_pid = $PID
        active_stage = $activeStage
        active_job = $activeJob
        completed = [ordered]@{
            training = $completedTraining.Count
            candidate_evaluation = $completedCandidateEvaluations.Count
            generation_000_active_matrix = $completedActiveMatrix.Count
            archive_baseline = $completedArchiveBaseline.Count
        }
        expected = [ordered]@{
            training = [int]$plan.summary.training_job_count
            candidate_evaluation = [int]$plan.summary.candidate_evaluation_pair_count
            generation_000_active_matrix = [int]$plan.summary.missing_active_pair_count
            archive_baseline = [int]$plan.summary.archive_baseline_pair_count
        }
        total_additional_training_steps = [int]$plan.training.total_additional_agent_steps
        total_queued_evaluation_games = [int]$plan.summary.queued_evaluation_game_count
        data_partition = [string]$plan.data_partition
        updated_at = [DateTimeOffset]::Now.ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Validate-TrainingJob {
    param([Parameter(Mandatory = $true)]$Job)
    $checkpoint = Resolve-RepoPath ([string]$Job.checkpoint)
    $metricsPath = Resolve-RepoPath ([string]$Job.metrics)
    if (
        -not (Test-Path -LiteralPath $checkpoint -PathType Leaf) -or
        -not (Test-Path -LiteralPath $metricsPath -PathType Leaf)
    ) {
        throw "Training job $($Job.job_id) is missing checkpoint or metrics."
    }
    $metrics = Get-Content -Raw -LiteralPath $metricsPath | ConvertFrom-Json
    if (
        [int]$metrics.fork_master_seed -ne [int]$Job.training_seed -or
        [int]$metrics.starting_agent_steps -ne [int]$Job.parent_agent_steps -or
        [int]$metrics.requested_agent_steps -ne [int]$Job.target_agent_steps -or
        [int]$metrics.completed_agent_steps -lt [int]$Job.target_agent_steps -or
        [int]$metrics.trained_agent_steps -lt [int]$Job.additional_agent_steps
    ) {
        throw "Training job $($Job.job_id) has the wrong fork or step contract."
    }
    if (
        [string]$metrics.fork_metadata.parent_checkpoint_sha256 -ne
            [string]$Job.parent_checkpoint_sha256 -or
        [string]$metrics.league_diagnostics.external_opponent_manifest.file_sha256 -ne
            [string]$Job.opponent_manifest_sha256
    ) {
        throw "Training job $($Job.job_id) has the wrong parent or manifest hash."
    }
    if (
        -not [bool]$metrics.league_diagnostics.completed_without_exception -or
        [int]$metrics.league_diagnostics.truncated_episodes -ne 0 -or
        [int]$metrics.league_diagnostics.illegal_action_errors -ne 0 -or
        [int]$metrics.league_diagnostics.action_mask_mismatch_errors -ne 0
    ) {
        throw "Training job $($Job.job_id) failed safety validation."
    }
    return $metrics
}

function Invoke-TrainingJob {
    param([Parameter(Mandatory = $true)]$Job)
    $checkpoint = Resolve-RepoPath ([string]$Job.checkpoint)
    $metricsPath = Resolve-RepoPath ([string]$Job.metrics)
    $runLog = Resolve-RepoPath ([string]$Job.log)
    $parent = Resolve-RepoPath ([string]$Job.parent_checkpoint)
    $manifest = Resolve-RepoPath ([string]$Job.opponent_manifest)
    if (
        (Test-Path -LiteralPath $checkpoint -PathType Leaf) -and
        (Test-Path -LiteralPath $metricsPath -PathType Leaf)
    ) {
        $null = Validate-TrainingJob -Job $Job
        return "reused_valid_existing_result"
    }
    if (
        (Test-Path -LiteralPath $checkpoint) -or
        (Test-Path -LiteralPath $metricsPath)
    ) {
        throw "Refusing ambiguous partial training output for $($Job.job_id)."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $checkpoint) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $metricsPath) | Out-Null
    $arguments = @(
        "-m", "scripts.train_ppo",
        "--resume", $parent,
        "--fork-master-seed", "$($Job.training_seed)",
        "--resume-runtime-overrides",
        "--resume-opponent-pool-overrides",
        "--total-agent-steps", "$($Job.target_agent_steps)",
        "--rollout-workers", "$($plan.training.runtime.rollout_workers)",
        "--rollout-worker-threads", "$($plan.training.runtime.rollout_worker_torch_threads)",
        "--central-inference-batch-wait-ms", "$($plan.training.runtime.central_inference_batch_wait_ms)",
        "--device", "$($plan.training.runtime.device)",
        "--opponent-current-weight", "0",
        "--opponent-random-weight", "0",
        "--opponent-fixed-weight", "0",
        "--opponent-historical-weight", "0",
        "--opponent-external-manifest", $manifest,
        "--opponent-external-weight", "1",
        "--opponent-model-cache-size", "$($plan.training.runtime.opponent_model_cache_size)",
        "--opponent-model-cache-max-mib", "$($plan.training.runtime.opponent_model_cache_max_mib)",
        "--opponent-batching-mode", "$($plan.training.runtime.opponent_batching_mode)",
        "--opponent-snapshot-interval-steps", "250000",
        "--checkpoint", $checkpoint,
        "--checkpoint-interval-agent-steps", "$($plan.training.runtime.checkpoint_interval_agent_steps)",
        "--metrics-output", $metricsPath,
        "--monitor-system"
    )
    & $python @arguments *> $runLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training job $($Job.job_id) exited with code $LASTEXITCODE; see $runLog"
    }
    $null = Validate-TrainingJob -Job $Job
    return "completed"
}

function Validate-EvaluationJob {
    param([Parameter(Mandatory = $true)]$Job)
    $output = Resolve-RepoPath ([string]$Job.output)
    $focal = Resolve-RepoPath ([string]$Job.focal_checkpoint)
    if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw "Evaluation job $($Job.job_id) is missing its report."
    }
    $report = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
    if (
        [int]$report.metrics.games -ne [int]$Job.games -or
        [int]$report.metrics.terminated -ne [int]$Job.games -or
        [int]$report.metrics.truncated -ne 0 -or
        [int]$report.metrics.illegal_actions -ne 0 -or
        [int]$report.metrics.action_mask_mismatches -ne 0
    ) {
        throw "Evaluation job $($Job.job_id) failed game or safety validation."
    }
    $expectedFocalHash = [string]$Job.focal_checkpoint_sha256
    if ([string]::IsNullOrWhiteSpace($expectedFocalHash)) {
        $expectedFocalHash = Get-Sha256 -Path $focal
    }
    if (
        [string]$report.checkpoint.sha256 -ne $expectedFocalHash -or
        [string]$report.configuration.opponent_checkpoint_sha256 -ne
            [string]$Job.opponent_checkpoint_sha256
    ) {
        throw "Evaluation job $($Job.job_id) used the wrong checkpoint."
    }
    if (
        [int]$report.configuration.master_seed -ne [int]$Job.master_seed -or
        [int]$report.configuration.seed_count -ne [int]$Job.seed_count
    ) {
        throw "Evaluation job $($Job.job_id) used the wrong tuning seed."
    }
    return $report
}

function Invoke-EvaluationJob {
    param([Parameter(Mandatory = $true)]$Job)
    $output = Resolve-RepoPath ([string]$Job.output)
    $runLog = Resolve-RepoPath ([string]$Job.log)
    $focal = Resolve-RepoPath ([string]$Job.focal_checkpoint)
    $opponent = Resolve-RepoPath ([string]$Job.opponent_checkpoint)
    if (Test-Path -LiteralPath $output -PathType Leaf) {
        $null = Validate-EvaluationJob -Job $Job
        return "reused_valid_existing_report"
    }
    if (
        -not (Test-Path -LiteralPath $focal -PathType Leaf) -or
        -not (Test-Path -LiteralPath $opponent -PathType Leaf)
    ) {
        throw "Evaluation job $($Job.job_id) is missing a checkpoint."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $output) | Out-Null
    $arguments = @(
        "-m", "scripts.evaluate_ppo",
        $focal,
        "--seed-count", "$($Job.seed_count)",
        "--max-agent-steps", "512",
        "--master-seed", "$($Job.master_seed)",
        "--device", "cuda",
        "--classes", "1", "2", "3", "4", "5", "6", "7",
        "--full-matchup-matrix",
        "--opponent", "historical",
        "--opponent-checkpoint", $opponent,
        "--output", $output
    )
    & $python @arguments *> $runLog
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation job $($Job.job_id) exited with code $LASTEXITCODE; see $runLog"
    }
    $null = Validate-EvaluationJob -Job $Job
    return "completed"
}

function Run-EvaluationStage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)]$Jobs,
        [Parameter(Mandatory = $true)]$Completed
    )
    $script:activeStage = $Stage
    foreach ($job in $Jobs) {
        $script:activeJob = [string]$job.job_id
        Write-LauncherStatus -State "running"
        Write-LauncherLog -Message "Starting $Stage / $($job.job_id)."
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $state = Invoke-EvaluationJob -Job $job
        $stopwatch.Stop()
        $Completed.Add([string]$job.job_id)
        Write-LauncherLog -Message (
            "Completed {0} / {1}: elapsed={2:N1}s, state={3}." -f
            $Stage, $job.job_id, $stopwatch.Elapsed.TotalSeconds, $state
        )
        $script:activeJob = ""
        Write-LauncherStatus -State "between_jobs"
    }
}

if ($ValidateOnly) {
    Write-Output (
        "Validated queue plan: training={0}, evaluations={1}, games={2}." -f
        $plan.summary.training_job_count,
        $plan.summary.queued_evaluation_pair_count,
        $plan.summary.queued_evaluation_game_count
    )
    exit 0
}

try {
    Write-LauncherStatus -State "starting"
    Write-LauncherLog -Message (
        "Starting sampler screen night queue: training={0}, evaluation_pairs={1}, evaluation_games={2}." -f
        $plan.summary.training_job_count,
        $plan.summary.queued_evaluation_pair_count,
        $plan.summary.queued_evaluation_game_count
    )
    $activeStage = "training"
    foreach ($job in $plan.training.jobs) {
        $activeJob = [string]$job.job_id
        Write-LauncherStatus -State "running"
        Write-LauncherLog -Message "Starting training / $($job.job_id)."
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $state = Invoke-TrainingJob -Job $job
        $stopwatch.Stop()
        $completedTraining.Add([string]$job.job_id)
        Write-LauncherLog -Message (
            "Completed training / {0}: elapsed={1:N1}s, state={2}." -f
            $job.job_id, $stopwatch.Elapsed.TotalSeconds, $state
        )
        $activeJob = ""
        Write-LauncherStatus -State "between_jobs"
    }

    Run-EvaluationStage `
        -Stage "candidate_evaluation" `
        -Jobs $plan.candidate_evaluation.jobs `
        -Completed $completedCandidateEvaluations
    Run-EvaluationStage `
        -Stage "generation_000_active_matrix" `
        -Jobs $plan.generation_000_active_matrix.jobs `
        -Completed $completedActiveMatrix
    Run-EvaluationStage `
        -Stage "archive_baseline" `
        -Jobs $plan.archive_baseline.jobs `
        -Completed $completedArchiveBaseline

    [ordered]@{
        schema_version = 1
        report_kind = "ppo_league_sampler_screen_night_queue_summary"
        state = "completed"
        plan_sha256 = Get-Sha256 -Path $planPath
        completed_training_jobs = @($completedTraining)
        completed_candidate_evaluations = @($completedCandidateEvaluations)
        completed_generation_000_active_matrix = @($completedActiveMatrix)
        completed_archive_baseline = @($completedArchiveBaseline)
        completed_at = [DateTimeOffset]::Now.ToString("o")
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    $activeStage = ""
    $activeJob = ""
    Write-LauncherStatus -State "completed"
    Write-LauncherLog -Message "Sampler screen night queue completed."
}
catch {
    Write-LauncherStatus -State "failed" -ErrorMessage $_.Exception.Message
    Write-LauncherLog -Message "FAILED: $($_.Exception.Message)"
    exit 1
}

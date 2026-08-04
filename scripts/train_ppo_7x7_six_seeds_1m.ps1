Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\python.exe"
$reportRoot = Join-Path $projectRoot "data\reports\ppo_7x7_scaling_1m_expansion_20260803"
$checkpointRoot = Join-Path $projectRoot "data\checkpoints\ppo_7x7_scaling_20260801"
$statusPath = Join-Path $reportRoot "launcher_status.json"
$launcherLog = Join-Path $reportRoot "launcher.log"
$seeds = @("20260903", "20260904", "20260905", "20260906", "20260907", "20260908")
$completedSeeds = [System.Collections.Generic.List[string]]::new()
$activeSeed = ""

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Set-Location -LiteralPath $projectRoot

function Write-LauncherStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$ErrorMessage = ""
    )

    [ordered]@{
        schema_version = 1
        state = $State
        launcher_pid = $PID
        active_seed = $activeSeed
        completed_seeds = @($completedSeeds)
        queued_seeds = @(
            $seeds | Where-Object {
                $_ -notin $completedSeeds -and $_ -ne $activeSeed
            }
        )
        target_agent_steps_per_seed = 1000000
        runtime = [ordered]@{
            rollout_workers = 7
            rollout_worker_torch_threads = 2
            central_inference_batch_wait_seconds = 0.0005
        }
        updated_at = [DateTimeOffset]::Now.ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding utf8
}

try {
    Write-LauncherStatus -State "starting"
    Write-LauncherLog -Message "Starting serial six-seed 1M PPO queue."

    foreach ($seed in $seeds) {
        $activeSeed = $seed
        $seedRoot = Join-Path $checkpointRoot "seed_$seed"
        $targetCheckpoint = Join-Path $seedRoot "final.pt"
        $metricsOutput = Join-Path $reportRoot "seed_$seed.json"
        $runLog = Join-Path $reportRoot "seed_$seed.log"

        if (Test-Path -LiteralPath $targetCheckpoint) {
            throw "Refusing to overwrite existing checkpoint: $targetCheckpoint"
        }
        if (Test-Path -LiteralPath $metricsOutput) {
            throw "Refusing to overwrite existing metrics: $metricsOutput"
        }

        New-Item -ItemType Directory -Force -Path $seedRoot | Out-Null
        Write-LauncherStatus -State "running"
        Write-LauncherLog -Message "Starting seed $seed."
        $arguments = @(
            "-m", "scripts.train_ppo",
            "--total-agent-steps", "1000000",
            "--rollout-steps", "2048",
            "--rollout-workers", "7",
            "--rollout-worker-threads", "2",
            "--central-inference-batch-wait-ms", "0.5",
            "--max-episode-steps", "256",
            "--sequence-length", "32",
            "--minibatch-sequences", "8",
            "--update-epochs", "2",
            "--policy-architecture", "entity_action_v1",
            "--observation-version", "v4.1",
            "--hidden-size", "512",
            "--card-embedding-dim", "128",
            "--model-dim", "256",
            "--transformer-layers", "4",
            "--attention-heads", "8",
            "--feedforward-dim", "1024",
            "--learning-rate", "0.0001",
            "--entropy-coefficient", "0.01",
            "--clip-ratio", "0.2",
            "--master-seed", $seed,
            "--classes", "1", "2", "3", "4", "5", "6", "7",
            "--device", "cuda",
            "--match-setup", "official",
            "--opponent-current-weight", "1",
            "--opponent-random-weight", "0",
            "--opponent-fixed-weight", "0",
            "--opponent-historical-weight", "0.25",
            "--opponent-max-history", "4",
            "--opponent-snapshot-interval-steps", "250000",
            "--checkpoint", $targetCheckpoint,
            "--checkpoint-interval-agent-steps", "100000",
            "--metrics-output", $metricsOutput,
            "--monitor-system"
        )
        & $python @arguments *> $runLog
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Seed $seed exited with code $exitCode; see $runLog"
        }

        $completedSeeds.Add($seed)
        Write-LauncherLog -Message "Completed seed $seed."
        $activeSeed = ""
        Write-LauncherStatus -State "between_seeds"
    }

    Write-LauncherStatus -State "completed"
    Write-LauncherLog -Message "All six seeds completed."
}
catch {
    Write-LauncherStatus -State "failed" -ErrorMessage $_.Exception.Message
    Write-LauncherLog -Message "FAILED: $($_.Exception.Message)"
    exit 1
}

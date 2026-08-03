Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\python.exe"
$reportRoot = Join-Path $projectRoot "data\reports\ppo_7x7_scaling_3m_20260802"
$checkpointRoot = Join-Path $projectRoot "data\checkpoints\ppo_7x7_scaling_20260801"
$statusPath = Join-Path $reportRoot "launcher_status.json"
$launcherLog = Join-Path $reportRoot "launcher.log"
$seeds = @("20260831", "20260901", "20260902")
$completedSeeds = [System.Collections.Generic.List[string]]::new()

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Set-Location -LiteralPath $projectRoot

function Write-LauncherStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$ActiveSeed = "",
        [string]$ErrorMessage = ""
    )

    [ordered]@{
        schema_version = 1
        state = $State
        launcher_pid = $PID
        active_seed = $ActiveSeed
        completed_seeds = @($completedSeeds)
        queued_seeds = @($seeds | Where-Object { $_ -notin $completedSeeds })
        target_agent_steps = 3000000
        updated_at = [DateTimeOffset]::Now.ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding utf8
}

try {
    Write-LauncherStatus -State "starting"
    Write-LauncherLog -Message "Starting serial 1M-to-3M PPO queue."

    foreach ($seed in $seeds) {
        $seedRoot = Join-Path $checkpointRoot "seed_$seed"
        $sourceCheckpoint = Join-Path $seedRoot "final.pt"
        $targetCheckpoint = Join-Path $seedRoot "final_3m.pt"
        $metricsOutput = Join-Path $reportRoot "seed_$seed.json"
        $runLog = Join-Path $reportRoot "seed_$seed.log"

        if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
            throw "Source checkpoint not found: $sourceCheckpoint"
        }
        if (Test-Path -LiteralPath $targetCheckpoint) {
            throw "Refusing to overwrite existing 3M checkpoint: $targetCheckpoint"
        }

        Write-LauncherStatus -State "running" -ActiveSeed $seed
        Write-LauncherLog -Message "Starting seed $seed."
        $arguments = @(
            "-m", "scripts.train_ppo",
            "--resume", $sourceCheckpoint,
            "--total-agent-steps", "3000000",
            "--checkpoint", $targetCheckpoint,
            "--checkpoint-interval-agent-steps", "100000",
            "--metrics-output", $metricsOutput,
            "--device", "cuda"
        )
        & $python @arguments *> $runLog
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Seed $seed exited with code $exitCode; see $runLog"
        }

        $completedSeeds.Add($seed)
        Write-LauncherLog -Message "Completed seed $seed."
        Write-LauncherStatus -State "between_seeds"
    }

    Write-LauncherStatus -State "completed"
    Write-LauncherLog -Message "All three seeds completed."
}
catch {
    Write-LauncherStatus -State "failed" -ErrorMessage $_.Exception.Message
    Write-LauncherLog -Message "FAILED: $($_.Exception.Message)"
    exit 1
}

# 7x7 1M tactical case analysis

## Result

All three independently trained 1M-step checkpoints fail `TACT-SE-0001`.
Every checkpoint chooses to Super Evolve `天司长的继承者·圣德芬` instead of
the immediately attacking Storm follower `英雄幻视·托路` on an empty opposing
board.

| Seed | Earlier policy choices matching the recorded prefix | Preferred probability | Disfavored probability | Preferred-minus-disfavored logit | Result |
|---|---:|---:|---:|---:|---|
| 20260831 | 24/36 | 0.004764% | 99.995208% | -9.9517 | FAIL |
| 20260901 | 36/36 | 0.002044% | 99.997866% | -10.7981 | FAIL |
| 20260902 | 27/36 | 20.009260% | 79.869372% | -1.3842 | FAIL |

The prefix agreement is diagnostic only. Every checkpoint was teacher-forced
through the same 72 actions, and all three reproduced the exact target-state
SHA-256 `4bcf97fcfd855422a46e25d2360670fed9f16321a6fe4f6eaa99670e96d5ec65`.
The source checkpoint's 36/36 agreement additionally verifies exact recurrent
hidden-state reconstruction.

This is therefore a shared policy-quality weakness, not a replay mismatch,
action-mask error, or one-seed anomaly. Seed 20260902 is materially closer to
the annotated preference, so the behavior is learnable rather than structurally
unreachable.

## Scaling evidence and wall-clock estimate

The three 7x7 runs completed 1,000,495–1,002,052 agent steps at 78.168,
81.483, and 87.610 agent steps/s. They took 3.56, 3.41, and 3.18 hours;
median throughput was 81.483 agent steps/s. Each run completed only
11,526–12,492 episodes across 49 ordered class matchups, or roughly 235–255
episodes per matchup before accounting for historical-opponent sampling.

At the measured median throughput:

- another 1M agent steps takes about 3.41 hours per seed;
- resuming all three checkpoints from 1M to 3M takes about 6.82 hours per seed,
  or 20.45 GPU-hours sequentially;
- 3M total steps from initialization takes about 10.23 hours per seed;
- 10M total steps takes about 34.09 hours per seed.

Observed throughput varies with episode length, so these are planning estimates,
not runtime guarantees.

## Recommended next scaling slice

Keep the full 7x7 class matrix and update both seats. Resume the same three
seeds to 3M total agent steps as the next attributable scaling point, retaining
periodic checkpoints. Evaluate the frozen 7x7 matrix and tactical suite at
1.5M, 2M, 2.5M, and 3M so the trend is visible rather than judged only at the
endpoint.

Do not optimize directly against this one case yet. First grow the suite with
independent examples covering immediate damage, trades, lethal, resource use,
evolution targets, sequencing, and avoidable truncation; later reserve a held-out
subset. If the tactical margin remains flat through 3M while aggregate Elo rises,
that is evidence to test a controlled curriculum, search/imitation target, or
auxiliary preference objective rather than simply extending identical PPO.

Before the long resume, audit the horizon failures. Training truncation was
4.00–6.68%, while the 196-game frozen evaluations truncated 12.24–16.33%; the
Portal matchups account for much of the long-tail behavior. Raising the horizon
alone can hide looping behavior and changes the training contract, so first
separate legitimate long games from repeated low-value action loops. Any move
from 256 to 512 training steps should be an explicit controlled experiment.

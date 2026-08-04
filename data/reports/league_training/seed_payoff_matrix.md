# League Seed Payoff Matrix Audit

## Audit result

- Planned topology: 33/33 pairs.
- Games: 6468/6468.
- Terminated/truncated: 6468/0.
- Illegal actions / mask mismatches: 0 / 0.
- Draws: 2.

The six 1M models form a complete same-rule candidate matrix. The three
3M models are cross-rule historical anchors; their results against 1M
models must not be interpreted as a pure training-step ablation.

## Pair results

| Pair | Score | 95% CI | Relative Elo | Side 0 / Side 1 | Avg turn | Avg steps | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1m_20260903_vs_1m_20260904 | 53.57% | 46.59%–60.42% | +24.9 | 54.08% / 53.06% | 17.41 | 79.74 | CI overlaps 50%; no forced order |
| 1m_20260903_vs_1m_20260905 | 52.04% | 45.07%–58.93% | +14.2 | 50.00% / 54.08% | 17.09 | 77.32 | CI overlaps 50%; no forced order |
| 1m_20260903_vs_1m_20260906 | 48.47% | 41.57%–55.43% | -10.6 | 48.98% / 47.96% | 17.34 | 78.54 | CI overlaps 50%; no forced order |
| 1m_20260903_vs_1m_20260907 | 54.08% | 47.09%–60.91% | +28.4 | 50.00% / 58.16% | 17.30 | 81.27 | CI overlaps 50%; no forced order |
| 1m_20260903_vs_1m_20260908 | 58.16% | 51.17%–64.85% | +57.2 | 55.10% / 61.22% | 17.18 | 79.51 | direction supported by 95% CI |
| 1m_20260904_vs_1m_20260905 | 53.57% | 46.59%–60.42% | +24.9 | 51.02% / 56.12% | 16.67 | 74.44 | CI overlaps 50%; no forced order |
| 1m_20260904_vs_1m_20260906 | 51.53% | 44.57%–58.43% | +10.6 | 56.12% / 46.94% | 17.13 | 77.86 | CI overlaps 50%; no forced order |
| 1m_20260904_vs_1m_20260907 | 52.55% | 45.58%–59.43% | +17.7 | 58.16% / 46.94% | 16.27 | 76.24 | CI overlaps 50%; no forced order |
| 1m_20260904_vs_1m_20260908 | 52.04% | 45.07%–58.93% | +14.2 | 51.02% / 53.06% | 16.56 | 76.57 | CI overlaps 50%; no forced order |
| 1m_20260905_vs_1m_20260906 | 44.39% | 37.61%–51.38% | -39.2 | 44.90% / 43.88% | 17.07 | 77.45 | CI overlaps 50%; no forced order |
| 1m_20260905_vs_1m_20260907 | 47.96% | 41.07%–54.93% | -14.2 | 44.90% / 51.02% | 16.95 | 80.01 | CI overlaps 50%; no forced order |
| 1m_20260905_vs_1m_20260908 | 51.53% | 44.57%–58.43% | +10.6 | 47.96% / 55.10% | 16.93 | 77.86 | CI overlaps 50%; no forced order |
| 1m_20260906_vs_1m_20260907 | 45.92% | 39.09%–52.91% | -28.4 | 51.02% / 40.82% | 16.64 | 77.59 | CI overlaps 50%; no forced order |
| 1m_20260906_vs_1m_20260908 | 56.12% | 49.12%–62.89% | +42.8 | 63.27% / 48.98% | 17.47 | 82.35 | CI overlaps 50%; no forced order |
| 1m_20260907_vs_1m_20260908 | 52.55% | 45.58%–59.43% | +17.7 | 52.04% / 53.06% | 16.24 | 76.62 | CI overlaps 50%; no forced order |
| 3m_20260831_vs_1m_20260903 | 40.82% | 34.18%–47.81% | -64.5 | 40.82% / 40.82% | 18.67 | 87.00 | direction supported by 95% CI |
| 3m_20260831_vs_1m_20260904 | 45.41% | 38.59%–52.40% | -32.0 | 47.96% / 42.86% | 17.94 | 82.22 | CI overlaps 50%; no forced order |
| 3m_20260831_vs_1m_20260905 | 40.82% | 34.18%–47.81% | -64.5 | 41.84% / 39.80% | 18.44 | 84.02 | direction supported by 95% CI |
| 3m_20260831_vs_1m_20260906 | 44.39% | 37.61%–51.38% | -39.2 | 41.84% / 46.94% | 18.45 | 85.40 | CI overlaps 50%; no forced order |
| 3m_20260831_vs_1m_20260907 | 40.82% | 34.18%–47.81% | -64.5 | 41.84% / 39.80% | 18.02 | 83.34 | direction supported by 95% CI |
| 3m_20260831_vs_1m_20260908 | 45.41% | 38.59%–52.40% | -32.0 | 46.94% / 43.88% | 18.45 | 87.15 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260903 | 44.39% | 37.61%–51.38% | -39.2 | 45.92% / 42.86% | 18.85 | 87.10 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260904 | 44.90% | 38.10%–51.89% | -35.6 | 46.94% / 42.86% | 19.10 | 89.13 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260905 | 50.00% | 43.07%–56.93% | +0.0 | 44.90% / 55.10% | 18.01 | 81.76 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260906 | 49.74% | 42.82%–56.68% | -1.8 | 50.00% / 49.49% | 18.36 | 84.20 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260907 | 46.43% | 39.58%–53.41% | -24.9 | 45.92% / 46.94% | 18.30 | 86.03 | CI overlaps 50%; no forced order |
| 3m_20260901_vs_1m_20260908 | 49.49% | 42.57%–56.43% | -3.5 | 47.96% / 51.02% | 18.60 | 87.14 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260903 | 46.68% | 39.83%–53.66% | -23.1 | 47.96% / 45.41% | 17.11 | 79.90 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260904 | 55.10% | 48.11%–61.90% | +35.6 | 55.10% / 55.10% | 17.38 | 81.28 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260905 | 52.55% | 45.58%–59.43% | +17.7 | 53.06% / 52.04% | 17.12 | 78.37 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260906 | 51.02% | 44.07%–57.93% | +7.1 | 45.92% / 56.12% | 17.78 | 82.94 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260907 | 50.00% | 43.07%–56.93% | +0.0 | 47.96% / 52.04% | 17.10 | 80.62 | CI overlaps 50%; no forced order |
| 3m_20260902_vs_1m_20260908 | 56.12% | 49.12%–62.89% | +42.8 | 60.20% / 52.04% | 16.84 | 79.31 | CI overlaps 50%; no forced order |

## Six-candidate score matrix

Rows are focal models; diagonal is 50%.

| Model | seed_20260903_1m | seed_20260904_1m | seed_20260905_1m | seed_20260906_1m | seed_20260907_1m | seed_20260908_1m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed_20260903_1m | 50.00% | 53.57% | 52.04% | 48.47% | 54.08% | 58.16% |
| seed_20260904_1m | 46.43% | 50.00% | 53.57% | 51.53% | 52.55% | 52.04% |
| seed_20260905_1m | 47.96% | 46.43% | 50.00% | 44.39% | 47.96% | 51.53% |
| seed_20260906_1m | 51.53% | 48.47% | 55.61% | 50.00% | 45.92% | 56.12% |
| seed_20260907_1m | 45.92% | 47.45% | 52.04% | 54.08% | 50.00% | 52.55% |
| seed_20260908_1m | 41.84% | 47.96% | 48.47% | 43.88% | 47.45% | 50.00% |

## Candidate class aggregate

Each game is counted once from each model side, so class strength is not
tied to which checkpoint happened to be the report learner.

| Class ID | Games | Score | 95% CI |
| ---: | ---: | ---: | ---: |
| 1 | 840 | 44.88% | 41.55%–48.26% |
| 2 | 840 | 56.79% | 53.41%–60.10% |
| 3 | 840 | 44.29% | 40.96%–47.66% |
| 4 | 840 | 54.52% | 51.14%–57.86% |
| 5 | 840 | 42.14% | 38.85%–45.51% |
| 6 | 840 | 49.29% | 45.92%–52.66% |
| 7 | 840 | 58.10% | 54.73%–61.39% |

## Cycle and close-pair decision

- Point-estimate cycles (>50% each edge): 2.
- Preregistered strong cycles (>55% each edge): 0.
- Close pairs whose 95% CI includes 50%: 29.
- Generation 0 retains every candidate and anchor, so none of the close
  196-game screens is used to alter pool membership. A future removal
  decision requires a 980-game confirmation first.

# League Meta-game Report

## Result

The deterministic zero-sum solve uses only the complete six-model,
same-rule 1M candidate submatrix. Old 3M checkpoints remain anchors.

- Uniform worst expected payoff: -0.054422.
- Uniform exploitability proxy: 0.054422.
- Nash worst expected payoff: -0.000000.
- Nash exploitability proxy: 0.000000.
- Nash effective population size: 2.635.

## Mixture weights

| Model | Uniform | Nash | Nash bootstrap 95% CI | Support frequency | Payoff vs Nash |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed_20260903_1m | 16.67% | 42.11% | 0.00%–100.00% | 77.65% | +0.000000 |
| seed_20260904_1m | 16.67% | 0.00% | 0.00%–100.00% | 40.50% | -0.009130 |
| seed_20260905_1m | 16.67% | 0.00% | 0.00%–41.67% | 12.65% | -0.070892 |
| seed_20260906_1m | 16.67% | 42.11% | 0.00%–74.65% | 50.25% | +0.000000 |
| seed_20260907_1m | 16.67% | 15.79% | 0.00%–100.00% | 41.35% | -0.000000 |
| seed_20260908_1m | 16.67% | 0.00% | 0.00%–36.84% | 11.00% | -0.128357 |

## Diagnostics

- Matrix warnings: none.
- Global point cycles: 2.
- Global >55% cycles: 0.
- Class point cycles: 3.
- Class >55% cycles: 1.
- The exploitability number is only an internal-population proxy; it
  cannot certify robustness against a newly trained best response.

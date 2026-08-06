# PPO 训练速度稳定性报告

所有运行使用同一只读 3M checkpoint；每轮请求 20,480 agent steps，并排除前 2 次 update。GPU/CPU/内存每 0.5 秒采样。

| 配置 | 次数 | 稳态 steps/s | batch mean | forward ms/step | GPU 峰值 MiB | 截断 |
|---|---:|---:|---:|---:|---:|---:|
| 6 workers / 1.0ms 初始 | 3 | 58.27 (56.93–121.33) | 1.77 | 12.31 | 14015 | 0 |
| 6 workers / 1.0ms 回切 | 1 | 121.80 (121.80–121.80) | 1.84 | 5.10 | 11440 | 0 |
| 7 workers / 1.0ms | 3 | 129.35 (127.99–130.21) | 1.91 | 4.85 | 14287 | 0 |
| 7 workers / 0.5ms（采用） | 3 | 137.14 (135.52–138.57) | 1.71 | 5.32 | 14290 | 0 |
| 7 workers / 0.25ms | 1 | 135.27 (135.27–135.27) | 1.69 | 5.41 | 14285 | 0 |
| 8 workers / 1.0ms | 1 | 131.23 (131.23–131.23) | 1.99 | 4.70 | 15779 | 0 |

## 结论

- 同配置 6-worker 初始三次最大相差 2.13×；中央 forward 与 GPU 时钟/功耗状态同步变化，seed 不是原因。
- 保守快态 6-worker 参考为 121.57 steps/s。
- 最终候选中位数相对该参考提升 12.8%，三次波动 1.022×。
- 最终三次共 644 局、截断 0、checkpoint 前后哈希不变、无硬件 throttle。

## 采用配置

```text
rollout_workers = 7
rollout_worker_torch_threads = 2
central_inference_batch_wait_seconds = 0.0005
```

决策门：通过。

## 拒绝或受阻候选

- 8 workers, 1.0 ms：Only one screen and insufficient incremental gain; GPU peak 15779 MiB left 597 MiB headroom.
- 7 workers, 0.25 ms：Screen was slower than the three-run 0.5 ms median (135.27 vs 137.14 steps/s).
- lock GPU clocks to 2520-2820 MHz：nvidia-smi rejected the reversible clock-lock probe: current user does not have permission; no clock state was changed.

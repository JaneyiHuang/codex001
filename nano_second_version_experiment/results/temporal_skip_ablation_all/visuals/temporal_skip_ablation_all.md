# Temporal Skip Ablation

| Method | Delay | Reward | Drop | Offload | Nano call reduction | RTT/step | Nano infer/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| No-skip | 195.82 | -0.979099 | 0.210417 | 0.770875 | 0.00% | 47.8686 | 1.32167 |
| Safety-aware | 195.997 | -0.979985 | 0.212625 | 0.769625 | 20.55% | 28.0037 | 0.909806 |
| Relaxed safety | 197.373 | -0.986866 | 0.20425 | 0.783667 | 53.08% | 18.2294 | 0.503439 |
| No-safety | 197.185 | -0.985927 | 0.207875 | 0.779458 | 70.33% | 12.1973 | 0.41061 |

Safety conditions:

- No-skip: temporal repeat is disabled; every step calls Nano.
- Safety-aware: energy<=0.12, queue>=0.80, obs-change>=0.35, channel-drop>=0.35 interrupt repeat.
- Relaxed safety: energy<=0.08, queue>=0.90, obs-change>=0.50, channel-drop>=0.50 interrupt repeat.
- No-safety: safety interrupts disabled.

RTT and Nano inference overhead are reported separately and are not added to simulated MEC delay.

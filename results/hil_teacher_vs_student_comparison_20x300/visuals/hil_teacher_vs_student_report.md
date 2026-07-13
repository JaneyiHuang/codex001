# Nano-HIL Student vs Nano-HIL Teacher

| Metric | Student HIL | Teacher HIL | Student - Teacher | Relative diff |
|---|---:|---:|---:|---:|
| Average reward | -0.979099 | -0.97883 | -0.00026875 | -0.0275% |
| Average delay | 195.82 | 195.766 | 0.05375 | 0.0275% |
| Drop rate | 0.210417 | 0.210708 | -0.000291667 | -0.1384% |
| Offload rate | 0.770875 | 0.770167 | 0.000708333 | 0.0920% |
| Agreement with teacher | 0.9985 | 1 | -0.00150001 | -0.1500% |

## Deployment Overhead

| Metric | Student HIL | Teacher HIL |
|---|---:|---:|
| Nano pure infer ms | 1.32167 | 1.40202 |
| PC-Nano round trip ms | 47.8686 | 43.5924 |

Note: round-trip latency is not added to simulated MEC delay.

# PC Student vs Nano HIL Student Comparison

| Metric | PC all-student | Nano HIL | HIL - PC | Relative diff |
|---|---:|---:|---:|---:|
| Average reward | -0.979099 | -0.979099 | 0 | 0.0000% |
| Average delay | 195.82 | 195.82 | 0 | 0.0000% |
| Drop rate | 0.210417 | 0.210417 | 0 | 0.0000% |
| Offload rate | 0.770875 | 0.770875 | 0 | 0.0000% |
| Teacher agreement | 0.9985 | 0.9985 | 0 | 0.0000% |

## Deployment Overhead

| Metric | Value |
|---|---:|
| PC student batch infer ms | 0.19752 |
| Nano ONNX pure infer ms | 1.32167 |
| PC-Nano round-trip ms | 47.8686 |

## Step-Level Consistency

| Metric | Value |
|---|---:|
| Total aligned steps | 6000 |
| Joint action match rate | 1 |
| Target action match rate | 1 |
| Delay match rate | 1 |
| Max absolute delay diff | 0 |

Note: round-trip latency is deployment communication overhead and is not added to simulated MEC delay.

# No-Skip HIL vs Mixed Temporal-Skip HIL

| Metric | No skip | Mixed skip | Skip - No skip | Relative diff |
|---|---:|---:|---:|---:|
| Average reward | -0.979099 | -0.979985 | -0.000886042 | -0.0905% |
| Average delay | 195.82 | 195.997 | 0.177208 | 0.0905% |
| Drop rate | 0.210417 | 0.212625 | 0.00220833 | 1.0495% |
| Offload rate | 0.770875 | 0.769625 | -0.00125 | -0.1622% |

## Temporal Skip Calls

| Metric | Value |
|---|---:|
| Total env steps | 6000 |
| Nano model calls | 4767 |
| Nano call reduction rate | 20.5500% |
| PC repeat model calls | 4754 |
| PC repeat call reduction rate | 20.7667% |
| Nano safety interrupts | 4121 |
| PC repeat safety interrupts | 4104 |

## Deployment Overhead

| Metric | No skip | Mixed skip |
|---|---:|---:|
| Nano infer ms per env step | 1.32167 | 0.909806 |
| Round-trip ms per env step | 47.8686 | 28.0037 |

Note: deployment overhead is reported separately and is not added to simulated MEC delay.

# No-Skip HIL vs Mixed Temporal-Skip HIL

| Metric | No skip | Mixed skip | Skip - No skip | Relative diff |
|---|---:|---:|---:|---:|
| Average reward | -0.979099 | -0.986866 | -0.00776729 | -0.7933% |
| Average delay | 195.82 | 197.373 | 1.55346 | 0.7933% |
| Drop rate | 0.210417 | 0.20425 | -0.00616667 | -2.9307% |
| Offload rate | 0.770875 | 0.783667 | 0.0127917 | 1.6594% |

## Temporal Skip Calls

| Metric | Value |
|---|---:|
| Total env steps | 6000 |
| Nano model calls | 2815 |
| Nano call reduction rate | 53.0833% |
| PC repeat model calls | 2838 |
| PC repeat call reduction rate | 52.7000% |
| Nano safety interrupts | 1713 |
| PC repeat safety interrupts | 1727 |

## Deployment Overhead

| Metric | No skip | Mixed skip |
|---|---:|---:|
| Nano infer ms per env step | 1.32167 | 0.503439 |
| Round-trip ms per env step | 47.8686 | 18.2294 |

Note: deployment overhead is reported separately and is not added to simulated MEC delay.

# No-Skip HIL vs Mixed Temporal-Skip HIL

| Metric | No skip | Mixed skip | Skip - No skip | Relative diff |
|---|---:|---:|---:|---:|
| Average reward | -0.979099 | -0.985927 | -0.00682833 | -0.6974% |
| Average delay | 195.82 | 197.185 | 1.36567 | 0.6974% |
| Drop rate | 0.210417 | 0.207875 | -0.00254167 | -1.2079% |
| Offload rate | 0.770875 | 0.779458 | 0.00858333 | 1.1135% |

## Temporal Skip Calls

| Metric | Value |
|---|---:|
| Total env steps | 6000 |
| Nano model calls | 1780 |
| Nano call reduction rate | 70.3333% |
| PC repeat model calls | 1792 |
| PC repeat call reduction rate | 70.1333% |
| Nano safety interrupts | 0 |
| PC repeat safety interrupts | 0 |

## Deployment Overhead

| Metric | No skip | Mixed skip |
|---|---:|---:|
| Nano infer ms per env step | 1.32167 | 0.41061 |
| Round-trip ms per env step | 47.8686 | 12.1973 |

Note: deployment overhead is reported separately and is not added to simulated MEC delay.

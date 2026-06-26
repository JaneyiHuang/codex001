# Temporal Distillation Experiments

This folder is isolated from the original project scripts and result directory.
Default outputs are written under `temporal/results/`.

## Run Order

Generate an isolated pruned student actor:

```powershell
python temporal\prune_mappo_actor.py --pruning-percentage 0.25
```

Train the temporal-distilled student:

```powershell
python temporal\temporal_distill_mappo_actor.py --repeat-scale 5 --max-repeat 5
```

Run a quick comparison:

```powershell
python temporal\compare_experiments.py --episodes 1
```

Run a formal fixed-scenario comparison:

```powershell
python temporal\compare_experiments.py --episodes 50
```

Run a load-sweep comparison:

```powershell
python temporal\compare_experiments.py --mode loads --episodes 50 --load-points 13
```

Redraw the cross-safety summary figures after running the default, no-safety, and
relaxed-safety comparisons:

```powershell
python temporal\draw_temporal_safety_comparison.py
```

## Safety Interrupt

Temporal policies reuse cached actions until their repeat counter expires. A cached action
is interrupted early when safety checks detect low energy, high queue pressure, large
critical-state changes, or a sudden channel drop while offloading.

Important knobs:

```powershell
--safe-energy-min 0.12
--safe-queue-threshold 0.80
--safe-obs-change-threshold 0.35
--safe-channel-drop-threshold 0.35
--disable-safety
```

## Main New Metrics

- `decision_count_mean`: average per-agent decisions made by the temporal actor.
- `agent_temporal_compression_ratio_mean`: original per-agent decisions divided by temporal decisions.
- `inference_call_count_mean`: average slot-level actor forward calls.
- `inference_temporal_compression_ratio_mean`: original slot-level calls divided by temporal calls.
- `safety_interrupt_count_mean`: average early repeat cancellations from safety checks.

## Clearer Figures

`temporal\compare_experiments.py` saves two fixed-scenario dashboards:

- `comparison_dashboard.png`: all selected policies, shown as gaps or savings.
- `compressed_methods_dashboard.png`: MAPPO, pruned, distilled, and temporal only.

`temporal\draw_temporal_safety_comparison.py` saves cross-safety figures under
`temporal\results\safety_comparison\`:

- `temporal_safety_performance_gaps.png`
- `temporal_safety_compression_savings.png`
- `temporal_safety_tradeoff.png`
- `temporal_safety_summary.csv`
- `temporal_safety_summary.md`

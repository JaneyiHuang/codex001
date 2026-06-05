# QMIX baseline

This folder contains a QMIX baseline for the MEC multi-agent offloading environment.

Run training from the project root:

```powershell
python -m QMIX.train_qmix --episodes 2000
```

Main outputs are saved under `QMIX/results/`:

- `qmix_checkpoint.pt`: best checkpoint by recent reward moving average
- `qmix_last.pt`: final checkpoint
- `qmix_training_log.csv`: training log
- `reward_curve.png`, `delay_curve.png`, `drop_rate_curve.png`, `loss_curve.png`

After training, include it in comparison experiments:

```powershell
python compare_experiments.py --experiments mappo,qmix --qmix-model-path QMIX/results/qmix_checkpoint.pt
```

Draw MAPPO/QMIX training-curve comparisons:

```powershell
python -m QMIX.draw_mappo_qmix_training --window 50
```

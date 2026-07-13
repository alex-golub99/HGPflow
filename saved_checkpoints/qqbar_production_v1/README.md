# qqbar production v1 (July 2026)

HGPflow trained on ~305k self-generated COCOA qqbar events (8.55M segments,
305 train / 10 val files, `/pscratch/sd/a/agolub/qqbar_events/`).

## Checkpoints

| file | contents | val loss |
|---|---|---|
| `checkpoints/stage1_epoch24_val0.0943.ckpt` | stage 1 (incidence + indicator) | 0.0943 |
| `checkpoints/stage2full_epoch5_val0.0553.ckpt` | **full model** (frozen stage 1 + hyperedge corrections) — use this for inference | 0.0553 |

## Recipe

- **Stage 1**: 4 nodes x 4 GPUs DDP, global batch 2048 (128/rank), lr 6.0e-4
  (sqrt-scaled from 1e-4@128), 25 epochs, cosine to 5e-6. Length-grouped
  batching + per-rank file sharding (`train_config_stage1.yml`).
- **Stage-1 inference** for stage-2 data: `ind_threshold_loose: 0.2`.
- **Stage 2**: single GPU, batch 512, lr 1.0e-4, 6 epochs,
  **`fix_candidate_context: true`** (`train_config_stage2.yml`). A controlled A/B
  vs the published loader semantics (misaligned per-segment context, ~20% of
  candidates used) showed the fix cuts neutral energy bias ~4x:
  neutral pT residual median +0.017 (fixed) vs +0.068 (paper), IQR 0.490 vs 0.538.
  A longer 4-GPU run (18 epochs, global 2048, lr 2e-4, shuffled) converged
  slightly WORSE (median +0.035) - the small model saturates; keep this recipe.
- **Operating threshold**: `ind_threshold ~ 0.4` (energy-weighted neutral miss
  ~1.6%, zero above 50 GeV; see `quantify_missed_neutrals.py`).

## Inference

```
python -m hgpflow_v2.eval -i configs/cocoa/inference_fullval.yml   # uses these configs/ckpt
python evaluate.py -p '<...>/inference/fullval/pred_*_merged.root' -t 0.4 -o eval_fullval
```

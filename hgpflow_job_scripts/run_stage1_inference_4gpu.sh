#!/bin/bash
# Stage-1 inference split across the 4 GPUs of one node.
# Worker i pins GPU i and processes files sorted(glob)[i::4] -- disjoint shards,
# separate output files per input, so no collisions. Run inside a 4-GPU
# interactive allocation (plain background processes; no srun needed).
#
# Usage:  ./hgpflow_job_scripts/run_stage1_inference_4gpu.sh
# Watch:  tail -f job_logs/stage1_inf_gpu0.log

set -u
cd /global/u1/a/agolub/HGPflow
source .venv/bin/activate
mkdir -p job_logs

RUN_DIR=/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largeqqbartraining4nodev1epoch25
CKPT=$RUN_DIR/checkpoints/epoch=24-val_total_loss=0.0943.ckpt
TRAIN_GLOB="/pscratch/sd/a/agolub/qqbar_events/split_train/*.root"
VAL_GLOB="/pscratch/sd/a/agolub/qqbar_events/split_val/*.root"

# ── pre-flight: remove corrupt/partial outputs from a previous killed session ──
python - << EOF
import glob, os, uproot
for d in ['qqbar_train', 'qqbar_val']:
    for p in glob.glob('$RUN_DIR/inference/' + d + '/stage1_inference_*.root'):
        try:
            with uproot.open(p) as f:
                _ = f['event_tree'].num_entries
        except Exception:
            print('removing corrupt/partial output:', p)
            os.remove(p)
print('valid existing outputs:',
      len(glob.glob('$RUN_DIR/inference/*/stage1_inference_*.root')), '(will be skipped)')
EOF

# ── generate one config per GPU ──────────────────────────────
CFG_DIR=$(mktemp -d)
for i in 0 1 2 3; do
cat > "$CFG_DIR/stage1_inf_gpu$i.yml" << EOF
init:
  detector: COCOA

  gpu: $i
  chunk_size: 256
  batch_size: 256
  num_workers: 8
  precision: medium

  model:
    config_path_v   : $RUN_DIR/config_v.yml
    config_path_ms1 : $RUN_DIR/config_ms1.yml
    config_path_t   : $RUN_DIR/config_t.yml
    checkpoint_path : $CKPT

  ind_threshold_loose: 0.2

items:
  # skip-existing: files whose output is already present are filtered out, so
  # relaunching after a killed session only processes the remainder
  - info: "qqbar train shard $i/4"
    seg_path: |
      [f for f in sorted(glob.glob('$TRAIN_GLOB'))[$i::4] if not os.path.exists('$RUN_DIR/inference/qqbar_train/stage1_inference_' + os.path.basename(f))]
    dir_flag: 'qqbar_train'
    output_filepath: null
    n_events: -1

  - info: "qqbar val shard $i/4"
    seg_path: |
      [f for f in sorted(glob.glob('$VAL_GLOB'))[$i::4] if not os.path.exists('$RUN_DIR/inference/qqbar_val/stage1_inference_' + os.path.basename(f))]
    dir_flag: 'qqbar_val'
    output_filepath: null
    n_events: -1
EOF
done

# ── launch the 4 workers ─────────────────────────────────────
START=$(date +%s)
declare -a PIDS
for i in 0 1 2 3; do
    python -m hgpflow_v2.hyperedge_data_prep -i "$CFG_DIR/stage1_inf_gpu$i.yml" \
        > "job_logs/stage1_inf_gpu$i.log" 2>&1 &
    PIDS[$i]=$!
    echo "GPU $i: PID ${PIDS[$i]}  log: job_logs/stage1_inf_gpu$i.log"
done

FAIL=0
for i in 0 1 2 3; do
    if ! wait "${PIDS[$i]}"; then
        echo "GPU $i worker FAILED (see job_logs/stage1_inf_gpu$i.log)"
        FAIL=1
    fi
done
echo "wall time: $(( ($(date +%s) - START) / 60 )) min"

# ── completeness check: one output per input file ────────────
N_IN=$(ls $TRAIN_GLOB | wc -l)
N_OUT=$(ls "$RUN_DIR/inference/qqbar_train/"stage1_inference_*.root 2>/dev/null | wc -l)
NV_IN=$(ls $VAL_GLOB | wc -l)
NV_OUT=$(ls "$RUN_DIR/inference/qqbar_val/"stage1_inference_*.root 2>/dev/null | wc -l)
echo "train outputs: $N_OUT / $N_IN    val outputs: $NV_OUT / $NV_IN"

if [ $FAIL -eq 0 ] && [ "$N_OUT" -eq "$N_IN" ] && [ "$NV_OUT" -eq "$NV_IN" ]; then
    echo "ALL DONE - stage-2 training inputs are ready in $RUN_DIR/inference/"
else
    echo "INCOMPLETE - check worker logs before starting stage 2"
fi

#!/bin/bash
# 4-node DDP scaling experiment: 16 ranks, global batch 2048, lr 6e-4, 10 epochs.
# Runs alongside the 1-node production job (separate config + EXP_KEY, no collisions).

#SBATCH --job-name=hgpflow_4node
#SBATCH --account=m3246
#SBATCH --constraint=gpu
#SBATCH --nodes=4                # must match num_nodes in the 4node config (guard below)
#SBATCH --ntasks-per-node=4      # one task per GPU (DDP rank)
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16       # 64 cores / 4 ranks
#SBATCH --qos=regular
#SBATCH -t 06:00:00              # ~9 min/epoch x 10 epochs + load + margin
#SBATCH --output=job_logs/training4node_%j.out
#SBATCH --error=job_logs/training4node_%j.err

# ── Setup ────────────────────────────────────────────────────
cd /global/u1/a/agolub/HGPflow
source .venv/bin/activate
mkdir -p job_logs

# Slurm >= 22.05: srun does NOT inherit --cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# ── Config consistency guard ─────────────────────────────────
CFG=configs/cocoa/train_incidence_4node.yml
CFG_DEVICES=$(grep -E '^num_devices:' $CFG | awk '{print $2}')
CFG_NODES=$(grep -E '^num_nodes:' $CFG | awk '{print $2}')
if [ "$CFG_NODES" != "$SLURM_JOB_NUM_NODES" ] || [ "$CFG_DEVICES" != "4" ]; then
    echo "FATAL: config ($CFG_DEVICES devices x $CFG_NODES nodes) != allocation (4 x $SLURM_JOB_NUM_NODES)"
    echo "Fix num_devices/num_nodes in $CFG or the #SBATCH directives."
    exit 1
fi

# Own experiment key: distinct run dir + Comet run from the 1-node production chain.
EXP_KEY=largeqqbartraining4nodev1epoch25

# Auto-resume within this experiment's own chain only.
RUN_ROOT="$SCRATCH/hgpflow_runs"
CKPT=$(find "$RUN_ROOT" -name 'last.ckpt' -path "*$EXP_KEY*" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
RESUME=""
if [ -n "$CKPT" ]; then RESUME="-resume $CKPT"; echo "Resuming from $CKPT"; fi

# ── Run training ─────────────────────────────────────────────
srun python -m hgpflow_v2.train \
    -cv  configs/cocoa/model_configs/var_mini.yml \
    -cms1 configs/cocoa/model_configs/model_stage1_mini.yml \
    -ct  $CFG \
    -ekey $EXP_KEY $RESUME

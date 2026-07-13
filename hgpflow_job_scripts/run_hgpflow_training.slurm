#!/bin/bash

#SBATCH --job-name=hgpflow_training
#SBATCH --account=m3246
#SBATCH --constraint=gpu
#SBATCH --nodes=1                # multi-node: raise this AND num_nodes in train_incidence.yml
#SBATCH --ntasks-per-node=4      # one task per GPU (DDP rank)
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16       # 64 cores / 4 ranks
#SBATCH --qos=regular
#SBATCH -t 12:00:00              # QOS max; minimizes number of chained jobs
#SBATCH --output=job_logs/training_%j.out
#SBATCH --error=job_logs/training_%j.err

# ── Setup ────────────────────────────────────────────────────
cd /global/u1/a/agolub/HGPflow
source .venv/bin/activate
mkdir -p job_logs

# Slurm >= 22.05: srun does NOT inherit --cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# ── Config consistency guard ─────────────────────────────────
# Lightning sizes the DDP world from the config (num_devices x num_nodes) while
# SLURM launches SLURM_NTASKS tasks; if they disagree, ranks hang waiting for peers.
CFG=configs/cocoa/train_incidence.yml
CFG_DEVICES=$(grep -E '^num_devices:' $CFG | awk '{print $2}')
CFG_NODES=$(grep -E '^num_nodes:' $CFG | awk '{print $2}')
if [ "$CFG_NODES" != "$SLURM_JOB_NUM_NODES" ] || [ "$CFG_DEVICES" != "4" ]; then
    echo "FATAL: config ($CFG_DEVICES devices x $CFG_NODES nodes) != allocation (4 x $SLURM_JOB_NUM_NODES)"
    echo "Fix num_devices/num_nodes in $CFG or the #SBATCH directives."
    exit 1
fi

# Fixed experiment key so every job in the chain shares one run dir + Comet run.
EXP_KEY=largeqqbartrainingddpv2

# Auto-resume: if a checkpoint from a previous job exists, resume from it.
RUN_ROOT="$SCRATCH/hgpflow_runs"
CKPT=$(find "$RUN_ROOT" -name 'last.ckpt' -path "*$EXP_KEY*" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
RESUME=""
if [ -n "$CKPT" ]; then RESUME="-resume $CKPT"; echo "Resuming from $CKPT"; fi

# ── Run training ─────────────────────────────────────────────
# (no -g flag: GPU assignment is per-rank under DDP, handled by Lightning/SLURM)
srun python -m hgpflow_v2.train \
    -cv  configs/cocoa/model_configs/var_mini.yml \
    -cms1 configs/cocoa/model_configs/model_stage1_mini.yml \
    -ct  configs/cocoa/train_incidence.yml \
    -ekey $EXP_KEY $RESUME

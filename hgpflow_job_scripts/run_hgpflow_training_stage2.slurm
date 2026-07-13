#!/bin/bash

#SBATCH --job-name=hgpflow_training_stage2
#SBATCH --account=m3246
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --qos=regular
#SBATCH -t 08:00:00
#SBATCH --output=job_logs/training_st2_%j.out
#SBATCH --error=job_logs/training_st2_%j.err

# ── Setup ────────────────────────────────────────────────────
cd /global/u1/a/agolub/HGPflow
source .venv/bin/activate
mkdir -p job_logs

# ── Run training ─────────────────────────────────────────────
srun python -m hgpflow_v2.train -cms2 configs/cocoa/model_configs/model_stage2.yml -ct configs/cocoa/train_hyperedge.yml

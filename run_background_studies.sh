#!/bin/bash
# (a) fake rate and (b) feature migration, for the LLP background question.
# Run inside an interactive CPU allocation:
#   salloc --nodes=1 --qos=interactive --time=02:00:00 --constraint=cpu --account=m3246
#   ./run_background_studies.sh
set -u
cd /global/u1/a/agolub/HGPflow
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
PY=.venv/bin/python
S=notebooks/cocoa/llp_studies
R=$S/results
mkdir -p $R

QRUN=/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largeqqbartraining4nodev1epoch25/inference
HRUN=/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largehsstraining4node25epoch/inference
PAPER=/pscratch/sd/a/agolub/hgpflow_paper_files
HSS_TRUTH=/pscratch/sd/a/agolub/hss_events/HSS_events_with_ppflow/cocoa_hss_pflow_30k.root

echo "############ (a) FAKE RATE ############"

# QCD benchmarks -- qqbar-trained only (no hss inference exists on these yet)
$PY $S/fake_rate_by_sample.py --nprocs 32 \
    --truth $PAPER/dijet_test.root --pred "$QRUN/dijet_test/pred_*_merged.root" \
    --label "dijet | qqbar-trained" 2>&1 | tee $R/fake_dijet_qqbar.txt

$PY $S/fake_rate_by_sample.py --nprocs 32 \
    --truth $PAPER/ttbar.root --pred "$QRUN/ttbar/pred_*_merged.root" \
    --label "ttbar | qqbar-trained" 2>&1 | tee $R/fake_ttbar_qqbar.txt

# HSS sample -- BOTH models (the only sample where both exist)
$PY $S/fake_rate_by_sample.py --nprocs 32 \
    --truth $HSS_TRUTH --pred "$HRUN/hss_test/pred_*_merged.root" \
    --label "hss | hss-trained" 2>&1 | tee $R/fake_hss_hss.txt

$PY $S/fake_rate_by_sample.py --nprocs 32 \
    --truth $HSS_TRUTH --pred "$QRUN/hss_ppflow/pred_*_merged.root" \
    --label "hss | qqbar-trained" 2>&1 | tee $R/fake_hss_qqbar.txt

echo "############ (b) FEATURE MIGRATION ############"

$PY $S/feature_migration.py --nprocs 32 \
    --truth $PAPER/dijet_test.root --pred "$QRUN/dijet_test/pred_*_merged.root" \
    --label "dijet | qqbar-trained" --dump $R/feat_dijet_qqbar.npz 2>&1 | tee $R/feat_dijet_qqbar.txt

$PY $S/feature_migration.py --nprocs 32 \
    --truth $PAPER/ttbar.root --pred "$QRUN/ttbar/pred_*_merged.root" \
    --label "ttbar | qqbar-trained" --dump $R/feat_ttbar_qqbar.npz 2>&1 | tee $R/feat_ttbar_qqbar.txt

$PY $S/feature_migration.py --nprocs 32 \
    --truth $HSS_TRUTH --pred "$HRUN/hss_test/pred_*_merged.root" \
    --label "hss | hss-trained" --dump $R/feat_hss_hss.npz 2>&1 | tee $R/feat_hss_hss.txt

$PY $S/feature_migration.py --nprocs 32 \
    --truth $HSS_TRUTH --pred "$QRUN/hss_ppflow/pred_*_merged.root" \
    --label "hss | qqbar-trained" --dump $R/feat_hss_qqbar.npz 2>&1 | tee $R/feat_hss_qqbar.txt

echo "############ DONE -- results in $R ############"
ls -la $R/fake_*.txt $R/feat_*.txt

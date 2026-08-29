"""The bridge measurement: the REAL (model-output) matching efficiency split by decay region,
with the region-appropriate reference axis. LLP_jet_findings.md follow-up to section 10.

Everything in sections 6.4-10 of the findings doc is truth-level idealisation; the actual
section-2 efficiency (~0.44 plateau) was never decomposed. This runs the real evaluation
pipeline (PerformanceCOCOA on HGPflow predictions) and reports, per decay region, efficiency
against the momentum axis AND against the A0 flight direction.

Truth-level predictions to test against (section 10): tracker-region jets ~30% on the momentum
axis (rising with track coverage); calo-region jets 28% -> 47% when switching to the flight
direction; beyond-calo jets ~0 on both.

UNLIKE the other scripts in this directory, this one needs the hgpflow_v2 package and the
model's merged prediction files, and takes ~10-30 min for the 30k test set.

Run:
  PYTHONPATH=<repo root> python region_efficiency_reco.py \
      [--truth .../cocoa_hss_pflow_30k.root] [--pred '.../hss_test/pred_*_merged.root'] \
      [--nprocs 8] [--comps hgpflow ppflow]
"""
import argparse

from hgpflow_v2.performance.performance import PerformanceCOCOA
from hgpflow_v2.performance.llp_helper import (
    tag_truth_jets_llp, tag_truth_jets_decay_region, llp_efficiency_by_region)

DEFAULT_TRUTH = '/pscratch/sd/a/agolub/hss_events/HSS_events_with_ppflow/cocoa_hss_pflow_30k.root'
DEFAULT_PRED = ('/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largehsstraining4node25epoch'
                '/inference/hss_test/pred_*_merged.root')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default=DEFAULT_TRUTH)
    ap.add_argument('--pred', default=DEFAULT_PRED)
    ap.add_argument('--nprocs', type=int, default=8)
    ap.add_argument('--comps', nargs='+', default=['hgpflow', 'ppflow'])
    ap.add_argument('--frac', type=float, default=0.9)
    args = ap.parse_args()

    perf = PerformanceCOCOA(truth_path=args.truth, pred_path=args.pred,
                            ind_threshold=0.4, event_number_offset=0, llp_lxy_mm=10)
    perf.compute_jets(radius=0.4, algo='antikt', n_procs=args.nprocs,
                      store_constituent_idxs=True)
    tag_truth_jets_llp(perf)
    tag_truth_jets_decay_region(perf)

    results = {}
    for comp in args.comps:
        for dr in (0.1, 0.2):
            try:
                results[(comp, dr)] = llp_efficiency_by_region(
                    perf, comp=comp, frac=args.frac, dr_cut=dr)
            except RuntimeError as e:
                print(f'[skip {comp}] {e}')
            print()
    return results


if __name__ == '__main__':
    main()

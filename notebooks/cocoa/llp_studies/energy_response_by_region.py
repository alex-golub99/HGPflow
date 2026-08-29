"""Quantify step 3 -- jet-level ENERGY response for LLP jets, including the UNMATCHED ones.
Closes gap 1 of LLP_jet_findings.md section 9.3.

Section 9.2 claims HGPflow "recovers the energy" of displaced jets. The evidence for that is
section 3.2's cone study, which is supporting evidence rather than a response measurement, and the
jet-pT residuals of section 1, which exist only for MATCHED jets -- a ~25%-selected subset. So the
energy claim has never been tested on the population where it matters.

This measures a matched-independent response: sum reconstructed particle pT inside a cone around
the truth jet's reference axis, divided by the truth jet pT. No reco jet is required, so unmatched
jets get a number too. Reported by decay region, split matched/unmatched, at three cone sizes.

Reference axis is region-appropriate (section 10.1): the momentum axis for tracker/beyond decays,
the A0 FLIGHT direction for calorimeter decays, where the deposits collimate onto the line of
flight. Using one global axis would understate the calorimeter-region response badly.

Reading the output:
  - Eratio ~ 1 at a wide cone means the energy is present, just mispositioned (section 3.2's
    finding, now per-region and including unmatched jets).
  - A systematic offset between prompt and LLP rows at the WIDEST cone -- where geometry is
    largely integrated out -- is the signature of gap 2 of section 9.3, the calibration bias from
    reconstructing charged pion showers as neutral hadrons. The reco class composition is printed
    alongside to help attribute it.

NOTE: like region_efficiency_reco.py this needs the hgpflow_v2 package and the model's merged
prediction files, not just the raw file. Runtime is comparable (~2-3 min on a compute node).

Run:
  PYTHONPATH=<repo root> python energy_response_by_region.py \
      [--truth .../cocoa_hss_pflow_30k.root] [--pred '.../hss_test/pred_*_merged.root'] [--nprocs 16]
"""
import argparse
import numpy as np

from hgpflow_v2.performance.performance import PerformanceCOCOA
from hgpflow_v2.performance.llp_helper import (
    tag_truth_jets_llp, tag_truth_jets_decay_region)

DEFAULT_TRUTH = '/pscratch/sd/a/agolub/hss_events/HSS_events_with_ppflow/cocoa_hss_pflow_30k.root'
DEFAULT_PRED = ('/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largehsstraining4node25epoch'
                '/inference/hss_test/pred_*_merged.root')
CONES = (0.4, 0.8, 1.2)
DR_MATCH = 0.1


def _dR(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default=DEFAULT_TRUTH)
    ap.add_argument('--pred', default=DEFAULT_PRED)
    ap.add_argument('--nprocs', type=int, default=16)
    ap.add_argument('--frac', type=float, default=0.9)
    ap.add_argument('--prompt-frac', type=float, default=0.1)
    ap.add_argument('--pt-min', type=float, default=10.0)
    ap.add_argument('--eta-max', type=float, default=2.5)
    args = ap.parse_args()

    perf = PerformanceCOCOA(truth_path=args.truth, pred_path=args.pred,
                            ind_threshold=0.4, event_number_offset=0, llp_lxy_mm=10)
    perf.compute_jets(radius=0.4, algo='antikt', n_procs=args.nprocs,
                      store_constituent_idxs=True)
    tag_truth_jets_llp(perf)
    tag_truth_jets_decay_region(perf)

    ppt = perf.hgpflow_dict['hgpflow_pt']
    peta = perf.hgpflow_dict['hgpflow_eta']
    pphi = perf.hgpflow_dict['hgpflow_phi']
    pcls = perf.hgpflow_dict['hgpflow_class']
    reco_jets = perf.hgpflow_dict['jets']

    # rows: (region, matched, [Eratio per cone], ch_frac, nh_frac, ph_frac)
    rows = []
    for tj, rj, pt_a, eta_a, phi_a, cl_a in zip(
            perf.truth_dict['truth_jets'], reco_jets, ppt, peta, pphi, pcls):
        pt_a = np.asarray(pt_a); eta_a = np.asarray(eta_a)
        phi_a = np.asarray(phi_a); cl_a = np.asarray(cl_a)
        for t in tj:
            if t.pt < args.pt_min or abs(t.eta) >= args.eta_max:
                continue
            if t.llp_frac >= args.frac:
                region = t.decay_region
                if region is None:
                    continue
            elif t.llp_frac < args.prompt_frac:
                region = 'prompt'
            else:
                region = 'mixed'

            # region-appropriate reference axis (section 10.1)
            if region == 'calo' and getattr(t, 'flight_eta', None) is not None:
                ref_eta, ref_phi = t.flight_eta, t.flight_phi
            else:
                ref_eta, ref_phi = t.eta, t.phi

            matched = any(_dR(ref_eta, ref_phi, r.eta, r.phi) < DR_MATCH for r in rj)
            if len(pt_a) == 0:
                rows.append((region, matched, [0.0] * len(CONES), np.nan, np.nan, np.nan))
                continue
            dr = _dR(ref_eta, ref_phi, eta_a, phi_a)
            ratios = [pt_a[dr < c].sum() / t.pt for c in CONES]
            inc = dr < CONES[-1]
            tot = pt_a[inc].sum()
            if tot > 0:
                ch = pt_a[inc & (cl_a <= 2)].sum() / tot
                nh = pt_a[inc & (cl_a == 3)].sum() / tot
                ph = pt_a[inc & (cl_a == 4)].sum() / tot
            else:
                ch = nh = ph = np.nan
            rows.append((region, matched, ratios, ch, nh, ph))

    hdr_cones = ' '.join(f'{"R<"+str(c):>9}' for c in CONES)
    print(f'\n--- median Eratio = (reco pT in cone) / (truth jet pT), '
          f'reference axis is region-appropriate ---')
    print(f'{"region":>9} {"sel":>10} {"N":>7} {hdr_cones}   '
          f'{"<ch>":>6} {"<nh>":>6} {"<ph>":>6}')
    for region in ('prompt', 'mixed', 'tracker', 'calo', 'beyond'):
        for sel_name, want in (('all', None), ('matched', True), ('unmatched', False)):
            sub = [r for r in rows if r[0] == region and (want is None or r[1] == want)]
            if len(sub) < 20:
                continue
            meds = [np.median([r[2][i] for r in sub]) for i in range(len(CONES))]
            comp = [np.nanmean([r[3 + j] for r in sub]) for j in range(3)]
            cells = ' '.join(f'{m:>9.2f}' for m in meds)
            print(f'{region:>9} {sel_name:>10} {len(sub):>7} {cells}   '
                  f'{comp[0]:>6.2f} {comp[1]:>6.2f} {comp[2]:>6.2f}')
        print()

    print('  Eratio ~ 1 at a wide cone => energy present but mispositioned (gap 1 of section 9.3).')
    print('  Compare the WIDEST-cone prompt row against the LLP rows: a systematic offset there,')
    print('  where geometry is largely integrated out, is the calibration-bias signature (gap 2).')
    print(f'  matched/unmatched uses dR < {DR_MATCH} to the SAME region-appropriate axis.')


if __name__ == '__main__':
    main()

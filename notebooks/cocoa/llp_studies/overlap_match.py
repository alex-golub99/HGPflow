"""Quantify step 4 -- REFERENCE-FREE matching by energy overlap (LLP_jet_findings.md section 8,
point 5).

Every efficiency in this study so far depends on choosing a reference axis, and §10/§11 showed the
correct axis is region-dependent: the momentum axis for tracker decays, the A0 flight direction for
calorimeter ones. That makes cross-region comparison awkward and leaves the numbers hostage to a
convention.

This sidesteps the axis entirely. Instead of asking "is a reco jet pointing where I expect?", it
asks "does some reco jet actually CONTAIN this truth jet's energy?":

  for each truth jet T:
      mark every reco particle lying within dr_assoc of ANY of T's constituents' calorimeter
      DEPOSIT positions (particle_*_extrap_calo -- where the energy lands, not where the
      momentum points)
      for each reco jet R: shared(T,R) = sum of pT of R's constituents that are marked
      overlap(T) = max_R shared(T,R) / pT(T)
  T is matched if overlap(T) > threshold

No axis is chosen anywhere, so the same number is meaningful in every decay region. The dR-based
efficiency is printed alongside, computed on the identical selection, so the two are directly
comparable.

Caveats:
  - Normalised by TRUTH jet pT, so a region whose energy is genuinely missing (region B unmatched,
    §13.2 -- only ~36% of the energy present at R<0.8) cannot reach a high overlap no matter how
    good the matching is. Overlap measures "energy found AND clustered together", which is the
    honest thing to ask, but do not read a low value as purely a matching failure.
  - dr_assoc is a position-resolution parameter, not physics: reco particles aggregate several
    truth particles, so it must be loose enough to absorb that. Scan it with --dr-assoc.

Needs the hgpflow_v2 package and the model's merged prediction files (like steps 1 and 3).

Run:
  PYTHONPATH=<repo root> python overlap_match.py [--truth ...] [--pred ...] [--nprocs 16]
"""
import argparse
import numpy as np

from hgpflow_v2.performance.performance import PerformanceCOCOA
from hgpflow_v2.performance.llp_helper import (
    tag_truth_jets_llp, tag_truth_jets_decay_region)

DEFAULT_TRUTH = '/pscratch/sd/a/agolub/hss_events/HSS_events_with_ppflow/cocoa_hss_pflow_30k.root'
DEFAULT_PRED = ('/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largehsstraining4node25epoch'
                '/inference/hss_test/pred_*_merged.root')
THRESHOLDS = (0.3, 0.5, 0.7)
DR_MATCH = 0.1


def _dR2(e1, p1, e2, p2):
    """pairwise, broadcasting-friendly"""
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default=DEFAULT_TRUTH)
    ap.add_argument('--pred', default=DEFAULT_PRED)
    ap.add_argument('--nprocs', type=int, default=16)
    ap.add_argument('--dr-assoc', type=float, default=0.15,
                    help='reco particle counts as shared if within this of a truth deposit')
    ap.add_argument('--comps', nargs='+', default=['hgpflow', 'ppflow'],
                    help='reconstructions to evaluate: hgpflow and/or ppflow')
    ap.add_argument('--label', default='', help='free-text tag printed with the results '
                    '(e.g. which training the predictions came from)')
    ap.add_argument('--frac', type=float, default=0.9)
    ap.add_argument('--prompt-frac', type=float, default=0.1)
    ap.add_argument('--pt-min', type=float, default=10.0)
    ap.add_argument('--eta-max', type=float, default=2.5)
    args = ap.parse_args()

    perf = PerformanceCOCOA(truth_path=args.truth, pred_path=args.pred,
                            ind_threshold=0.4, event_number_offset=0, llp_lxy_mm=10)
    if 'particle_eta_extrap_calo' not in perf.truth_dict:
        raise RuntimeError('truth file has no particle_*_extrap_calo branches -- '
                           'overlap matching needs the deposit positions')
    perf.compute_jets(radius=0.4, algo='antikt', n_procs=args.nprocs,
                      store_constituent_idxs=True)
    tag_truth_jets_llp(perf)
    tag_truth_jets_decay_region(perf)

    ex_all = perf.truth_dict['particle_eta_extrap_calo']
    px_all = perf.truth_dict['particle_phi_extrap_calo']

    sources = {
        'hgpflow': ((perf.hgpflow_dict['hgpflow_pt'], perf.hgpflow_dict['hgpflow_eta'],
                     perf.hgpflow_dict['hgpflow_phi']), perf.hgpflow_dict.get('jets')),
        'ppflow': ((perf.truth_dict.get('pflow_pt'), perf.truth_dict.get('pflow_eta'),
                    perf.truth_dict.get('pflow_phi')), perf.truth_dict.get('ppflow_jets')),
    }

    for comp in args.comps:
        (cpt, ceta, cphi), comp_jets = sources[comp]
        if cpt is None or comp_jets is None:
            print(f'[skip {comp}] not available (truth file has no pflow_* branches?)')
            continue

        acc = {}   # region -> [N, n_overlap>thr..., n_dR]
        for ev, (tj, rj) in enumerate(zip(perf.truth_dict['truth_jets'], comp_jets)):
            pt_a = np.asarray(cpt[ev]); eta_a = np.asarray(ceta[ev]); phi_a = np.asarray(cphi[ev])
            ex, px_ = np.asarray(ex_all[ev]), np.asarray(px_all[ev])
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
                a = acc.setdefault(region, [0] + [0] * len(THRESHOLDS) + [0])
                a[0] += 1

                # dR-based reference (region-appropriate), for side-by-side comparison
                if region == 'calo' and getattr(t, 'flight_eta', None) is not None:
                    ref_e, ref_p = t.flight_eta, t.flight_phi
                else:
                    ref_e, ref_p = t.eta, t.phi
                if any(_dR2(ref_e, ref_p, r.eta, r.phi) < DR_MATCH for r in rj):
                    a[-1] += 1

                if len(pt_a) == 0 or len(rj) == 0:
                    continue
                idxs = t.constituent_idxs
                dep_e, dep_p = ex[idxs], px_[idxs]
                good = np.isfinite(dep_e) & np.isfinite(dep_p)
                if not good.any():
                    continue
                # (n_reco, n_truth_constituents): is each reco particle near ANY truth deposit
                dmat = _dR2(eta_a[:, None], phi_a[:, None],
                            dep_e[good][None, :], dep_p[good][None, :])
                near = (dmat < args.dr_assoc).any(axis=1)

                best = 0.0
                for r in rj:
                    ridx = r.constituent_idxs
                    if ridx is None or len(ridx) == 0:
                        continue
                    sel = ridx[near[ridx]]
                    if len(sel):
                        best = max(best, pt_a[sel].sum() / t.pt)
                for i, thr in enumerate(THRESHOLDS):
                    if best > thr:
                        a[1 + i] += 1

        thr_hdr = ' '.join(f'{"ovl>"+str(t):>9}' for t in THRESHOLDS)
        lab = f' | {args.label}' if args.label else ''
        print(f'\n--- {comp}{lab} | energy-overlap matching (dr_assoc = {args.dr_assoc}) ---')
        print(f'{"region":>9} {"N":>7} {thr_hdr}   {"dR<0.1 (ref)":>13}')
        for region in ('prompt', 'mixed', 'tracker', 'calo', 'beyond'):
            if region not in acc:
                continue
            a = acc[region]
            n = a[0]
            cells = ' '.join(f'{a[1+i]/n:>9.1%}' for i in range(len(THRESHOLDS)))
            print(f'{region:>9} {n:>7} {cells}   {a[-1]/n:>12.1%}')

    print('\n  Overlap needs NO reference axis, so it is comparable across regions; the dR column')
    print('  uses the region-appropriate axis (momentum, or flight direction for calo) and is')
    print('  shown on the identical selection for direct comparison.')
    print('  PPflow rows come from the shared truth file, so they must be IDENTICAL between runs')
    print('  on different model predictions -- a built-in alignment check.')


if __name__ == '__main__':
    main()

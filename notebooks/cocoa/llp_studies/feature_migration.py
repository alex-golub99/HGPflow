"""Measurement (b): does the choice of reconstruction push ordinary QCD jets toward
SIGNAL-LIKE features?

Measurement (a) -- fake_rate_by_sample.py -- counts spurious jets. This asks the other and
probably larger question: for jets that are REAL and correctly reconstructed, do their
tagger-visible features shift with the reconstruction algorithm?

Motivation: a displaced-jet tagger keys on tracklessness -- low charged fraction, high
neutral-hadron fraction. Section 5 showed that in HGPflow the charge label is TRACK-SEEDED, so
anything affecting track-cluster association changes the apparent neutral fraction of ordinary
jets too. If a given model systematically reconstructs QCD jets with less charged energy, those
jets migrate into the tagger's signal region without any fake being produced. There are far more
real QCD jets than fakes, so a small systematic shift can outweigh the fake rate entirely.

Method: for each truth jet find the reco jet of maximum ENERGY OVERLAP (deposit-position
association, as in overlap_match.py -- so displaced jets are handled correctly, unlike dR
matching), require overlap > --min-overlap, then compute that reco jet's feature composition from
its own constituents. Reports medians plus the fraction below a series of charged-fraction
thresholds -- i.e. the fraction of jets sitting in progressively more "signal-like" territory.

Run it on the SIGNAL sample and on a QCD sample with each model, then compare:
  - signal rows define where the tagger's signal region is;
  - the QCD row's threshold columns are the background leakage into it;
  - the comparison ACROSS reconstructions is the portable result. Absolute rates do not transfer
    to another detector/simulation, but "does in-domain training shift QCD toward signal-like
    features more than out-of-domain training does" is a property of the algorithm.

PPflow carries only a charge flag (no class), so its neutral-hadron / photon split reads as n/a;
its charged fraction -- the main tagger variable -- is available.

Run:
  PYTHONPATH=<repo root> python feature_migration.py --truth ... --pred ... --label ...
"""
import argparse
import numpy as np

from hgpflow_v2.performance.performance import PerformanceCOCOA
from hgpflow_v2.performance.llp_helper import (
    tag_truth_jets_llp, tag_truth_jets_decay_region)

CH_THRESHOLDS = (0.1, 0.2, 0.3)


def _dR2(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--nprocs', type=int, default=16)
    ap.add_argument('--dr-assoc', type=float, default=0.15)
    ap.add_argument('--min-overlap', type=float, default=0.3,
                    help='a truth jet must be this well contained before its features are used')
    ap.add_argument('--pt-min', type=float, default=10.0)
    ap.add_argument('--eta-max', type=float, default=2.5)
    ap.add_argument('--comps', nargs='+', default=['hgpflow', 'ppflow'])
    ap.add_argument('--dump', default='', help='optional .npz of per-jet features for ROC work')
    args = ap.parse_args()

    perf = PerformanceCOCOA(truth_path=args.truth, pred_path=args.pred,
                            ind_threshold=0.4, event_number_offset=0, llp_lxy_mm=10)
    if 'particle_eta_extrap_calo' not in perf.truth_dict:
        raise RuntimeError('truth file lacks particle_*_extrap_calo -- needed for association')
    perf.compute_jets(radius=0.4, algo='antikt', n_procs=args.nprocs,
                      store_constituent_idxs=True)
    tag_truth_jets_llp(perf)
    try:
        tag_truth_jets_decay_region(perf)
        have_regions = True
    except Exception as e:
        print(f'(no decay-region tagging: {e})')
        have_regions = False

    ex_all = perf.truth_dict['particle_eta_extrap_calo']
    px_all = perf.truth_dict['particle_phi_extrap_calo']

    sources = {
        'hgpflow': ((perf.hgpflow_dict['hgpflow_pt'], perf.hgpflow_dict['hgpflow_eta'],
                     perf.hgpflow_dict['hgpflow_phi'], perf.hgpflow_dict['hgpflow_class']),
                    perf.hgpflow_dict.get('jets'), True),
        'ppflow': ((perf.truth_dict.get('pflow_pt'), perf.truth_dict.get('pflow_eta'),
                    perf.truth_dict.get('pflow_phi'), perf.truth_dict.get('pflow_charge')),
                   perf.truth_dict.get('ppflow_jets'), False),
    }

    dump = {}
    for comp in args.comps:
        (cpt, ceta, cphi, ccls), comp_jets, has_class = sources[comp]
        if cpt is None or comp_jets is None:
            print(f'[skip {comp}] not available')
            continue

        rows = {}   # category -> list of (ch, nh, ph, n_const, n_ch)
        for ev, (tj, rj) in enumerate(zip(perf.truth_dict['truth_jets'], comp_jets)):
            pt_a = np.asarray(cpt[ev]); eta_a = np.asarray(ceta[ev])
            phi_a = np.asarray(cphi[ev]); cl_a = np.asarray(ccls[ev])
            ex, px_ = np.asarray(ex_all[ev]), np.asarray(px_all[ev])
            for t in tj:
                if t.pt < args.pt_min or abs(t.eta) >= args.eta_max:
                    continue
                if t.llp_frac >= 0.9:
                    cat = (t.decay_region if have_regions else 'llp') or 'llp'
                elif t.llp_frac < 0.1:
                    cat = 'prompt'
                else:
                    cat = 'mixed'
                if len(pt_a) == 0 or len(rj) == 0:
                    continue
                dep_e, dep_p = ex[t.constituent_idxs], px_[t.constituent_idxs]
                good = np.isfinite(dep_e) & np.isfinite(dep_p)
                if not good.any():
                    continue
                dmat = _dR2(eta_a[:, None], phi_a[:, None],
                            dep_e[good][None, :], dep_p[good][None, :])
                near = (dmat < args.dr_assoc).any(axis=1)

                best, best_r = 0.0, None
                for r in rj:
                    ridx = r.constituent_idxs
                    if ridx is None or len(ridx) == 0:
                        continue
                    sel = ridx[near[ridx]]
                    if len(sel):
                        o = pt_a[sel].sum() / t.pt
                        if o > best:
                            best, best_r = o, r
                if best < args.min_overlap or best_r is None:
                    continue     # not reconstructed well enough to have meaningful features

                ridx = best_r.constituent_idxs
                w = pt_a[ridx]
                tot = w.sum()
                if tot <= 0:
                    continue
                if has_class:                      # hgpflow: full class information
                    c = cl_a[ridx]
                    ch = w[c <= 2].sum() / tot
                    nh = w[c == 3].sum() / tot
                    ph = w[c == 4].sum() / tot
                    n_ch = int((c <= 2).sum())
                else:                              # ppflow: charge flag only
                    q = cl_a[ridx]
                    ch = w[q > 0].sum() / tot
                    nh = ph = np.nan
                    n_ch = int((q > 0).sum())
                rows.setdefault(cat, []).append((ch, nh, ph, len(ridx), n_ch))

        lab = f' | {args.label}' if args.label else ''
        thr_hdr = ' '.join(f'{"ch<"+str(t):>8}' for t in CH_THRESHOLDS)
        print(f'\n--- {comp}{lab} | features of well-reconstructed jets '
              f'(overlap > {args.min_overlap}) ---')
        print(f'{"category":>9} {"N":>7} {"med ch":>7} {"med nh":>7} {"med ph":>7} '
              f'{"med nconst":>11} {"med nch":>8}   {thr_hdr}')
        for cat in ('prompt', 'mixed', 'llp', 'tracker', 'calo', 'beyond'):
            if cat not in rows:
                continue
            a = np.array(rows[cat], dtype=float)
            if len(a) < 20:
                continue
            cells = ' '.join(f'{np.mean(a[:,0] < t):>8.1%}' for t in CH_THRESHOLDS)
            nh = f'{np.nanmedian(a[:,1]):>7.2f}' if np.isfinite(a[:, 1]).any() else f'{"n/a":>7}'
            ph = f'{np.nanmedian(a[:,2]):>7.2f}' if np.isfinite(a[:, 2]).any() else f'{"n/a":>7}'
            print(f'{cat:>9} {len(a):>7} {np.median(a[:,0]):>7.2f} {nh} {ph} '
                  f'{np.median(a[:,3]):>11.0f} {np.median(a[:,4]):>8.0f}   {cells}')
            dump[f'{comp}_{cat}'] = a

    print('\n  "med ch" = median CHARGED energy fraction of the reconstructed jet.')
    print('  "ch<X" columns = fraction of jets below that charged fraction, i.e. sitting in')
    print('  progressively more signal-like (trackless) territory. Compare the QCD/prompt row')
    print('  ACROSS reconstructions: a model that pushes it left is manufacturing signal-like')
    print('  background out of ordinary jets, with no fake jet involved.')
    if args.dump:
        np.savez_compressed(args.dump, **dump)
        print(f'\n  per-jet features written to {args.dump} '
              f'(columns: ch, nh, ph, n_const, n_charged)')


if __name__ == '__main__':
    main()

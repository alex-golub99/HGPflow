"""The background side of the story: RECO-jet-centric fake rate and multiplicity.

Everything in LLP_jet_findings.md sections 11-15 is signal-side -- it asks what fraction of TRUTH
jets get reconstructed. A search's sensitivity is signal efficiency against BACKGROUND rejection,
and section 15's finding that in-domain training buys +11 points of containment says nothing about
whether it also invents jets that are not there. Section 1 hints at a cost (hss produces 2.165
reco jets/event with a 0.517 fake rate, against PPflow's 1.894 / 0.467) but that was never broken
down or measured on a background sample.

This inverts the question. For every RECO jet:
  dR fake     : no truth jet within dr_match of it
  overlap fake: no truth jet supplies more than `thr` of its pT
                (shared pT computed via truth DEPOSIT positions, as in overlap_match.py, so a
                 displaced-but-correctly-clustered jet is not counted as fake)

The overlap definition is the meaningful one for the same reason it was in section 14: a reco jet
sitting on real displaced energy is not a fake merely because no truth jet points at it.

Works on any sample. Run it on signal and background with each model and compare -- the numbers
are only interpretable side by side.

Run:
  PYTHONPATH=<repo root> python fake_rate_by_sample.py --truth ... --pred ... --label ...
"""
import argparse
import numpy as np

from hgpflow_v2.performance.performance import PerformanceCOCOA

THRESHOLDS = (0.3, 0.5)
DR_MATCH = 0.1


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
    ap.add_argument('--pt-min', type=float, default=10.0)
    ap.add_argument('--eta-max', type=float, default=2.5)
    ap.add_argument('--comps', nargs='+', default=['hgpflow', 'ppflow'])
    args = ap.parse_args()

    perf = PerformanceCOCOA(truth_path=args.truth, pred_path=args.pred,
                            ind_threshold=0.4, event_number_offset=0, llp_lxy_mm=10)
    perf.compute_jets(radius=0.4, algo='antikt', n_procs=args.nprocs,
                      store_constituent_idxs=True)

    have_extrap = 'particle_eta_extrap_calo' in perf.truth_dict
    if have_extrap:
        ex_all = perf.truth_dict['particle_eta_extrap_calo']
        px_all = perf.truth_dict['particle_phi_extrap_calo']

    sources = {
        'hgpflow': ((perf.hgpflow_dict['hgpflow_pt'], perf.hgpflow_dict['hgpflow_eta'],
                     perf.hgpflow_dict['hgpflow_phi']), perf.hgpflow_dict.get('jets')),
        'ppflow': ((perf.truth_dict.get('pflow_pt'), perf.truth_dict.get('pflow_eta'),
                    perf.truth_dict.get('pflow_phi')), perf.truth_dict.get('ppflow_jets')),
    }

    n_ev = len(perf.truth_dict['truth_jets'])
    n_truth_jets = sum(1 for tj in perf.truth_dict['truth_jets'] for t in tj
                       if t.pt >= args.pt_min and abs(t.eta) < args.eta_max)
    lab = f' | {args.label}' if args.label else ''
    print(f'\n=== fake rate{lab} ===')
    print(f'events {n_ev} | truth jets (pT>{args.pt_min}, |eta|<{args.eta_max}): {n_truth_jets} '
          f'({n_truth_jets/n_ev:.2f}/ev)')
    print(f"\n{'comp':>9} {'reco jets':>10} {'/ev':>6} {'dR fake':>9} "
          f"{'ovl<0.3 fake':>13} {'ovl<0.5 fake':>13}")

    for comp in args.comps:
        (cpt, ceta, cphi), comp_jets = sources[comp]
        if cpt is None or comp_jets is None:
            print(f'[skip {comp}] not available')
            continue
        n_reco = 0
        n_dr_fake = 0
        n_ovl_fake = [0] * len(THRESHOLDS)
        for ev, (tj, rj) in enumerate(zip(perf.truth_dict['truth_jets'], comp_jets)):
            pt_a = np.asarray(cpt[ev]); eta_a = np.asarray(ceta[ev]); phi_a = np.asarray(cphi[ev])
            if have_extrap:
                ex, px_ = np.asarray(ex_all[ev]), np.asarray(px_all[ev])
            tsel = [t for t in tj if t.pt >= args.pt_min and abs(t.eta) < args.eta_max]
            for r in rj:
                if r.pt < args.pt_min or abs(r.eta) >= args.eta_max:
                    continue
                n_reco += 1
                if not any(_dR2(r.eta, r.phi, t.eta, t.phi) < DR_MATCH for t in tsel):
                    n_dr_fake += 1
                if not have_extrap or len(pt_a) == 0:
                    continue
                ridx = r.constituent_idxs
                if ridx is None or len(ridx) == 0:
                    continue
                # best fraction of THIS reco jet's pT supplied by any single truth jet
                best = 0.0
                r_pt = pt_a[ridx].sum()
                for t in tsel:
                    dep_e, dep_p = ex[t.constituent_idxs], px_[t.constituent_idxs]
                    good = np.isfinite(dep_e) & np.isfinite(dep_p)
                    if not good.any():
                        continue
                    d = _dR2(eta_a[ridx][:, None], phi_a[ridx][:, None],
                             dep_e[good][None, :], dep_p[good][None, :])
                    near = (d < args.dr_assoc).any(axis=1)
                    if near.any() and r_pt > 0:
                        best = max(best, pt_a[ridx][near].sum() / r_pt)
                for i, thr in enumerate(THRESHOLDS):
                    if best < thr:
                        n_ovl_fake[i] += 1
        if n_reco == 0:
            continue
        cells = ' '.join(f'{n_ovl_fake[i]/n_reco:>12.1%}' for i in range(len(THRESHOLDS)))
        print(f'{comp:>9} {n_reco:>10} {n_reco/n_ev:>6.2f} {n_dr_fake/n_reco:>8.1%} {cells}')

    print('\n  "dR fake"  = no truth jet within %.1f of the reco jet.' % DR_MATCH)
    print('  "ovl fake" = no single truth jet supplies that fraction of the reco jet\'s pT.')
    print('  On DISPLACED signal the dR definition over-counts fakes (a correctly reconstructed')
    print('  displaced jet has no truth jet pointing at it); the overlap definition does not.')


if __name__ == '__main__':
    main()

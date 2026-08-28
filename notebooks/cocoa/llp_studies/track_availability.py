"""LLP_jet_findings.md sections 7.1 and 6.1 -- track availability, and per-particle
deposit-vs-momentum offset, both vs decay radius.

Section 7.1: COCOA's parametric tracker never CREATES a track object beyond ~200 mm.
The four columns (entry / in_acceptance / reconstructed / linked) are identical, which is
what makes this an availability limit rather than an efficiency loss that could be loosened.

Section 6.1 (second table): dR between a particle's momentum direction and its
calo-extrapolated deposit position, split neutral vs charged. Neutrals show pure geometry
(0.004 prompt -> 0.29 displaced); charged are bending-dominated at all radii.

Run:  PYTHONPATH=<repo root> python track_availability.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

from hgpflow_v2.utility.helper_dicts import pdgid_class_dict

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
LXY_BINS = [(0, 1), (1, 10), (10, 50), (50, 200), (200, 1000), (1000, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=2000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pdgid', 'particle_track_idx', 'particle_pt',
                  'particle_prod_x', 'particle_prod_y',
                  'particle_eta', 'particle_phi',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo',
                  'track_parent_idx', 'track_reconstructed', 'track_in_acceptance'],
                 entry_stop=args.events, library='ak')
    F = lambda k: ak.to_numpy(ak.flatten(d[k]))

    pdg = F('particle_pdgid').astype(np.int64)
    cls = np.array([pdgid_class_dict.get(int(p), 5) for p in pdg])
    lxy = np.hypot(F('particle_prod_x'), F('particle_prod_y'))
    pt = F('particle_pt') / 1000.
    linked = F('particle_track_idx') >= 0

    # per-particle: does ANY track point at it, and with which flags
    he, ac, rc = [], [], []
    for ev in range(len(d['particle_pdgid'])):
        n = len(d['particle_pdgid'][ev])
        a = np.zeros(n, bool); b = np.zeros(n, bool); c = np.zeros(n, bool)
        par = ak.to_numpy(d['track_parent_idx'][ev]).astype(np.int64)
        tr = ak.to_numpy(d['track_reconstructed'][ev]).astype(bool)
        ta = ak.to_numpy(d['track_in_acceptance'][ev]).astype(bool)
        ok = (par >= 0) & (par < n)
        a[par[ok]] = True
        b[par[ok & ta]] = True
        c[par[ok & tr]] = True
        he.append(a); ac.append(b); rc.append(c)
    he, ac, rc = map(np.concatenate, (he, ac, rc))

    charged = np.isin(cls, [0, 1, 2])
    print('Track availability vs decay radius (charged particles, pT > 0.5 GeV)')
    print(f"{'Lxy [mm]':>14} {'N charged':>10} {'entry':>8} {'in_acc':>8} {'reco':>8} {'LINKED':>8}")
    for lo, hi in LXY_BINS:
        m = charged & (lxy >= lo) & (lxy < hi) & (pt > 0.5)
        if m.sum() == 0:
            continue
        print(f'{lo:>6g}-{hi:<7g} {m.sum():>10} {he[m].mean():>7.1%} {ac[m].mean():>7.1%} '
              f'{rc[m].mean():>7.1%} {linked[m].mean():>7.1%}')

    # deposit vs momentum direction
    ex, px_ = F('particle_eta_extrap_calo'), F('particle_phi_extrap_calo')
    et, ph = F('particle_eta'), F('particle_phi')
    dphi = np.abs((px_ - ph + np.pi) % (2 * np.pi) - np.pi)
    dr = np.hypot(ex - et, dphi)

    print('\ndR(momentum direction, calo-extrapolated deposit position), pT > 1 GeV')
    neutral = np.isin(cls, [3, 4])
    for name, sel in (('neutral', neutral), ('charged', charged)):
        print(f'  --- {name}')
        for lo, hi in LXY_BINS:
            m = sel & (lxy >= lo) & (lxy < hi) & (pt > 1) & np.isfinite(dr) & (np.abs(ex) < 5)
            if m.sum() < 20:
                continue
            print(f'  {lo:>6g}-{hi:<7g} N={m.sum():>8}  median dR={np.median(dr[m]):.4f}  '
                  f'90%={np.percentile(dr[m],90):.4f}')


if __name__ == '__main__':
    main()

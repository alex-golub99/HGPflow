"""LLP_jet_findings.md section 5 -- truth composition of HSS particles vs decay radius.

Shows that the truth composition is FLAT in Lxy (~68% charged everywhere), so the
"neutral hadron" excess seen in the reconstructed class columns of section 3.2 is the
trackless-charged relabelling, not a property of the LLP decay. Also reports how much
energy HGPflow can structurally label charged, given that charge is track-seeded.

Run:  PYTHONPATH=<repo root> python truth_composition.py [--raw FILE] [--events N]
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
    ap.add_argument('--events', type=int, default=3000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pdgid', 'particle_pt', 'particle_prod_x', 'particle_prod_y',
                  'particle_track_idx', 'particle_eta'],
                 entry_stop=args.events, library='ak')
    F = lambda k: ak.to_numpy(ak.flatten(d[k]))

    pdg = F('particle_pdgid').astype(np.int64)
    cls = np.array([pdgid_class_dict.get(int(p), 5) for p in pdg])
    lxy = np.hypot(F('particle_prod_x'), F('particle_prod_y'))
    pt = F('particle_pt') / 1000.
    eta = F('particle_eta')
    linked = F('particle_track_idx') >= 0

    fid = (pt > 0.5) & (np.abs(eta) < 2.5) & (cls != 5)
    charged = np.isin(cls, [0, 1, 2])

    print('pT-weighted TRUTH composition by production radius (pT>0.5 GeV, |eta|<2.5)')
    print(f"{'Lxy [mm]':>14} {'sum pT':>10} | {'charged':>8} {'neut.had':>9} {'photon':>8} "
          f"| {'ch w/ track':>12} {'ch w/o track':>13}")
    for lo, hi in LXY_BINS:
        m = fid & (lxy >= lo) & (lxy < hi)
        if m.sum() < 20:
            continue
        tot = pt[m].sum()
        chm = m & charged
        print(f'{lo:>6g}-{hi:<7g} {tot:>10.0f} | '
              f'{pt[chm].sum()/tot:>7.1%} {pt[m & (cls==3)].sum()/tot:>8.1%} '
              f'{pt[m & (cls==4)].sum()/tot:>7.1%} | '
              f'{pt[chm & linked].sum()/tot:>11.1%} {pt[chm & ~linked].sum()/tot:>12.1%}')

    print('\n--- Energy HGPflow can structurally call "charged" (charge is track-seeded) ---')
    for lo, hi in [(0, 10), (10, 200), (200, np.inf)]:
        m = fid & (lxy >= lo) & (lxy < hi)
        tot = pt[m].sum()
        truly = pt[m & charged].sum() / tot
        callable_ = pt[m & charged & linked].sum() / tot
        print(f'  Lxy {lo:g}-{hi:g} mm:  truly charged {truly:.1%},  '
              f'can be called charged {callable_:.1%}  -> forced neutral {truly-callable_:.1%}')


if __name__ == '__main__':
    main()

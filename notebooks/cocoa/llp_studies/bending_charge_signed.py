"""LLP_jet_findings.md section 6.1 -- isolate B-field bending from geometric displacement.

The `a` is neutral so it does not bend, but its charged decay products are born inside the
solenoid and bend from the decay point outward -- with a lever arm shortened by Lxy.
Bending is charge-COHERENT while geometric displacement is not, so signing dphi by the
electric charge isolates it: the geometric part cancels in the median.

Restricted to pi+-, K+-, p where sign(pdgid) == electric charge.

Result: median q*dphi falls from -0.350 (prompt, full lever arm) to exactly 0.0000 past
1 m, while median |dphi| stays ~0.17 -- bending switches itself off and what remains is
pure geometry.

Run:  PYTHONPATH=<repo root> python bending_charge_signed.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
LXY_BINS = [(0, 1), (1, 10), (10, 50), (50, 200), (200, 1000), (1000, np.inf)]
SIGN_IS_CHARGE = [211, 321, 2212]  # pi+-, K+-, p : sign(pdgid) == charge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=3000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pdgid', 'particle_pt', 'particle_eta', 'particle_phi',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    F = lambda k: ak.to_numpy(ak.flatten(d[k]))

    pdg = F('particle_pdgid').astype(np.int64)
    pt = F('particle_pt') / 1000.
    phi, ex, pe = F('particle_phi'), F('particle_eta_extrap_calo'), F('particle_phi_extrap_calo')
    lxy = np.hypot(F('particle_prod_x'), F('particle_prod_y'))
    d3 = np.sqrt(F('particle_prod_x')**2 + F('particle_prod_y')**2 + F('particle_prod_z')**2)

    q = np.sign(pdg)
    dphi = (pe - phi + np.pi) % (2 * np.pi) - np.pi          # SIGNED
    ok = (np.isin(np.abs(pdg), SIGN_IS_CHARGE) & (pt > 1)
          & (np.abs(ex) < 2.5) & np.isfinite(dphi))

    print('Charge-signed azimuthal offset q*dphi(deposit - momentum) -- isolates bending')
    print('(the sample is strongly forward, so both orderings are shown; bending itself depends')
    print(' on the TRANSVERSE lever arm, while track availability depends on 3D distance)')
    for var, name in ((d3, '3D production distance'), (lxy, 'Lxy (transverse only)')):
        print(f'\n  binned by {name}:')
        print(f"    {'bin [mm]':>14} {'N':>8} {'median q*dphi':>15} {'median |dphi|':>15}")
        for lo, hi in LXY_BINS:
            m = ok & (var >= lo) & (var < hi)
            if m.sum() < 50:
                continue
            print(f'    {lo:>6g}-{hi:<7g} {m.sum():>8} {np.median(q[m]*dphi[m]):>15.4f} '
                  f'{np.median(np.abs(dphi[m])):>15.4f}')


if __name__ == '__main__':
    main()

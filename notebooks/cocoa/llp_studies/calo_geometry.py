"""LLP_jet_findings.md methodology -- determine the COCOA calorimeter surface.

The raw file stores particle_eta_extrap_calo / particle_phi_extrap_calo (angles only), so
the 3D deposit position needs the surface. Solve for the path length by requiring
v + t*m to be parallel to the stored extrapolated direction, then look at where the
solutions land.

Result: barrel rho ~ 1496 mm for |eta| < 1.5, endcap |z| ~ 3220 mm beyond.
(Prompt particles are a degenerate case -- v ~ 0 makes the ray collinear with the stored
direction and the solve blows up; that is why the other scripts intersect the ray with the
surface analytically instead of solving.)

Run:  PYTHONPATH=<repo root> python calo_geometry.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'


def unit(e, p):
    return np.stack([np.cos(p), np.sin(p), np.sinh(e)], 1) / np.cosh(e)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=800)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    F = lambda k: ak.to_numpy(ak.flatten(d[k]))

    m = unit(F('particle_eta'), F('particle_phi'))
    u = unit(F('particle_eta_extrap_calo'), F('particle_phi_extrap_calo'))
    v = np.stack([F('particle_prod_x'), F('particle_prod_y'), F('particle_prod_z')], 1)
    pt = F('particle_pt') / 1000.
    ee = F('particle_eta_extrap_calo')

    # v + t*m parallel to u  =>  (v x u) + t (m x u) = 0
    vxu, mxu = np.cross(v, u), np.cross(m, u)
    k = np.argmax(np.abs(mxu), 1)
    i = np.arange(len(k))
    den = mxu[i, k]
    tpar = np.where(np.abs(den) > 1e-12, -vxu[i, k] / np.where(np.abs(den) > 1e-12, den, 1), np.nan)
    x = v + tpar[:, None] * m
    rho, z = np.hypot(x[:, 0], x[:, 1]), x[:, 2]

    ok = np.isfinite(rho) & (tpar > 0) & (pt > 1) & np.isfinite(ee) & (np.abs(ee) < 3)
    print(f'solved deposit positions: {ok.sum()}')
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
        s = ok & (np.abs(ee) >= lo) & (np.abs(ee) < hi)
        if s.sum() < 50:
            continue
        print(f'  |eta_extrap| {lo}-{hi}: rho median {np.median(rho[s]):7.1f} mm   '
              f'|z| median {np.median(np.abs(z[s])):7.1f} mm   N={s.sum()}')
    print('\n  -> barrel R_BAR ~ 1496.5 mm, endcap Z_EC ~ 3220 mm')


if __name__ == '__main__':
    main()

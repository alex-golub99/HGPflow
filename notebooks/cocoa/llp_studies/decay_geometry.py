"""LLP_jet_findings.md sections 6.2 and 6.3 -- the delta*L/R mechanism, and how wide
these decays actually are.

Section 6.2. For a decay product emitted at angle theta from the `a` flight direction n,
decaying at radius L and landing at calorimeter radius R:
    momentum angle from n = theta
    deposit  angle from n = theta * (1 - L/R)      <- deposits collimate toward n
Both stay CENTRED on n, so a balanced decay would suffer no shift. But a jet axis is an
energy-weighted ANGULAR POSITION. An asymmetric decay has its centroid off-axis at delta;
the rescaling moves the deposit centroid to delta*(1-L/R) while the momentum centroid
stays at delta. Residual = delta * L/R, linear in decay radius.

Column A is the physics (two pT-weighted centroids, like-for-like).
Column C is a control: a vector-sum axis vs a coordinate centroid is a pure ESTIMATOR
mismatch worth ~1.19 on prompt groups. An earlier version of this study compared those two
and was invalidated by it -- C is printed so the trap stays visible.

Section 6.3 reports the decay mass/boost/spread that make delta large.

CAVEAT: groups final-state particles by production vertex rounded to 5 mm. The raw file
has no particle_parent_idx, so decay chains cannot be reconstructed and secondary decays
contaminate the grouping. Read the SLOPE of column A, not its absolute values.

Run:  PYTHONPATH=<repo root> python decay_geometry.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0   # calorimeter inner faces, see calo_geometry.py
# The sample is strongly forward, so L/R must be computed PER PARTICLE from the actual 3D
# flight distances -- using the barrel radius for everything (as an earlier version did)
# is wrong for any decay that lands in the endcap.


def dR(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def axis_of(p, e, h):
    """direction of the vector sum of momenta"""
    vx, vy, vz = np.sum(p * np.cos(h)), np.sum(p * np.sin(h)), np.sum(p * np.sinh(e))
    return np.arcsinh(vz / np.hypot(vx, vy)), np.arctan2(vy, vx)


def centroid_of(p, e, h):
    """pT-weighted centroid of angular coordinates (phi via unit vectors)"""
    w = p / p.sum()
    return np.sum(w * e), np.arctan2(np.sum(w * np.sin(h)), np.sum(w * np.cos(h)))


def unit(e, p):
    return np.stack([np.cos(p), np.sin(p), np.sinh(e)], 1) / np.cosh(e)[:, None]


def deposit_dist(ee, pe):
    """|deposit position| : intersect the ray from the origin along the stored extrapolated
    direction with the barrel (rho=R_BAR) or endcap (|z|=Z_EC), whichever is hit first."""
    m = unit(ee, pe)
    a = m[:, 0] ** 2 + m[:, 1] ** 2
    tb = np.where(a > 1e-12, R_BAR / np.sqrt(np.maximum(a, 1e-12)), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        te = np.where(np.abs(m[:, 2]) > 1e-12, Z_EC / np.abs(m[:, 2]), np.inf)
    return np.minimum(tb, te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=3000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')

    rows, spread, masses, boosts = [], [], [], []
    for ev in range(len(d['particle_pt'])):
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev]); P = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        E = ak.to_numpy(d['particle_eta'][ev]);    H = ak.to_numpy(d['particle_phi'][ev])
        EN = ak.to_numpy(d['particle_e'][ev]) / 1000.
        CE = ak.to_numpy(d['particle_eta_extrap_calo'][ev])
        CH = ak.to_numpy(d['particle_phi_extrap_calo'][ev])
        L = np.hypot(X, Y)
        D3 = np.sqrt(X**2 + Y**2 + Z**2)              # true 3D flight distance of the decay
        RDEP = deposit_dist(CE, CH)                    # 3D distance to the deposit
        if len(P) == 0:
            continue
        key = np.stack([np.round(X / 5), np.round(Y / 5), np.round(Z / 5)], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        for g in range(len(uniq)):
            m = (inv == g) & np.isfinite(CE) & (np.abs(CE) < 3) & (P > 0.5)
            if m.sum() < 5 or P[m].sum() < 20:
                continue
            p, e, h, ce, ch = P[m], E[m], H[m], CE[m], CH[m]
            ae, ap_ = axis_of(p, e, h)
            me, mp = centroid_of(p, e, h)
            de, dp_ = centroid_of(p, ce, ch)
            lr = np.average(np.clip(D3[m] / np.maximum(RDEP[m], 1e-9), 0, 1), weights=p)
            rows.append((np.median(L[m]),
                         dR(me, mp, de, dp_),     # A: centroid vs centroid (the physics)
                         dR(ae, ap_, me, mp),     # C: estimator artifact (control)
                         lr))                     # per-group pT-weighted <L/R>, 3D
            if np.median(L[m]) > 10:
                spread.append(np.sum(p * dR(e, h, ae, ap_)) / p.sum())
                px, py, pz = (np.sum(p*np.cos(h)), np.sum(p*np.sin(h)), np.sum(p*np.sinh(e)))
                m2 = EN[m].sum()**2 - (px**2 + py**2 + pz**2)
                mm = np.sqrt(max(m2, 1e-9))
                masses.append(mm)
                boosts.append(EN[m].sum() / mm if mm > 1e-3 else np.nan)

    R = np.array(rows)
    print('--- section 6.2: centroid decomposition ---')
    print(f'N groups = {len(R)} (prompt groups included as control)\n')
    print(f"{'Lxy [mm]':>16} {'N':>6} | {'A: mom-cen vs dep-cen':>22} {'C: estimator artifact':>22}")
    for lo, hi in [(0, 10), (10, 200), (200, 500), (500, 1000), (1000, np.inf)]:
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() < 20:
            continue
        print(f'{lo:>7g}-{hi:<8g} {s.sum():>6} | {np.median(R[s,1]):>22.4f} {np.median(R[s,2]):>22.4f}')

    # The mechanism says shift = delta * (L/R). Bin directly in the PER-PARTICLE 3D L/R
    # rather than in Lxy divided by a fixed barrel radius.
    print('\n  shift vs the dimensionless <L/R> the mechanism actually predicts:')
    print(f"    {'<L/R> bin':>14} {'N':>6} {'median A':>10} {'implied delta = A/<L/R>':>26}")
    base = np.median(R[R[:, 3] < 0.02, 1]) if (R[:, 3] < 0.02).sum() > 20 else 0.0
    deltas = []
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]:
        m = (R[:, 3] >= lo) & (R[:, 3] < hi)
        if m.sum() < 25:
            continue
        a, lr = np.median(R[m, 1]), np.mean(R[m, 3])
        dl = (a - base) / lr if lr > 0.02 else np.nan
        if np.isfinite(dl) and lr > 0.05:
            deltas.append(dl)
        print(f'    {lo:>5.2f}-{hi:<8.2f} {m.sum():>6} {a:>10.4f} {dl:>26.3f}')
    print(f'    baseline (L/R < 0.02) = {base:.4f}')
    if deltas:
        print(f'\n  implied delta (mean over bins with L/R > 0.05): {np.mean(deltas):.3f} rad')

    print('\n--- section 6.3: how wide are these decays ---')
    sp = np.array(spread)
    print(f'  decay invariant mass (median)          : {np.median(masses):.1f} GeV')
    print(f'  boost E/m (median)                     : {np.nanmedian(boosts):.1f}')
    print(f'  pT-weighted angular spread about a-axis: {np.median(sp):.3f} rad')
    print(f'  decays too wide for one R=0.4 jet      : {(sp>0.4).mean():.1%}')


if __name__ == '__main__':
    main()

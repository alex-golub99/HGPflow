"""LLP_jet_findings.md section 6.5 -- is HGPflow the limitation, or are the inputs?

Takes LLP jets ONLY and bins them by their own decay radius, building the jet axis the way
HGPflow effectively does (track direction where a track exists, deposit position otherwise).

The answer: jets decaying at Lxy 10-50 mm are genuinely displaced yet reconstruct at 86.4%
axis accuracy (vs 95.8% for prompt jets). Displacement per se does NOT break the
reconstruction. Accuracy collapses in lockstep with track coverage (60.7% -> ~1%) and with
nothing else -- so HGPflow is not the bottleneck, the track collection is.

This is also what dissolves the "strong resolution but low efficiency" tension: the IQR of
0.194 in section 1 is computed on MATCHED jets, which are predominantly this well-tracked
low-Lxy population, while the efficiency is computed over ALL LLP truth jets. Two metrics,
two different populations, one underlying fact.

Note the <0.1 fraction ticks up in the last two rows while the median keeps degrading -- that
is the distribution going bimodal as L -> R, not performance recovering.

Same-constituent-set idealisation (perfect energy, perfect clustering, only POSITION varies),
as in jet_axis_track_coverage.py.

Run:  PYTHONPATH=<repo root> python llp_axis_vs_decay_radius.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0
RJET, PTMIN, NCONST, ETAMAX = 0.4, 8.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9
BINS = [(10, 50), (50, 150), (150, 300), (300, 600), (600, 1200), (1200, np.inf)]
# NOTE: COCOA's track cutoff depends on BOTH Lxy and |z| (a particle at Lxy=30mm, |z|=800mm has
# zero track availability while Lxy=30mm, |z|=100mm has 100%). This sample is strongly forward,
# so 3D production distance is the correct ordering variable; Lxy is reported alongside only to
# show what the transverse-only view was hiding.


def unit(e, p):
    return np.stack([np.cos(p), np.sin(p), np.sinh(e)], 1) / np.cosh(e)[:, None]


def ang(x):
    rho = np.hypot(x[:, 0], x[:, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.arcsinh(x[:, 2] / np.maximum(rho, 1e-9)), np.arctan2(x[:, 1], x[:, 0])


def dR(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def deposit_xyz(v, m):
    a = m[:, 0] ** 2 + m[:, 1] ** 2
    b = 2 * (v[:, 0] * m[:, 0] + v[:, 1] * m[:, 1])
    c = v[:, 0] ** 2 + v[:, 1] ** 2 - R_BAR ** 2
    tb = np.where(a > 1e-12,
                  (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-12)),
                  np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        te = np.where(np.abs(m[:, 2]) > 1e-12, (np.sign(m[:, 2]) * Z_EC - v[:, 2]) / m[:, 2], np.inf)
    return v + np.minimum(np.where(tb > 0, tb, np.inf), np.where(te > 0, te, np.inf))[:, None] * m


def axis(pmag, e, p):
    pt = pmag / np.cosh(e)
    vx, vy, vz = np.sum(pt * np.cos(p)), np.sum(pt * np.sin(p)), np.sum(pt * np.sinh(e))
    vt = np.hypot(vx, vy)
    return (np.nan, np.nan) if vt <= 0 else (np.arcsinh(vz / vt), np.arctan2(vy, vx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=4000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_track_idx', 'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    rows = []
    for ev in range(len(d['particle_pt'])):
        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        ee = ak.to_numpy(d['particle_eta_extrap_calo'][ev])
        pe = ak.to_numpy(d['particle_phi_extrap_calo'][ev])
        tk = ak.to_numpy(d['particle_track_idx'][ev]) >= 0
        PX = ak.to_numpy(d['particle_prod_x'][ev]); PY = ak.to_numpy(d['particle_prod_y'][ev])
        PZ = ak.to_numpy(d['particle_prod_z'][ev])
        lxy = np.hypot(PX, PY)
        d3 = np.sqrt(PX**2 + PY**2 + PZ**2)
        k = (pt > 0.5) & (np.abs(eta) < ETAMAX) & np.isfinite(ee)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        ee, pe, tk, lxy, d3 = ee[k], pe[k], tk[k], lxy[k], d3[k]
        de, dp_ = ang(deposit_xyz(np.zeros((k.sum(), 3)), unit(ee, pe)))
        he, hp = np.where(tk, eta, de), np.where(tk, phi, dp_)

        pjs = []
        for i in range(len(pt)):
            pj = fj.PseudoJet(float(pt[i] * np.cos(phi[i])), float(pt[i] * np.sin(phi[i])),
                              float(pt[i] * np.sinh(eta[i])), float(en[i]))
            pj.set_user_index(i)
            pjs.append(pj)
        cs = fj.ClusterSequence(pjs, jetdef)
        for J in fj.sorted_by_pt(cs.inclusive_jets(PTMIN)):
            idx = np.array([c.user_index() for c in J.constituents()])
            if len(idx) < NCONST or abs(J.eta()) >= ETAMAX:
                continue
            w = pt[idx]
            llpm = lxy[idx] > LXY_LLP
            if w[llpm].sum() / w.sum() < FRAC:
                continue
            te, tp = axis(en[idx], eta[idx], phi[idx])
            rows.append((np.average(lxy[idx][llpm], weights=w[llpm]),
                         dR(te, tp, *axis(en[idx], he[idx], hp[idx])),
                         w[tk[idx]].sum() / w.sum(),
                         np.average(d3[idx][llpm], weights=w[llpm])))

    R = np.array(rows)
    R = R[np.isfinite(R[:, 1])]
    for col, name in ((3, '3D production distance  <-- correct ordering variable'),
                      (0, 'Lxy (transverse only)   <-- what the earlier version used')):
        print(f'\nLLP jets (llp_frac >= {FRAC}) binned by {name}')
        print(f"{'bin [mm]':>16} {'N':>6} {'pT frac w/ track':>17} {'median dR':>11} "
              f"{'axis <0.1':>11} {'<0.2':>8}")
        for lo, hi in BINS:
            m = (R[:, col] >= lo) & (R[:, col] < hi)
            if m.sum() < 25:
                continue
            a = R[m]
            print(f'{lo:>7g}-{hi:<8g} {m.sum():>6} {np.mean(a[:,2]):>16.1%} '
                  f'{np.median(a[:,1]):>11.4f} {(a[:,1]<0.1).mean():>10.1%} '
                  f'{(a[:,1]<0.2).mean():>7.1%}')
    print(f'\n  total LLP jets: {len(R)}')


if __name__ == '__main__':
    main()

"""LLP_jet_findings.md section 10 -- split LLP jets by DECAY REGION, and score each against the
reference that is physically appropriate for it.

The single ~0.44 matching efficiency averages three populations with different physics and
different achievable ceilings. Splitting them makes each tractable:

  A. decay INSIDE the tracker volume (77% of A0, 85% of analysed jets)
       -> trackable in principle; extending tracking recovers 93-97% (section 6.5)
  B. decay IN the calorimeter (15% of A0, 11% of jets)
       -> no tracking can ever reach these, BUT see below
  C. decay BEYOND the calorimeter (8% of A0, 4% of jets)
       -> no energy deposited, no jet: a hard ceiling, permanently unmatched

The finding for B: with L/R -> 1 the deposits collimate onto the A0's LINE OF FLIGHT, so the
reconstructed axis measures the FLIGHT DIRECTION (origin -> decay vertex), not the decay products'
momentum axis. Scored against the flight direction, efficiency goes 28.4% -> 47.5% with no
algorithm change. The reconstruction is not failing for these jets; it is measuring a different
and arguably more useful quantity, since the flight direction points back at the vertex.

*** CAVEAT on region C ***  The numbers printed for "beyond calorimeter" are an ARTIFACT, not a
result. Particles produced past the calorimeter are never propagated, so their stored
particle_*_extrap_calo degenerates to the momentum direction and this truth-level idealisation
returns dR = 0. In reality they deposit nothing and form no reco jet at all. Region C should be
read as "permanently lost", never as "reconstructed perfectly".

Run:  PYTHONPATH=<repo root> python region_split_axis.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0
R_OUT, Z_OUT = 3825.0, 7127.0
RJET, PTMIN, NCONST, ETAMAX = 0.4, 10.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9
NAMES = {0: 'decay in TRACKER volume', 1: 'decay in CALORIMETER', 2: 'decay BEYOND calo (artifact)'}


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
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        k = (pt > 0.5) & (np.abs(eta) < ETAMAX) & np.isfinite(ee)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        ee, pe, tk = ee[k], pe[k], tk[k]
        X, Y, Z = X[k], Y[k], Z[k]
        lxy = np.hypot(X, Y)
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
            lm = lxy[idx] > LXY_LLP
            if w[lm].sum() / w.sum() < FRAC:
                continue
            vx = np.average(X[idx][lm], weights=w[lm]); vy = np.average(Y[idx][lm], weights=w[lm])
            vz = np.average(Z[idx][lm], weights=w[lm])
            vl, va = np.hypot(vx, vy), abs(vz)
            reg = 0 if (vl < R_BAR and va < Z_EC) else (1 if (vl < R_OUT and va < Z_OUT) else 2)
            te, tp = axis(en[idx], eta[idx], phi[idx])          # truth MOMENTUM axis
            re, rp = axis(en[idx], he[idx], hp[idx])            # reconstructed axis
            v = np.array([vx, vy, vz]); nv = np.linalg.norm(v)
            if nv <= 0:
                continue
            ve, vp = ang((v / nv).reshape(1, 3))                # FLIGHT direction
            rows.append((reg, dR(te, tp, re, rp), dR(ve[0], vp[0], re, rp)))

    R = np.array(rows)
    R = R[np.isfinite(R[:, 1]) & np.isfinite(R[:, 2])]
    print(f"{'LLP jets by decay region':>30} {'N':>6} | "
          f"{'vs MOMENTUM axis':>26} | {'vs FLIGHT direction':>22}")
    print(f"{'':>30} {'':>6} | {'median':>10} {'<0.1':>7} {'<0.2':>7} | {'median':>10} {'<0.1':>7}")
    for r in (0, 1, 2):
        a = R[R[:, 0] == r]
        if len(a) < 20:
            continue
        print(f'{NAMES[r]:>30} {len(a):>6} | {np.median(a[:,1]):>10.4f} '
              f'{(a[:,1]<0.1).mean():>6.1%} {(a[:,1]<0.2).mean():>6.1%} | '
              f'{np.median(a[:,2]):>10.4f} {(a[:,2]<0.1).mean():>6.1%}')
    print('\n  Region 1 (calorimeter decays): the reco axis measures the FLIGHT DIRECTION, not the')
    print('  momentum axis -- 6x better median. A reference change, not an algorithm change.')
    print('  Region 2 numbers are a degenerate-extrapolation ARTIFACT; those jets are lost. See docstring.')


if __name__ == '__main__':
    main()

"""LLP_jet_findings.md section 6.4 -- the net effect at jet level: track coverage is the
variable that sets matching efficiency.

Builds each truth jet's axis the way HGPflow effectively does -- direction from the TRACK
where a track exists, from the DEPOSIT POSITION where it does not -- and compares it to the
truth (momentum) axis. Splits jets by LLP fraction.

The result is monotonic in track coverage, not in the per-particle offset size:
    pT frac with a track   65.7% -> 35.2% -> 9.5%
    frac within dR < 0.1   95.8% -> 57.7% -> 31.3%

Prompt jets are accurate because most of their pT carries a track (exact direction) and
their remaining neutrals have deposit ~ momentum anyway. Bending never enters, because
deposit positions are barely used. The second block shows the hypothetical where the tracks
are thrown away: prompt jets fall to 75%, so bending WOULD hurt, but only mildly -- it is
not what makes prompt jets match. LLP jets barely move (31.3% -> 27.6%) because at 9.5%
coverage they are already in the trackless regime.

Same-constituent-set idealisation: perfect energy, perfect clustering, only POSITION varies.
So this isolates the axis effect; it is not a full efficiency prediction (fragmentation and
thresholds are untouched -- see section 6.3).

Run:  PYTHONPATH=<repo root> python jet_axis_track_coverage.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0        # see calo_geometry.py
RJET, PTMIN, NCONST, ETAMAX = 0.4, 8.0, 2, 2.5
LXY_LLP = 10.0


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
    tb = np.where(tb > 0, tb, np.inf)
    te = np.where(te > 0, te, np.inf)
    return v + np.minimum(tb, te)[:, None] * m


def axis(pmag, e, p):
    pt = pmag / np.cosh(e)
    vx, vy, vz = np.sum(pt * np.cos(p)), np.sum(pt * np.sin(p)), np.sum(pt * np.sinh(e))
    vt = np.hypot(vx, vy)
    return (np.nan, np.nan) if vt <= 0 else (np.arcsinh(vz / vt), np.arctan2(vy, vx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=2000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_track_idx', 'particle_prod_x', 'particle_prod_y',
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
        lxy = np.hypot(ak.to_numpy(d['particle_prod_x'][ev]),
                       ak.to_numpy(d['particle_prod_y'][ev]))

        k = (pt > 0.5) & (np.abs(eta) < ETAMAX) & np.isfinite(ee)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        ee, pe, tk, lxy = ee[k], pe[k], tk[k], lxy[k]

        de, dp_ = ang(deposit_xyz(np.zeros((k.sum(), 3)), unit(ee, pe)))
        he = np.where(tk, eta, de)          # track direction if available, else deposit
        hp = np.where(tk, phi, dp_)

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
            te, tp = axis(en[idx], eta[idx], phi[idx])
            rows.append((w[lxy[idx] > LXY_LLP].sum() / w.sum(),
                         dR(te, tp, *axis(en[idx], he[idx], hp[idx])),
                         dR(te, tp, *axis(en[idx], de[idx], dp_[idx])),
                         w[tk[idx]].sum() / w.sum()))

    R = np.array(rows)
    R = R[np.isfinite(R[:, 1]) & np.isfinite(R[:, 2])]
    print(f"{'jet type':>26} {'N':>6} {'pT frac w/ track':>17} | "
          f"{'REALISTIC (tracks used)':>28} | {'deposits for ALL':>19}")
    print(f"{'':>26} {'':>6} {'':>17} | {'med dR':>9} {'<0.1':>8} {'<0.2':>8} | {'med dR':>9} {'<0.1':>8}")
    for lbl, s in [('prompt (llp_frac<0.1)', R[:, 0] < 0.1),
                   ('mixed (0.1-0.9)', (R[:, 0] >= 0.1) & (R[:, 0] < 0.9)),
                   ('LLP (llp_frac>=0.9)', R[:, 0] >= 0.9)]:
        a = R[s]
        if len(a) < 20:
            continue
        print(f'{lbl:>26} {len(a):>6} {np.mean(a[:,3]):>16.1%} | '
              f'{np.median(a[:,1]):>9.4f} {(a[:,1]<0.1).mean():>7.1%} {(a[:,1]<0.2).mean():>7.1%} | '
              f'{np.median(a[:,2]):>9.4f} {(a[:,2]<0.1).mean():>7.1%}')


if __name__ == '__main__':
    main()

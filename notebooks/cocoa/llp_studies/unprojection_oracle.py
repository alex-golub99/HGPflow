"""LLP_jet_findings.md section 7.2 -- oracle test of displacement-aware un-projection.

For a straight-line particle, knowing the decay vertex v and the deposit position x gives
the momentum direction exactly:  m = (x - v)/|x - v|.  No regression, closed form.

This measures the CEILING of that idea by handing it the true vertex, in several flavours:
  - exact per-particle vertex
  - ONE vertex per jet (what a vertex-finder would actually deliver)
  - that jet vertex smeared, to read off the required vertex resolution

Three readings (see the doc):
  - one vertex per jet is enough (59.9% vs 58.9% -- the shared-vertex assumption is nearly free)
  - bending caps it: with straight-line propagation for everything the recovery is 94.9%;
    with real bending it is 58.9%. The gap is irreducible without tracks.
  - the vertex must be known to <= 25 mm; at 100 mm it is no better than doing nothing.

UPPER BOUND, not a predicted efficiency: uses truth particles with the SAME constituent set
for both jets (perfect energy, perfect clustering, no thresholds). Real performance is lower.

The deposit positions come from the STORED extrapolation, so B-field bending is fully
present. The script also validates its own straight-line geometry model against that stored
extrapolation: the neutral residual (~0.0075) confirms the model, the charged residual
(~0.278) IS the bending.

Run:  PYTHONPATH=<repo root> python unprojection_oracle.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0            # see calo_geometry.py
RJET, PTMIN, NCONST, ETAMAX = 0.4, 8.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9
SMEARS = [0, 10, 25, 50, 100, 200]      # mm
CHARGED_PDG = [211, 321, 2212, 11, 13]


def unit(e, p):
    return np.stack([np.cos(p), np.sin(p), np.sinh(e)], 1) / np.cosh(e)[:, None]


def ang(x):
    rho = np.hypot(x[:, 0], x[:, 1])
    with np.errstate(divide='ignore', invalid='ignore'):   # rays that miss both surfaces -> inf
        return np.arcsinh(x[:, 2] / np.maximum(rho, 1e-9)), np.arctan2(x[:, 1], x[:, 0])


def dR(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def deposit_xyz(v, m):
    """intersect ray v + t*m with barrel rho=R_BAR or endcap |z|=Z_EC, whichever comes first"""
    a = m[:, 0] ** 2 + m[:, 1] ** 2
    b = 2 * (v[:, 0] * m[:, 0] + v[:, 1] * m[:, 1])
    c = v[:, 0] ** 2 + v[:, 1] ** 2 - R_BAR ** 2
    disc = np.maximum(b * b - 4 * a * c, 0.0)
    t_bar = np.where(a > 1e-12, (-b + np.sqrt(disc)) / (2 * np.maximum(a, 1e-12)), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_ec = np.where(np.abs(m[:, 2]) > 1e-12,
                        (np.sign(m[:, 2]) * Z_EC - v[:, 2]) / m[:, 2], np.inf)
    t_bar = np.where(t_bar > 0, t_bar, np.inf)
    t_ec = np.where(t_ec > 0, t_ec, np.inf)
    return v + np.minimum(t_bar, t_ec)[:, None] * m


def axis_from(pmag, e, p):
    """four-vector sum direction: energy fixed, direction given by (e, p)"""
    pt = pmag / np.cosh(e)
    vx, vy, vz = np.sum(pt * np.cos(p)), np.sum(pt * np.sin(p)), np.sum(pt * np.sinh(e))
    vt = np.hypot(vx, vy)
    if vt <= 0:
        return np.nan, np.nan
    return np.arcsinh(vz / vt), np.arctan2(vy, vx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=3000)
    ap.add_argument('--straight-line', action='store_true',
                    help='build deposits by straight-line propagation, i.e. REMOVE bending '
                         '(reproduces the 94.9% bending-free number quoted in the doc)')
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    keys = ['dep', 'exact'] + [f'sm{s}' for s in SMEARS]
    res = {k: [] for k in keys}
    val, n_llp = [], 0

    for ev in range(len(d['particle_pt'])):
        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        pid = ak.to_numpy(d['particle_pdgid'][ev])
        vx = ak.to_numpy(d['particle_prod_x'][ev]); vy = ak.to_numpy(d['particle_prod_y'][ev])
        vz = ak.to_numpy(d['particle_prod_z'][ev])
        ee = ak.to_numpy(d['particle_eta_extrap_calo'][ev])
        pe = ak.to_numpy(d['particle_phi_extrap_calo'][ev])

        keep = (pt > 0.5) & (np.abs(eta) < 3.0) & np.isfinite(ee)
        if keep.sum() < 3:
            continue
        pt, eta, phi, en = pt[keep], eta[keep], phi[keep], en[keep]
        chg = np.isin(np.abs(pid[keep]), CHARGED_PDG)
        v = np.stack([vx[keep], vy[keep], vz[keep]], 1)
        m = unit(eta, phi)
        ee, pe = ee[keep], pe[keep]

        # validation: straight-line model vs stored extrapolation (residual = bending)
        sl_e, sl_p = ang(deposit_xyz(v, m))
        val.append(np.stack([dR(sl_e, sl_p, ee, pe), chg.astype(float)], 1))

        if args.straight_line:
            x = deposit_xyz(v, m)
        else:   # true deposit 3-position: along the stored direction, onto the calo surface
            x = deposit_xyz(np.zeros_like(v), unit(ee, pe))
        de, dp_ = ang(x)

        is_llp = np.hypot(v[:, 0], v[:, 1]) > LXY_LLP
        pmag = en                                  # massless approximation

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
            if w[is_llp[idx]].sum() / w.sum() < FRAC:
                continue
            n_llp += 1
            te, tp = axis_from(pmag[idx], eta[idx], phi[idx])
            ce, cp = axis_from(pmag[idx], de[idx], dp_[idx])
            res['dep'].append(dR(te, tp, ce, cp))

            def unproj(vv):
                w2 = x[idx] - vv
                w2 = w2 / np.maximum(np.linalg.norm(w2, axis=1, keepdims=True), 1e-9)
                ue, up = ang(w2)
                return dR(te, tp, *axis_from(pmag[idx], ue, up))

            res['exact'].append(unproj(v[idx]))
            vj = np.average(v[idx][is_llp[idx]], axis=0, weights=w[is_llp[idx]])
            for s in SMEARS:
                res[f'sm{s}'].append(unproj(vj + (rng.normal(0, s, 3) if s else 0.0)))

    V = np.concatenate(val)
    V = V[np.isfinite(V[:, 0])]
    print('straight-line model vs stored extrapolation (residual = B-field bending):')
    print(f'   neutral: median dR = {np.median(V[V[:,1]==0,0]):.4f}   (validates the geometry model)')
    print(f'   charged: median dR = {np.median(V[V[:,1]==1,0]):.4f}   (this is the bending)')
    if args.straight_line:
        print('\n   *** --straight-line: bending REMOVED from the deposits below ***')

    lbl = {'dep': 'deposit positions (what reco sees)',
           'exact': 'un-proj, exact per-particle vertex'}
    for s in SMEARS:
        lbl[f'sm{s}'] = (f'un-proj, ONE jet vertex, smear {s} mm' if s
                         else 'un-proj, ONE jet vertex (perfect)')
    print(f'\nLLP truth jets (llp_frac>={FRAC}, antikt R={RJET}, pT>{PTMIN}): {n_llp}')
    print(f"\n{'axis definition':>36} | {'median dR':>10} | {'dR<0.1':>8} {'dR<0.2':>8}")
    for k in keys:
        a = np.array(res[k]); a = a[np.isfinite(a)]
        print(f'{lbl[k]:>36} | {np.median(a):>10.4f} | {(a<0.1).mean():>7.1%} {(a<0.2).mean():>7.1%}')


if __name__ == '__main__':
    main()

"""Region-B depth split: how does CALORIMETER-decay performance vary with decay DEPTH?

Sections 13/14/16 treat "decay in the calorimeter" as one bucket spanning rho 1496 -> 3825 mm.
That is almost certainly too coarse. A decay at the ECAL face has ~2.3 m of calorimeter left and
should shower and contain normally; one deep in the HCAL has a few hundred mm and leaks out the
back -- the mechanism section 13.2 PROPOSED for region B's missing energy but never tested.

This tests it directly, and it matters for CalRatio-type analyses specifically: CalRatio targets
HCAL decays (the high E_HCAL/E_ECAL signature), so the relevant population sits in the MIDDLE of
region B, and the region-B average mixes it with early-ECAL decays that behave differently.

The leakage test uses `particle_dep_energy` -- the energy each truth particle actually deposited
in the calorimeter -- so it needs no reconstruction at all. If late showers leak, dep_E/E must
fall monotonically with decay depth. If it does not, section 13.2's proposed mechanism is wrong
and the missing energy has another cause.

Also reports jet-axis accuracy against both references per depth bin (truth-level, same
constituent set, position only -- an upper bound, and section 11.2 showed truth-level
over-predicts MOST in exactly this region).

Run:  PYTHONPATH=<repo root> python calo_depth_split.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_ECAL, Z_ECAL = 1496.5, 3220.0        # calorimeter inner faces
R_OUT, Z_OUT = 3825.0, 7127.0          # outer extent of the calorimeter
RJET, PTMIN, NCONST, ETAMAX = 0.4, 10.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9
# decay radius bins within the calorimeter (rho for barrel-like, scaled for forward)
DEPTH_BINS = [(1496, 1800), (1800, 2200), (2200, 2800), (2800, 3825)]


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


def exit_dist(v, u):
    """distance from point v along unit direction u to the calorimeter OUTER surface."""
    a = u[0] ** 2 + u[1] ** 2
    t_r = np.inf
    if a > 1e-12:
        b = 2 * (v[0] * u[0] + v[1] * u[1])
        c = v[0] ** 2 + v[1] ** 2 - R_OUT ** 2
        disc = b * b - 4 * a * c
        if disc > 0:
            t_r = (-b + np.sqrt(disc)) / (2 * a)
    t_z = np.inf
    if abs(u[2]) > 1e-12:
        t_z = (np.sign(u[2]) * Z_OUT - v[2]) / u[2]
        if t_z <= 0:
            t_z = np.inf
    return max(min(t_r, t_z), 0.0)


def axis_of(pmag, e, p):
    pt = pmag / np.cosh(e)
    vx, vy, vz = np.sum(pt * np.cos(p)), np.sum(pt * np.sin(p)), np.sum(pt * np.sinh(e))
    vt = np.hypot(vx, vy)
    return (np.nan, np.nan) if vt <= 0 else (np.arcsinh(vz / vt), np.arctan2(vy, vx))


def deposit_xyz(v, m):
    a = m[:, 0] ** 2 + m[:, 1] ** 2
    b = 2 * (v[:, 0] * m[:, 0] + v[:, 1] * m[:, 1])
    c = v[:, 0] ** 2 + v[:, 1] ** 2 - R_ECAL ** 2
    tb = np.where(a > 1e-12,
                  (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-12)),
                  np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        te = np.where(np.abs(m[:, 2]) > 1e-12,
                      (np.sign(m[:, 2]) * Z_ECAL - v[:, 2]) / m[:, 2], np.inf)
    return v + np.minimum(np.where(tb > 0, tb, np.inf), np.where(te > 0, te, np.inf))[:, None] * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=6000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_dep_energy', 'particle_track_idx',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    rows, prompt_dep = [], []
    for ev in range(len(d['particle_pt'])):
        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        dep = ak.to_numpy(d['particle_dep_energy'][ev]) / 1000.
        tk = ak.to_numpy(d['particle_track_idx'][ev]) >= 0
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        ee = ak.to_numpy(d['particle_eta_extrap_calo'][ev])
        pe = ak.to_numpy(d['particle_phi_extrap_calo'][ev])
        k = (pt > 0.5) & (np.abs(eta) < ETAMAX) & np.isfinite(ee)
        if k.sum() < 3:
            continue
        pt, eta, phi, en, dep, tk = pt[k], eta[k], phi[k], en[k], dep[k], tk[k]
        X, Y, Z, ee, pe = X[k], Y[k], Z[k], ee[k], pe[k]
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
            llpm = lxy[idx] > LXY_LLP
            frac = w[llpm].sum() / w.sum()
            e_tot = en[idx].sum()
            dep_frac = dep[idx].sum() / e_tot if e_tot > 0 else np.nan
            if frac < FRAC:
                if frac < 0.1:
                    prompt_dep.append(dep_frac)
                continue
            vx = np.average(X[idx][llpm], weights=w[llpm])
            vy = np.average(Y[idx][llpm], weights=w[llpm])
            vz = np.average(Z[idx][llpm], weights=w[llpm])
            vl, az = np.hypot(vx, vy), abs(vz)
            if not (R_ECAL <= vl < R_OUT or Z_ECAL <= az < Z_OUT):
                continue                      # region B only
            if vl >= R_OUT or az >= Z_OUT:
                continue
            v = np.array([vx, vy, vz]); nv = np.linalg.norm(v)
            if nv <= 0:
                continue
            u = v / nv
            remain = exit_dist(v, u)
            te, tp = axis_of(en[idx], eta[idx], phi[idx])
            re, rp = axis_of(en[idx], he[idx], hp[idx])
            ve, vp = ang(u.reshape(1, 3))
            rows.append((max(vl, az * R_ECAL / Z_ECAL), remain, dep_frac,
                         dR(te, tp, re, rp), dR(ve[0], vp[0], re, rp)))

    R = np.array(rows)
    pd_ = np.array([x for x in prompt_dep if np.isfinite(x)])
    print(f'region-B (calorimeter-decay) LLP jets: {len(R)}')
    if len(pd_):
        print(f'prompt-jet baseline dep_E/E (same fiducial): {np.median(pd_):.3f}')
    print(f"\n{'decay radius':>16} {'N':>6} {'calo left':>10} {'dep_E/E':>9} "
          f"{'axis<0.1 (mom)':>15} {'axis<0.1 (flight)':>18}")
    for lo, hi in DEPTH_BINS:
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() < 15:
            continue
        a = R[s]
        print(f'{lo:>7g}-{hi:<8g} {s.sum():>6} {np.median(a[:,1]):>9.0f}mm '
              f'{np.nanmedian(a[:,2]):>9.3f} {np.mean(a[:,3]<0.1):>14.1%} '
              f'{np.mean(a[:,4]<0.1):>17.1%}')
    print('\n  "calo left" = distance from the decay point to the calorimeter exit, along flight.')
    print('  "dep_E/E"   = energy actually deposited / true energy, from particle_dep_energy.')
    print('  If section 13.2\'s late-shower-leakage mechanism is right, dep_E/E must FALL')
    print('  monotonically with decay radius. If it is flat, the missing energy is something else.')


if __name__ == '__main__':
    main()

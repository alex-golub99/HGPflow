"""Quantify step 2 -- the tracking-reach payoff curve (LLP_jet_findings.md section 8, point 6).

Section 8 proposes regenerating the samples with extended tracking. Before paying for that, this
answers: what does matching efficiency look like as a FUNCTION of how far tracking reaches?

Method: cluster truth jets once (momentum space, unchanged by the assumption), then sweep an
assumed reach X. A constituent is treated as tracked if it is CHARGED, within |eta| < 2.5, and its
3D production distance <= X; tracked constituents contribute their momentum direction, untracked
ones their calorimeter deposit position. Then measure the jet-axis accuracy exactly as section 6.5
does. X = 'actual' reproduces COCOA's real track collection as the baseline.

Reach points of interest:
  ~200 mm  COCOA's current cutoff
   300 mm  roughly what demonstrated large-radius tracking (e.g. ATLAS LRT) reaches
  1500 mm  the full tracker volume (calorimeter face) -- beyond real-detector precedent

Two things the curve makes explicit:
  - region B/C jets (decay in/past the calorimeter) can NEVER benefit: no tracker reaches them,
    so the curve saturates well below 100%.
  - the payoff is strongly non-linear in X, so "extend the tracking" needs a number attached.

OPTIMISTIC BY CONSTRUCTION: assumes perfect tracking efficiency and perfect direction out to X.
Real LRT efficiency degrades with radius, so read these as upper bounds per reach point. Also
truth-level, so apply the section 11 transfer factor (tracker-region jets measured ~6 points below
their truth-level value) before quoting a real-efficiency expectation.

Run:  PYTHONPATH=<repo root> python tracking_reach_payoff.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

from hgpflow_v2.utility.helper_dicts import pdgid_class_dict

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_BAR, Z_EC = 1496.5, 3220.0
R_OUT, Z_OUT = 3825.0, 7127.0
RJET, PTMIN, NCONST, ETAMAX = 0.4, 10.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9
REACHES = [200, 300, 500, 750, 1000, 1500]      # mm, assumed 3D tracking reach
# Self-calibrated: this script's own 'actual (COCOA)' baseline gives 29.5% truth-level for
# tracker-region jets, and section 11 measured 25.3% on real model output -> 4.2 pts. (An earlier
# value of 6.0 was imported from section 6.5's aggregate, which is a different selection.) Note the
# GAINS below are independent of this constant; only the absolute levels shift.
TRANSFER = 4.2
# Section 11 measured these on REAL model output; truth-level cannot be trusted for them
# (region B carries the momentum-space clustering bias, 28.4% truth vs 0.9% real; region C is a
# degenerate-extrapolation artifact that reads ~100% here but is 0.1% in reality). Tracking
# extension cannot reach either region, so both are held fixed at their measured values.
CALO_REAL = 0.269     # section 11: calorimeter decays, FLIGHT-direction reference, dR < 0.1
BEYOND_REAL = 0.001   # section 11: deposits nothing


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


def region_of(vx, vy, vz):
    lxy, az = np.hypot(vx, vy), abs(vz)
    if lxy < R_BAR and az < Z_EC:
        return 'tracker'
    if lxy < R_OUT and az < Z_OUT:
        return 'calo'
    return 'beyond'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=4000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid',
                  'particle_track_idx', 'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'particle_eta_extrap_calo', 'particle_phi_extrap_calo'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    scenarios = ['actual'] + [str(r) for r in REACHES]
    rows = []          # (region, {scenario: dR}, {scenario: covered pT frac})
    for ev in range(len(d['particle_pt'])):
        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        pid = ak.to_numpy(d['particle_pdgid'][ev]).astype(np.int64)
        tk_real = ak.to_numpy(d['particle_track_idx'][ev]) >= 0
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        ee = ak.to_numpy(d['particle_eta_extrap_calo'][ev])
        pe = ak.to_numpy(d['particle_phi_extrap_calo'][ev])

        k = (pt > 0.5) & (np.abs(eta) < ETAMAX) & np.isfinite(ee)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        pid, tk_real, ee, pe = pid[k], tk_real[k], ee[k], pe[k]
        X, Y, Z = X[k], Y[k], Z[k]
        lxy = np.hypot(X, Y)
        d3 = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        charged = np.array([pdgid_class_dict.get(int(p), 5) in (0, 1, 2) for p in pid])
        de, dp_ = ang(deposit_xyz(np.zeros((k.sum(), 3)), unit(ee, pe)))

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
            vx = np.average(X[idx][llpm], weights=w[llpm])
            vy = np.average(Y[idx][llpm], weights=w[llpm])
            vz = np.average(Z[idx][llpm], weights=w[llpm])
            reg = region_of(vx, vy, vz)
            te, tp = axis(en[idx], eta[idx], phi[idx])

            drs, covs = {}, {}
            for sc in scenarios:
                has_tk = tk_real[idx] if sc == 'actual' else (charged[idx] & (d3[idx] <= float(sc)))
                he = np.where(has_tk, eta[idx], de[idx])
                hp = np.where(has_tk, phi[idx], dp_[idx])
                drs[sc] = dR(te, tp, *axis(en[idx], he, hp))
                covs[sc] = w[has_tk].sum() / w.sum()
            rows.append((reg, drs, covs))

    n = len(rows)
    shares = {r: sum(1 for x in rows if x[0] == r) / n for r in ('tracker', 'calo', 'beyond')}
    print(f'LLP jets (llp_frac >= {FRAC}, pT > {PTMIN}): {n}')
    print('region shares: ' + ', '.join(f'{r} {v:.1%}' for r, v in shares.items()))
    print('\nCurve is reported for TRACKER-REGION jets only -- the only population a tracking')
    print('extension can reach, and (section 11.2) the only one where truth-level transfers')
    print(f'reliably. Regions B/C are held at their section 11 measured values '
          f'({CALO_REAL:.1%}, {BEYOND_REAL:.1%}).')

    trk = [x for x in rows if x[0] == 'tracker']
    print(f'\n{"assumed reach":>14} {"pT w/ track":>12} {"tracker <0.1":>13} {"<0.2":>8}'
          f' {"tracker real*":>14} {"ALL-LLP real*":>14}')
    for sc in scenarios:
        good = [x for x in trk if np.isfinite(x[1][sc])]
        eff = np.mean([x[1][sc] < 0.1 for x in good])
        eff2 = np.mean([x[1][sc] < 0.2 for x in good])
        cov = np.mean([x[2][sc] for x in good])
        eff_real = max(eff - TRANSFER / 100, 0.0)
        allllp = (shares['tracker'] * eff_real + shares['calo'] * CALO_REAL
                  + shares['beyond'] * BEYOND_REAL)
        lab = 'actual (COCOA)' if sc == 'actual' else f'{sc} mm'
        print(f'{lab:>14} {cov:>11.1%} {eff:>12.1%} {eff2:>7.1%} {eff_real:>13.1%} {allllp:>13.1%}')

    base = None
    for sc in scenarios:
        good = [x for x in trk if np.isfinite(x[1][sc])]
        e = max(np.mean([x[1][sc] < 0.1 for x in good]) - TRANSFER / 100, 0.0)
        a = (shares['tracker'] * e + shares['calo'] * CALO_REAL + shares['beyond'] * BEYOND_REAL)
        if sc == 'actual':
            base = a
        elif sc in ('300', '1500'):
            lab = 'LRT-precedented (300 mm)' if sc == '300' else 'full tracker volume (1500 mm)'
            print(f'  {lab:>30}: all-LLP {a:.1%}  (+{(a-base)*100:.1f} pts over today)')

    print('\n  * "real" applies the section 11 transfer (%.0f pts, measured for tracker-region'
          ' jets).' % TRANSFER)
    print('    Optimistic per reach point: perfect tracking efficiency and direction out to X.')
    print('    Regions B/C are excluded from the curve -- no tracker reaches them, and their')
    print('    truth-level numbers are known-biased (section 11.2).')


if __name__ == '__main__':
    main()

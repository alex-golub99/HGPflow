"""LLP_jet_findings.md section 9 -- would a deliberately TRACK-FREE reconstruction do better?

The natural challenge to this study is: "if tracks are what matter, and LLP jets have none, is
particle flow simply the wrong tool? Shouldn't you fall back to calorimeter jets, which never
depended on tracks in the first place?"

This tests that directly. It matches truth jets against COCOA's calorimeter-only `topo_jet_*`
collection (topoclusters clustered with anti-kT -- no tracks anywhere in the chain) and compares
prompt against LLP.

Answer: NO. Calorimeter-only jets are worse everywhere, and collapse with decay distance in
exactly the same shape. The geometric problem is not particle-flow-specific -- it belongs to ANY
reconstruction that takes direction from calorimeter deposit positions, because those positions
are interpreted as pointing back to the origin.

Two caveats on the comparison, stated up front:
  - `topo_jet_*` is UNCALIBRATED, so a pT cut bites harder on it than on a calibrated collection.
    Both a 5 and a 10 GeV cut are reported so the sensitivity is visible.
  - This is a full matching efficiency ("is there any topo jet within dR<0.1"), including
    clustering and threshold effects, whereas the axis numbers in section 6.5 isolate position
    only. Do not read the two as a clean ratio -- the shape and the direction of the comparison
    are what is robust.

Run:  PYTHONPATH=<repo root> python calo_jet_baseline.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
RJET, PTMIN, NCONST, ETAMAX = 0.4, 10.0, 2, 2.5
LXY_LLP, DR_MATCH = 10.0, 0.1
D3_BINS = [(10, 150), (150, 300), (300, 600), (600, 1200), (1200, np.inf)]


def dR(e1, p1, e2, p2):
    dp = np.abs(p1 - p2)
    dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
    return np.hypot(e1 - e2, dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=4000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z',
                  'topo_jet_pt', 'topo_jet_eta', 'topo_jet_phi'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    res = {(lbl, th): [0, 0] for th in (5.0, 10.0) for lbl in ('prompt', 'LLP')}
    rows = []
    for ev in range(len(d['particle_pt'])):
        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        tj_pt = ak.to_numpy(d['topo_jet_pt'][ev]) / 1000.
        tj_e = ak.to_numpy(d['topo_jet_eta'][ev]); tj_p = ak.to_numpy(d['topo_jet_phi'][ev])

        k = (pt > 0.5) & (np.abs(eta) < ETAMAX)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        lxy = np.hypot(X, Y)[k]
        d3 = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)[k]

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
            f = w[llpm].sum() / w.sum()
            lbl = 'LLP' if f >= 0.9 else ('prompt' if f < 0.1 else None)
            if lbl is None:
                continue
            for th in (5.0, 10.0):
                sel = tj_pt > th
                ok = bool(np.any(dR(J.eta(), J.phi(), tj_e[sel], tj_p[sel]) < DR_MATCH)) \
                    if sel.sum() else False
                res[(lbl, th)][0] += 1
                res[(lbl, th)][1] += ok
            if lbl == 'LLP':
                sel = tj_pt > 10.0
                ok = bool(np.any(dR(J.eta(), J.phi(), tj_e[sel], tj_p[sel]) < DR_MATCH)) \
                    if sel.sum() else False
                rows.append((np.average(d3[idx][llpm], weights=w[llpm]), ok))

    print(f'Truth-jet matching efficiency against CALORIMETER-ONLY (topo) jets, dR < {DR_MATCH}')
    print(f"{'truth jet type':>16} {'topo jet pT cut':>16} {'N truth':>9} {'efficiency':>12}")
    for th in (5.0, 10.0):
        for lbl in ('prompt', 'LLP'):
            n, m = res[(lbl, th)]
            if n:
                print(f'{lbl:>16} {">" + str(th) + " GeV":>16} {n:>9} {m/n:>11.1%}')

    L = np.array(rows, dtype=float)
    print('\n  LLP efficiency vs 3D decay distance (calo jets, pT > 10 GeV):')
    for lo, hi in D3_BINS:
        s = (L[:, 0] >= lo) & (L[:, 0] < hi)
        if s.sum() < 25:
            continue
        print(f'    d3D {lo:>5g}-{hi:<7g}  N={s.sum():>5}  eff = {L[s,1].mean():6.1%}')
    print('\n  -> the same collapse as particle flow, from a lower starting point:')
    print('     the geometric failure is not particle-flow-specific.')


if __name__ == '__main__':
    main()

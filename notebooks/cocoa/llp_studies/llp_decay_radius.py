"""LLP_jet_findings.md section 7.1 -- WHERE do these LLP decays actually happen?

Uses the truth decay graph directly: the A0 is pdg 36, and node_decx/y/z is its decay
vertex. Classification is done in FULL 3D, because these decays are strongly FORWARD --
89% of the ones that leave the tracker exit through the endcap, not the barrel. A
transverse-only test (Lxy < 1496 mm) badly misclassifies them: a decay at |z| = 4000 mm with
Lxy = 500 mm sits in the endcap calorimeter while looking "inside the tracker".

Detector envelope (calo_geometry.py): tracker volume = rho < 1496.5 mm AND |z| < 3220 mm;
the calorimeter extends to rho ~3825 mm / |z| ~7127 mm.

Result: the sample spans all three regions --
    A0 decays        : 76.7% tracker | 15.2% calorimeter |  8.1% beyond calorimeter
    LLP jets analysed: 84.8% tracker | 11.2% calorimeter |  4.0% beyond
So there ARE genuine calorimeter-decay LLP jets here (~11% of those analysed), and the study's
jet sample is only mildly biased toward tracker decays by the |eta|<2.5 / pT / jet-existence
requirements.

Consequence: a tracking extension addresses the ~77-85% decaying inside the tracker volume,
but the ~15% decaying in the calorimeter can never be tracked by anything and need
CalRatio / timing / shower-shape methods instead. The ~8% decaying beyond the calorimeter are
a genuine acceptance loss.

Run:  PYTHONPATH=<repo root> python llp_decay_radius.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import fastjet as fj
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_ECAL, Z_ECAL = 1496.5, 3220.0     # calorimeter inner faces
R_OUT, Z_OUT = 3825.0, 7127.0       # outer extent of the calorimeter layers
A0_PDG = 36
RJET, PTMIN, NCONST, ETAMAX = 0.4, 10.0, 2, 2.5
LXY_LLP, FRAC = 10.0, 0.9


def region(lxy, az):
    """(inside tracker, inside calorimeter, beyond calorimeter) -- full 3D"""
    it = (lxy < R_ECAL) & (az < Z_ECAL)
    ic = (~it) & (lxy < R_OUT) & (az < Z_OUT)
    return it, ic, ~(it | ic)


def report(name, lxy, az, extra=''):
    it, ic, bo = region(lxy, az)
    print(f'{name}:  N = {len(lxy)}{extra}')
    print(f'    inside tracker : {it.mean():7.1%}    in calorimeter : {ic.mean():7.1%}    '
          f'beyond calo : {bo.mean():7.1%}')
    print(f'    median Lxy {np.median(lxy):6.0f} mm    median |z| {np.median(az):6.0f} mm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=4000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['node_pdg_id', 'node_decx', 'node_decy', 'node_decz',
                  'particle_pt', 'particle_eta', 'particle_phi', 'particle_e',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z'],
                 entry_stop=args.events, library='ak')
    jetdef = fj.JetDefinition(fj.antikt_algorithm, RJET)

    a_lxy, a_az, a_flight, j_lxy, j_az = [], [], [], [], []
    for ev in range(len(d['particle_pt'])):
        npd = ak.to_numpy(d['node_pdg_id'][ev])
        am = np.abs(npd) == A0_PDG
        ax = ak.to_numpy(d['node_decx'][ev])[am]
        ay = ak.to_numpy(d['node_decy'][ev])[am]
        az_ = ak.to_numpy(d['node_decz'][ev])[am]
        g = np.isfinite(ax) & np.isfinite(az_)
        a_lxy.append(np.hypot(ax[g], ay[g]))
        a_az.append(np.abs(az_[g]))
        a_flight.append(np.sqrt(ax[g] ** 2 + ay[g] ** 2 + az_[g] ** 2))

        pt = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        eta = ak.to_numpy(d['particle_eta'][ev]); phi = ak.to_numpy(d['particle_phi'][ev])
        en = ak.to_numpy(d['particle_e'][ev]) / 1000.
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        k = (pt > 0.5) & (np.abs(eta) < ETAMAX)
        if k.sum() < 3:
            continue
        pt, eta, phi, en = pt[k], eta[k], phi[k], en[k]
        lxy, Z = np.hypot(X, Y)[k], Z[k]
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
            j_lxy.append(np.average(lxy[idx][llpm], weights=w[llpm]))
            j_az.append(np.average(np.abs(Z[idx][llpm]), weights=w[llpm]))

    a_lxy = np.concatenate(a_lxy); a_az = np.concatenate(a_az)
    a_flight = np.concatenate(a_flight)
    j_lxy, j_az = np.array(j_lxy), np.array(j_az)

    print(f'A0 3D flight distance: median {np.median(a_flight):.0f} mm, '
          f'90% {np.percentile(a_flight,90):.0f} mm, 99% {np.percentile(a_flight,99):.0f} mm\n')
    report('A0 DECAYS (truth decay graph, all)', a_lxy, a_az)
    out = ~region(a_lxy, a_az)[0]
    print(f'    of those leaving the tracker: {(a_lxy[out]>=R_ECAL).mean():.1%} exit the barrel, '
          f'{(a_az[out]>=Z_ECAL).mean():.1%} exit the endcap  <-- strongly FORWARD')
    print()
    report('LLP JETS analysed in the study', j_lxy, j_az)
    print(f'\n  The jet sample is only mildly biased toward tracker decays; genuine '
          f'calorimeter-decay\n  jets are ~11% of it, NOT a negligible fraction.')


if __name__ == '__main__':
    main()

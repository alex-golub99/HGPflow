"""LLP_jet_findings.md section 7.3, part 1 -- does the depth lever arm exist, and how good
is the shower axis it defines?

Part 1: the lever arm DOES exist -- ~83% of displaced particles make >=2 topoclusters, with
a median radial span between barycentres of ~612 mm.

Part 2: but the axes are poor. A single shower axis is good to ~12 deg at absolute best
(neutral, cell-level) and ~41 deg from the topocluster barycentres the network actually
sees -- WORSE than cells, because 2-3 barycentres are dominated by lateral shower
fluctuation. This is what makes the vertex fit fail (see vertex_resolution.py).

Two figures of merit per shower:
  - impact parameter: perpendicular distance from the TRUE vertex to the fitted line
  - angle error: angle between the fitted axis and the particle's true momentum direction

CAVEAT: uses truth cell->particle links (cell_parent_idx) to group showers. A real
algorithm would have to do that grouping itself, so these are BEST CASE numbers.

Run:  PYTHONPATH=<repo root> python shower_axis_quality.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
CHARGED_PDG = [211, 321, 2212, 11, 13]


def line_fit(pts, w):
    """energy-weighted PCA -> (point on line, unit direction)"""
    a = np.average(pts, axis=0, weights=w)
    Q = (pts - a) * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return a, Vt[0] / np.linalg.norm(Vt[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=400)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['cell_x', 'cell_y', 'cell_z', 'cell_e', 'cell_topo_idx', 'cell_parent_idx',
                  'particle_pt', 'particle_eta', 'particle_phi', 'particle_pdgid',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z'],
                 entry_stop=args.events, library='ak')

    ncell, span_cells, ntopo, span_topo, rows = [], [], [], [], []
    for ev in range(len(d['cell_x'])):
        cx = ak.to_numpy(d['cell_x'][ev]); cy = ak.to_numpy(d['cell_y'][ev])
        cz = ak.to_numpy(d['cell_z'][ev]); ce = ak.to_numpy(d['cell_e'][ev])
        cp = ak.to_numpy(d['cell_parent_idx'][ev]).astype(np.int64)
        ct = ak.to_numpy(d['cell_topo_idx'][ev]).astype(np.int64)
        P = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        pid = ak.to_numpy(d['particle_pdgid'][ev])
        E = ak.to_numpy(d['particle_eta'][ev]); H = ak.to_numpy(d['particle_phi'][ev])
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        if len(P) == 0:
            continue
        L = np.hypot(X, Y)
        pos = np.stack([cx, cy, cz], 1)
        cr = np.linalg.norm(pos, axis=1)

        for pi in np.unique(cp):
            if pi < 0 or pi >= len(P) or P[pi] < 1 or L[pi] < 10:
                continue
            m = cp == pi
            if m.sum() < 2:
                continue
            ncell.append(m.sum())
            span_cells.append(cr[m].max() - cr[m].min())

            tt = np.unique(ct[m][ct[m] >= 0])
            ntopo.append(len(tt))
            tp, tw, rr = [], [], []
            for T in tt:
                mm = m & (ct == T)
                if ce[mm].sum() <= 0:
                    continue
                tp.append(np.average(pos[mm], axis=0, weights=ce[mm]))
                tw.append(ce[mm].sum())
                rr.append(np.average(cr[mm], weights=ce[mm]))
            if len(rr) >= 2:
                span_topo.append(max(rr) - min(rr))

            if m.sum() < 3:
                continue
            v = np.array([X[pi], Y[pi], Z[pi]])
            mdir = np.array([np.cos(H[pi]), np.sin(H[pi]), np.sinh(E[pi])])
            mdir /= np.linalg.norm(mdir)
            a_c, d_c = line_fit(pos[m], ce[m])
            has_topo = len(tp) >= 2
            a_t, d_t = line_fit(np.array(tp), np.array(tw)) if has_topo else (a_c, d_c)

            def ip(a, dv):
                r = v - a
                return np.linalg.norm(r - np.dot(r, dv) * dv)

            def angerr(dv):
                return np.degrees(np.arccos(np.clip(abs(np.dot(dv, mdir)), -1, 1)))

            rows.append((ip(a_c, d_c), angerr(d_c),
                         ip(a_t, d_t) if has_topo else np.nan,
                         angerr(d_t) if has_topo else np.nan,
                         float(np.isin(abs(pid[pi]), CHARGED_PDG))))

    nc, st, nt, sp = map(np.array, (ncell, span_topo, ntopo, span_cells))
    print('--- part 1: does the depth lever arm exist? (displaced, pT>1 GeV) ---')
    print(f'  displaced particles with cells : {len(nc)}')
    print(f'  cells per particle (median)    : {np.median(nc):.0f}')
    print(f'  radial span of cells (median)  : {np.median(sp):.0f} mm   '
          f'(>500mm: {(sp>500).mean():.1%})')
    print(f'  topoclusters per particle      : median {np.median(nt):.0f}, '
          f'frac >=2: {(nt>=2).mean():.1%}')
    print(f'  radial span between topo barycentres (>=2): median {np.median(st):.0f} mm')

    R = np.array(rows)
    print('\n--- part 2: how well does ONE fitted shower axis point at the true vertex? ---')
    print(f"{'line source':>18} {'N':>7} | {'impact param to true vtx':>26} | {'angle err vs momentum':>24}")
    for lbl, ic, ia, sel in [
            ('cells (all)', 0, 1, np.ones(len(R), bool)),
            ('cells (neutral)', 0, 1, R[:, 4] == 0),
            ('cells (charged)', 0, 1, R[:, 4] == 1),
            ('topo bary (all)', 2, 3, np.isfinite(R[:, 2])),
            ('topo bary (neutral)', 2, 3, np.isfinite(R[:, 2]) & (R[:, 4] == 0))]:
        s = sel & np.isfinite(R[:, ic])
        if s.sum() < 20:
            continue
        print(f'{lbl:>18} {s.sum():>7} | med {np.median(R[s,ic]):>7.1f} mm  '
              f'68% {np.percentile(R[s,ic],68):>7.1f} | '
              f'med {np.median(R[s,ia]):>6.1f} deg  68% {np.percentile(R[s,ia],68):>6.1f}')


if __name__ == '__main__':
    main()

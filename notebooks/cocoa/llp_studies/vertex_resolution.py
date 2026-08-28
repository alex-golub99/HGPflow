"""LLP_jet_findings.md section 7.3, part 2 -- can the displaced vertex be found from the
calorimeter alone, to the <=25 mm that unprojection_oracle.py says is required?

Method: fit a line per displaced particle shower (energy-weighted PCA), then find the point
closest to all the lines from one decay:
    v = [sum_i w_i (I - d_i d_i^T)]^-1 [sum_i w_i (I - d_i d_i^T) a_i]

Answer: NO. Median 3D error ~1700 mm from topocluster barycentres (~1540 mm at cell level,
the upper bound) against a ~25 mm requirement -- short by a factor of ~70. More constituents
barely help, because the lines are nearly parallel and the fit is ill-conditioned along the
flight direction. This is an information limit, not an effort limit, and it is why the
calorimeter-pointing / vertex-finding route was dropped.

CAVEATS: uses truth cell->particle links to group showers (a real algorithm would have to do
that itself), and groups particles into decays by production vertex rounded to 5 mm. Both
make this a BEST CASE.

Run:  PYTHONPATH=<repo root> python vertex_resolution.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
CHARGED_PDG = [211, 321, 2212, 11, 13]


def line_fit(pts, w):
    a = np.average(pts, axis=0, weights=w)
    Q = (pts - a) * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return a, Vt[0] / np.linalg.norm(Vt[0])


def vertex_fit(lines, wts):
    """closest point to a set of lines: min sum_i w_i |(I - d d^T)(v - a)|^2"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (a, dv), w in zip(lines, wts):
        M = (np.eye(3) - np.outer(dv, dv)) * w
        A += M; b += M @ a
    if np.linalg.cond(A) > 1e10:
        return None
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=600)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['cell_x', 'cell_y', 'cell_z', 'cell_e', 'cell_topo_idx', 'cell_parent_idx',
                  'particle_pt', 'particle_e', 'particle_pdgid',
                  'particle_prod_x', 'particle_prod_y', 'particle_prod_z'],
                 entry_stop=args.events, library='ak')

    out = {'topo': [], 'cell': [], 'topo_neut': []}
    for ev in range(len(d['cell_x'])):
        cx = ak.to_numpy(d['cell_x'][ev]); cy = ak.to_numpy(d['cell_y'][ev])
        cz = ak.to_numpy(d['cell_z'][ev]); ce = ak.to_numpy(d['cell_e'][ev])
        cp = ak.to_numpy(d['cell_parent_idx'][ev]).astype(np.int64)
        ct = ak.to_numpy(d['cell_topo_idx'][ev]).astype(np.int64)
        P = ak.to_numpy(d['particle_pt'][ev]) / 1000.
        EN = ak.to_numpy(d['particle_e'][ev]) / 1000.
        pid = ak.to_numpy(d['particle_pdgid'][ev])
        X = ak.to_numpy(d['particle_prod_x'][ev]); Y = ak.to_numpy(d['particle_prod_y'][ev])
        Z = ak.to_numpy(d['particle_prod_z'][ev])
        if len(P) == 0:
            continue
        L = np.hypot(X, Y)
        pos = np.stack([cx, cy, cz], 1)

        sel = np.where((L > 10) & (P > 1))[0]
        if len(sel) < 2:
            continue
        key = np.stack([np.round(X[sel] / 5), np.round(Y[sel] / 5), np.round(Z[sel] / 5)], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        for g in range(len(uniq)):
            members = sel[inv == g]
            if len(members) < 2:
                continue
            vtrue = np.array([np.mean(X[members]), np.mean(Y[members]), np.mean(Z[members])])
            for mode in ('topo', 'cell', 'topo_neut'):
                lines, wts = [], []
                for pi in members:
                    if mode == 'topo_neut' and np.isin(abs(pid[pi]), CHARGED_PDG):
                        continue
                    m = cp == pi
                    if m.sum() < 2:
                        continue
                    if mode == 'cell':
                        pts, w = pos[m], ce[m]
                    else:
                        pts, w = [], []
                        for T in np.unique(ct[m][ct[m] >= 0]):
                            mm = m & (ct == T)
                            if ce[mm].sum() <= 0:
                                continue
                            pts.append(np.average(pos[mm], axis=0, weights=ce[mm]))
                            w.append(ce[mm].sum())
                        if len(pts) < 2:
                            continue
                        pts, w = np.array(pts), np.array(w)
                    if len(pts) < 2 or w.sum() <= 0:
                        continue
                    if np.linalg.norm(pts.max(0) - pts.min(0)) < 100:   # need a lever arm
                        continue
                    lines.append(line_fit(pts, w)); wts.append(EN[pi])
                if len(lines) < 2:
                    continue
                v = vertex_fit(lines, wts)
                if v is None:
                    continue
                out[mode].append((np.linalg.norm(v - vtrue),
                                  abs(np.hypot(v[0], v[1]) - np.hypot(vtrue[0], vtrue[1])),
                                  len(lines), np.hypot(vtrue[0], vtrue[1])))

    print(f"{'source':>16} {'N vtx':>7} | {'|dv| 3D med':>12} {'68%':>8} | {'dLxy med':>10} {'68%':>8}")
    for k, lbl in [('topo', 'topo barycentres'), ('cell', 'cells (best case)'),
                   ('topo_neut', 'topo, neutrals')]:
        a = np.array(out[k])
        if len(a) == 0:
            continue
        print(f'{lbl:>16} {len(a):>7} | {np.median(a[:,0]):>12.1f} '
              f'{np.percentile(a[:,0],68):>8.1f} | {np.median(a[:,1]):>10.1f} '
              f'{np.percentile(a[:,1],68):>8.1f}   [mm]')

    a = np.array(out['topo'])
    print('\n  topo-barycentre fit, by number of contributing particle-lines:')
    for lo, hi in [(2, 3), (3, 5), (5, 8), (8, 100)]:
        s = (a[:, 2] >= lo) & (a[:, 2] < hi)
        if s.sum() < 15:
            continue
        print(f'    {lo}-{hi-1} lines: N={s.sum():>5}  median |dv| = {np.median(a[s,0]):>7.1f} mm')
    print('\n  by true Lxy:')
    for lo, hi in [(10, 200), (200, 500), (500, 1000), (1000, np.inf)]:
        s = (a[:, 3] >= lo) & (a[:, 3] < hi)
        if s.sum() < 15:
            continue
        print(f'    Lxy {lo:>5g}-{hi:<7g}: N={s.sum():>5}  median |dv| = {np.median(a[s,0]):>7.1f} mm')
    print('\n  REQUIRED for the un-projection to help: <= 25 mm (see unprojection_oracle.py)')


if __name__ == '__main__':
    main()

"""LLP_jet_findings.md section 10.3 -- what ctau was this sample generated with, and could ANY
ctau have put "most" decays in the calorimeter?

Extracts the proper decay length per A0 from the truth graph: ctau_i = d3_i / (betagamma)_i,
whose distribution should be exponential with mean ctau if the generator used one lifetime.

Then the structural point: the decay-position distribution is EXPONENTIAL, with its mode at
zero. For each A0 (direction + boost known), the probability of decaying inside the calorimeter
shell is exp(-t_in/lambda) - exp(-t_out/lambda), lambda = betagamma*ctau, where t_in/t_out are
the 3D path lengths to the tracker and outer-calorimeter boundaries along its flight direction.
Scanning ctau gives the maximum achievable calorimeter fraction for this boost spectrum -- no
choice of ctau can beat it.

Also corrects two section-6.3 numbers that were VISIBLE-daughter quantities mislabelled as decay
properties: the true m_a is fixed at 55 GeV (the 18.3 GeV was the visible vertex-group mass),
and the true boost is E/m ~ 1.2 (betagamma ~ 0.6), far below the visible-group 3.1 -- the a is
heavy and slow, which is WHY the decays are wide and why the lab decay length is compressed.

Run:  PYTHONPATH=<repo root> python ctau_and_boost.py [--raw FILE] [--events N]
"""
import argparse
import awkward as ak
import numpy as np
import uproot

DEFAULT_RAW = '/pscratch/sd/a/agolub/hss_events/cocoa_hss_val_15k.root'
R_ECAL, Z_ECAL = 1496.5, 3220.0
R_OUT, Z_OUT = 3825.0, 7127.0


def path_to(u, R, Z):
    """3D path length from the origin along unit direction u to cylinder (R, |z|<Z) exit."""
    a = u[:, 0] ** 2 + u[:, 1] ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tr = np.where(a > 1e-12, R / np.sqrt(np.maximum(a, 1e-12)), np.inf)
        tz = np.where(np.abs(u[:, 2]) > 1e-12, Z / np.abs(u[:, 2]), np.inf)
    return np.minimum(tr, tz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=DEFAULT_RAW)
    ap.add_argument('--events', type=int, default=4000)
    args = ap.parse_args()

    t = uproot.open(args.raw)['Out_Tree']
    d = t.arrays(['node_pdg_id', 'node_pt', 'node_eta', 'node_m',
                  'node_prodx', 'node_prody', 'node_prodz',
                  'node_decx', 'node_decy', 'node_decz'],
                 entry_stop=args.events, library='ak')
    F = lambda k: ak.to_numpy(ak.flatten(d[k]))
    sel = np.abs(F('node_pdg_id')) == 36
    pt, eta, m = F('node_pt')[sel], F('node_eta')[sel], F('node_m')[sel]
    dx = F('node_decx')[sel] - F('node_prodx')[sel]
    dy = F('node_decy')[sel] - F('node_prody')[sel]
    dz = F('node_decz')[sel] - F('node_prodz')[sel]
    ok = np.isfinite(dx) & np.isfinite(dz)
    pt, eta, m, dx, dy, dz = pt[ok], eta[ok], m[ok], dx[ok], dy[ok], dz[ok]

    p = pt * np.cosh(eta)
    bg = p / m                       # betagamma
    Eom = np.sqrt(1 + bg ** 2)       # E/m
    d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    ell = d3 / bg                    # proper decay length per A0

    print(f'A0 sample (N = {len(d3)}):')
    print(f'  m_a            : {np.unique(m)} GeV (fixed at generation)')
    print(f'  pT             : median {np.median(pt):.1f} GeV')
    print(f'  betagamma      : median {np.median(bg):.2f}   [25,75] = '
          f'{np.percentile(bg,25):.2f}, {np.percentile(bg,75):.2f}')
    print(f'  boost E/m      : median {np.median(Eom):.2f}   <- heavy and SLOW')
    print(f'  lab flight d3  : median {np.median(d3):.0f} mm, mean {np.mean(d3):.0f} mm')
    print(f'\nproper decay length ell = d3/betagamma:')
    print(f'  median {np.median(ell):.0f} mm   mean {np.mean(ell):.0f} mm   '
          f'mean/median = {np.mean(ell)/np.median(ell):.3f} (exponential predicts 1.443)')
    ctau = np.mean(ell)
    print(f'  => generated ctau ~ {ctau:.0f} mm = {ctau/1000:.2f} m '
          f'(cross-check median/ln2 = {np.median(ell)/np.log(2):.0f} mm)')

    # region probabilities vs ctau, using each A0's true direction and boost
    u = np.stack([dx, dy, dz], 1) / np.maximum(d3, 1e-9)[:, None]
    t_in = path_to(u, R_ECAL, Z_ECAL)
    t_out = path_to(u, R_OUT, Z_OUT)

    def fracs(ct):
        lam = bg * ct
        p_trk = 1 - np.exp(-t_in / lam)
        p_cal = np.exp(-t_in / lam) - np.exp(-t_out / lam)
        return p_trk.mean(), p_cal.mean(), 1 - p_trk.mean() - p_cal.mean()

    a, b, c = fracs(ctau)
    print(f'\npredicted fractions at measured ctau : tracker {a:.1%} | calo {b:.1%} | beyond {c:.1%}')
    print(f'measured fractions (llp_decay_radius): tracker 76.7% | calo 15.2% | beyond  8.1%')

    grid = np.linspace(200, 12000, 200)
    cals = [fracs(ct)[1] for ct in grid]
    i = int(np.argmax(cals))
    print(f'\nscan: calo fraction is maximised at ctau = {grid[i]:.0f} mm '
          f'({grid[i]/1000:.1f} m), giving {cals[i]:.1%}')
    print(f'  -> the calorimeter fraction can NEVER exceed ~{cals[i]:.0%} for this boost')
    print(f'     spectrum, at any ctau: the exponential has its mode at zero, so most decays')
    print(f'     always come before the mean. "Most decays in the calorimeter" is unreachable.')


if __name__ == '__main__':
    main()

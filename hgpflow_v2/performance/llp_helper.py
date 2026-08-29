"""
Tag and select jets by long-lived-particle (LLP) origin, for evaluating HGPflow
on displaced jets (e.g. HSS: H -> aa, a a pseudoscalar decaying at a displaced
vertex).

An LLP-origin truth particle is one whose production transverse displacement
Lxy = sqrt(prod_x^2 + prod_y^2) exceeds a threshold (set via PerformanceCOCOA's
llp_lxy_mm). A jet's LLP fraction is the pT-weighted fraction of its constituents
that are LLP-origin. Tagging is done once (tag_truth_jets_llp); the fraction cut
is then a cheap plot-time parameter you can scan.

Usage:
    perf = PerformanceCOCOA(truth_path=..., pred_path=..., ind_threshold=0.4,
                            event_number_offset=<splitter -eo>, llp_lxy_mm=10)
    # ... alignment check (expect ~0.95) ...
    perf.compute_jets(radius=0.4, algo='antikt', store_constituent_idxs=True)  # MUST be True
    perf.match_jets()
    tag_truth_jets_llp(perf)          # stores .llp_frac on every truth jet, once
    llp_frac_summary(perf)            # see the distribution + counts per cut

    FRAC = 0.9                        # <-- scan this
    res = compute_jet_residual_dict({
        'hgpflow': select_matched_by_llp(perf.hgpflow_dict['matched_hgpflow_jets'], FRAC),
        'proxy'  : select_matched_by_llp(perf.hgpflow_dict['matched_proxy_jets'],   FRAC),
    })
    plot_jet_residuals(res, pt_relative=True)

    # efficiency + residual vs pT, all reconstructions at once -- the plotted form of
    # llp_efficiency_summary's table:
    fig, summaries = plot_llp_comparison(perf, frac=FRAC)
"""
import numpy as np


def tag_truth_jets_llp(perf):
    """Store `.llp_frac` (LLP-origin pT fraction, 0..1) on every truth jet.

    Requires: PerformanceCOCOA(..., llp_lxy_mm=<mm>) and
              compute_jets(..., store_constituent_idxs=True).
    """
    if 'particle_llp_origin' not in perf.truth_dict:
        raise RuntimeError("build PerformanceCOCOA with llp_lxy_mm=<mm> to enable LLP tagging")
    if 'truth_jets' not in perf.truth_dict:
        raise RuntimeError("call perf.compute_jets(..., store_constituent_idxs=True) first")

    pt_all = perf.truth_dict['particle_pt']
    llp_all = perf.truth_dict['particle_llp_origin']

    n_jets = 0
    for ev, jets in enumerate(perf.truth_dict['truth_jets']):
        pts, llp = pt_all[ev], llp_all[ev]
        for j in jets:
            if j.constituent_idxs is None:
                raise RuntimeError("truth jets have no constituent_idxs -- "
                                   "compute_jets(..., store_constituent_idxs=True)")
            idxs = j.constituent_idxs
            cpt = pts[idxs]
            tot = cpt.sum()
            j.llp_frac = float(cpt[llp[idxs]].sum() / tot) if tot > 0 else 0.0
            n_jets += 1
    print(f"tagged {n_jets} truth jets with .llp_frac "
          f"(Lxy > {perf.llp_lxy_mm} mm = LLP-origin)")


def get_matched_jets(perf, comp):
    """Resolve a reconstruction name to its (truth, reco) matched-jet pair.

    comp: 'hgpflow' (full model), 'proxy' (stage-1 output), or 'ppflow'
    (parametric particle-flow baseline -- only if the truth file has pflow_*
    branches). The truth 'ref' jet is the same object in all three, so its
    .llp_frac from tag_truth_jets_llp applies to every comparison.
    """
    loc = {
        'hgpflow': (perf.hgpflow_dict, 'matched_hgpflow_jets'),
        'proxy':   (perf.hgpflow_dict, 'matched_proxy_jets'),
        'ppflow':  (perf.truth_dict,   'matched_ppflow_jets'),
    }
    if comp not in loc:
        raise ValueError(f"comp must be one of {list(loc)}, got {comp!r}")
    d, key = loc[comp]
    if key not in d:
        raise RuntimeError(
            f"{key} not found. For comp='ppflow' the truth file must have pflow_* "
            f"branches; and call perf.match_jets() after perf.compute_jets(...).")
    return d[key]


def select_matched_by_llp(matched_pair, frac_cut, keep='llp'):
    """Filter matched (truth, reco) jet pairs by the TRUTH jet's LLP fraction.

    matched_pair : (ref_jets_per_event, comp_jets_per_event) as produced by
                   match_jets (e.g. perf.hgpflow_dict['matched_hgpflow_jets']).
    frac_cut     : threshold on truth-jet LLP pT fraction.
    keep         : 'llp'    -> keep jets with llp_frac >= frac_cut (displaced jets)
                   'prompt' -> keep jets with llp_frac <  frac_cut (prompt control)
    Returns a new (ref, comp) pair in the same structure, ready for
    compute_jet_residual_dict.
    """
    ref_ev, comp_ev = matched_pair
    out_ref, out_comp = [], []
    for refs, comps in zip(ref_ev, comp_ev):
        r_keep, c_keep = [], []
        for r, c in zip(refs, comps):
            f = getattr(r, 'llp_frac', None)
            if f is None:
                raise RuntimeError("run tag_truth_jets_llp(perf) before selecting")
            passes = (f >= frac_cut) if keep == 'llp' else (f < frac_cut)
            if passes:
                r_keep.append(r)
                c_keep.append(c)
        out_ref.append(r_keep)
        out_comp.append(c_keep)
    return (out_ref, out_comp)


def llp_efficiency_summary(perf, frac=0.9, comp='hgpflow', dr_cut=0.1,
                           pt_bins=(20, 30, 50, 80, 150), pt_min=10, eta_max=2.5,
                           verbose=True):
    """Per-pT-bin matching efficiency AND pT residual for LLP jets (llp_frac >= frac)
    vs prompt jets (llp_frac < frac), scored on the same events.

    Reporting BOTH is essential for LLP jets: the residual histogram
    (compute_jet_residual_dict / plot_jet_residuals) only includes jets that matched
    within dr_cut, so for track-less displaced jets it describes a heavily-selected
    survivor subset. Efficiency = fraction of truth jets that got a reco match within
    dr_cut; residual = (reco-truth)/truth among the matched.

    comp : 'hgpflow' (full model), 'proxy' (stage-1 output), or 'ppflow'
           (parametric baseline, if the truth file has pflow_* branches).
    Requires perf.compute_jets(store_constituent_idxs=True), perf.match_jets(),
    and tag_truth_jets_llp(perf).
    Returns {keep: {(lo,hi): {eff, med, iqr, q25, q75, n_truth, n_matched}}}.
    """
    matched = get_matched_jets(perf, comp)
    edges = list(zip(pt_bins[:-1], pt_bins[1:]))

    if verbose:
        print(f"--- {comp} jets | LLP cut llp_frac >= {frac} | match dr < {dr_cut} ---")
        print(f'{"pT bin":>9} | {"LLP eff":>8} {"LLP med":>8} {"LLP IQR":>8} '
              f'| {"prm eff":>8} {"prm med":>8} {"prm IQR":>8}')
    out = {}
    for lo, hi in edges:
        cells = {}
        for keep in ('llp', 'prompt'):
            ref_ev, comp_ev = select_matched_by_llp(matched, frac, keep=keep)
            n_sel = n_ok = 0
            res = []
            for refs, comps in zip(ref_ev, comp_ev):
                for r, c in zip(refs, comps):
                    if lo <= r.pt < hi and r.pt > pt_min and abs(r.eta) < eta_max:
                        n_sel += 1
                        if r.delta_R(c) < dr_cut:
                            n_ok += 1
                            res.append((c.pt - r.pt) / r.pt)
            res = np.array(res)
            q25, q75 = ((np.percentile(res, 25), np.percentile(res, 75))
                        if len(res) else (np.nan, np.nan))
            cells[keep] = dict(
                eff=(n_ok / n_sel if n_sel else np.nan),
                med=(np.median(res) if len(res) else np.nan),
                iqr=(q75 - q25), q25=q25, q75=q75,
                n_truth=n_sel, n_matched=n_ok)
            out.setdefault(keep, {})[(lo, hi)] = cells[keep]
        if verbose:
            l, p = cells['llp'], cells['prompt']
            print(f'{lo:>4}-{hi:<4} | {l["eff"]:>7.1%} {l["med"]:>+8.3f} {l["iqr"]:>8.3f} '
                  f'| {p["eff"]:>7.1%} {p["med"]:>+8.3f} {p["iqr"]:>8.3f}')
    return out


def _wilson(k, n, z=1.0):
    """Wilson score interval for a binomial efficiency k/n, as (lo_err, hi_err)
    distances from the central k/n. Used instead of sqrt(eff(1-eff)/n) because LLP
    bins are low-stat and often sit near eff=0 or 1, where the normal approximation
    puts error bars outside [0, 1]."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # clamp: at p exactly 0 or 1 the bound coincides with p up to float noise,
    # which would otherwise reach matplotlib as a negative error bar
    return max(0.0, p - lo), max(0.0, hi - p)


def plot_llp_comparison(perf, frac=0.9, comps=('hgpflow', 'proxy', 'ppflow'), dr_cut=0.1,
                        pt_bins=(20, 30, 50, 80, 150), pt_min=10, eta_max=2.5,
                        stylesheet=None, annotate_counts=True):
    """Visual form of llp_efficiency_summary: compare reconstructions on LLP vs prompt jets.

    2x2 grid -- matching efficiency (top) and pT-residual median (bottom), for
    LLP jets (left, llp_frac >= frac) and prompt jets (right). Rows share y-limits so
    the left/right panels read against each other: the LLP-vs-prompt gap at fixed pT
    is the quantity of interest, and the prompt column is the control.

    Efficiency error bars are Wilson intervals; residual error bars span the IQR
    (25th-75th percentile), which is asymmetric about the median by construction.

    Same requirements and arguments as llp_efficiency_summary.
    Returns (fig, {comp: summary_dict}).
    """
    import matplotlib.pyplot as plt
    from .style_sheet import FIG_W, FIG_H_1ROW, FIG_DPI
    from .utils import update_stylesheet

    _COLORS, _LABELS, _, _, _LINE_STYLES, _ = update_stylesheet(stylesheet)
    _MARKERS = {'hgpflow': 'o', 'proxy': 's', 'ppflow': '^'}

    summaries = {c: llp_efficiency_summary(perf, frac=frac, comp=c, dr_cut=dr_cut,
                                           pt_bins=pt_bins, pt_min=pt_min,
                                           eta_max=eta_max, verbose=False)
                 for c in comps}

    edges = list(zip(pt_bins[:-1], pt_bins[1:]))
    # categorical x: the pT bins are uneven, so plotting at bin centres would put a
    # dodged marker at an apparent pT it was never measured at (e.g. 123 in the 80-150
    # bin). One slot per bin instead -- same as the table this replaces.
    x0 = np.arange(len(edges))

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W * 0.75, FIG_H_1ROW * 2), dpi=FIG_DPI,
                             sharex=True)
    fig.subplots_adjust(hspace=0.12, wspace=0.28)

    for col, keep in enumerate(('llp', 'prompt')):
        ax_eff, ax_res = axes[0, col], axes[1, col]

        for i, comp in enumerate(comps):
            cells = [summaries[comp][keep][e] for e in edges]
            # dodge each series within its slot so error bars stay separable
            x = x0 + (i - (len(comps) - 1) / 2) * 0.16
            kw = dict(color=_COLORS[comp], marker=_MARKERS.get(comp, 'o'), markersize=8,
                      linewidth=2, linestyle=_LINE_STYLES[comp], label=_LABELS[comp],
                      markeredgecolor='white', markeredgewidth=0.8, zorder=3)

            eff = np.array([c['eff'] for c in cells])
            err = np.array([_wilson(c['n_matched'], c['n_truth']) for c in cells]).T
            ax_eff.errorbar(x, eff, yerr=err, capsize=3, **kw)

            med = np.array([c['med'] for c in cells])
            lo_err = med - np.array([c['q25'] for c in cells])
            hi_err = np.array([c['q75'] for c in cells]) - med
            ax_res.errorbar(x, med, yerr=np.vstack([lo_err, hi_err]), capsize=3, **kw)

        n_lab = 'LLP' if keep == 'llp' else 'prompt'
        ax_eff.set_title(f'{n_lab} jets' + (f'  ($f_{{LLP}} \\geq {frac}$)' if keep == 'llp'
                                            else f'  ($f_{{LLP}} < {frac}$)'),
                         fontsize=11, pad=8)
        ax_eff.set_ylabel(f'Matching efficiency  ($\\Delta R < {dr_cut}$)')
        ax_res.set_ylabel('Jet $(p_T^{reco} - p_T^{truth})/p_T^{truth}$')
        ax_res.set_xlabel('Truth jet $p_T$ [GeV]')
        ax_res.axhline(0, color='k', linestyle=':', linewidth=1, zorder=1)
        ax_res.set_xticks(x0)
        ax_res.set_xticklabels([f'{lo}-{hi}' for lo, hi in edges])
        ax_res.set_xlim(-0.5, len(edges) - 0.5)

        if annotate_counts:
            # truth-jet count per bin: the LLP panels are low-stat, and an efficiency
            # point means little without the denominator it was computed from
            n_truth = [summaries[comps[0]][keep][e]['n_truth'] for e in edges]
            for xc, n in zip(x0, n_truth):
                ax_eff.annotate(f'n={n}', xy=(xc, 0.02), xycoords=('data', 'axes fraction'),
                                ha='center', va='bottom', fontsize=7, color='dimgray',
                                zorder=4, bbox=dict(facecolor='white', edgecolor='none',
                                                    pad=0.6, alpha=0.85))

        for ax in (ax_eff, ax_res):
            ax.grid(color='k', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
            ax.tick_params(which='both', direction='in', top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax.minorticks_on()

    for row in axes:  # shared y per row -> left/right panels are directly comparable
        ylo = min(ax.get_ylim()[0] for ax in row)
        yhi = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(ylo, yhi)
    axes[0, 0].set_ylim(top=min(axes[0, 0].get_ylim()[1], 1.05))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(comps), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.96))

    return fig, summaries


def llp_frac_summary(perf, cuts=(0.1, 0.5, 0.9)):
    """Print the truth-jet LLP-fraction distribution and how many jets survive
    each candidate cut -- use it to pick a cut and know your statistics."""
    fracs = np.array([getattr(j, 'llp_frac', np.nan)
                      for ev in perf.truth_dict['truth_jets'] for j in ev])
    fracs = fracs[np.isfinite(fracs)]
    n = len(fracs)
    print(f"truth jets: {n}")
    print(f"  llp_frac ~0 (prompt): {(fracs < 0.05).mean():.1%}   "
          f"~1 (pure LLP): {(fracs > 0.95).mean():.1%}")
    for c in cuts:
        print(f"  llp_frac >= {c:.2f}: {(fracs >= c).sum():6d} jets ({(fracs >= c).mean():.1%})")
    return fracs


# --- decay-region tagging and region-aware efficiency (LLP_jet_findings.md section 10) ---
# Detector envelope, determined empirically from the stored calo extrapolation
# (see notebooks/cocoa/llp_studies/calo_geometry.py):
_R_ECAL, _Z_ECAL = 1496.5, 3220.0   # calorimeter inner faces = tracker-volume boundary [mm]
_R_OUT, _Z_OUT = 3825.0, 7127.0     # outer extent of the calorimeter layers [mm]


def _delta_R(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi
    return float(np.hypot(eta1 - eta2, dphi))


def tag_truth_jets_decay_region(perf):
    """Store the LLP decay vertex and its detector region on every truth jet.

    Per jet (pT-weighted mean production vertex of its LLP-origin constituents, mm):
      .decay_vtx    -- (x, y, z), or None for jets with no LLP constituent
      .decay_d3     -- 3D distance of that vertex from the origin
      .decay_region -- 'tracker' | 'calo' | 'beyond' | None
      .flight_eta / .flight_phi -- direction of the origin->vertex line (the A0
                       flight direction; the correct reference axis for 'calo' jets)

    Classification is FULL 3D (the HSS sample is strongly forward; a transverse-only
    test misclassifies endcap decays -- see LLP_jet_findings.md, header callout).
    Requires: PerformanceCOCOA(..., llp_lxy_mm=<mm>), compute_jets(...,
    store_constituent_idxs=True), and tag_truth_jets_llp(perf) run first.
    """
    for key in ('particle_prod_x', 'particle_prod_y', 'particle_prod_z'):
        if key not in perf.truth_dict:
            raise RuntimeError("production vertex not loaded -- rebuild PerformanceCOCOA "
                               "with llp_lxy_mm=<mm> (reader keeps prod_x/y/z aligned)")
    px_all = perf.truth_dict['particle_prod_x']
    py_all = perf.truth_dict['particle_prod_y']
    pz_all = perf.truth_dict['particle_prod_z']
    pt_all = perf.truth_dict['particle_pt']
    llp_all = perf.truth_dict['particle_llp_origin']

    counts = {'tracker': 0, 'calo': 0, 'beyond': 0, None: 0}
    for ev, jets in enumerate(perf.truth_dict['truth_jets']):
        px, py, pz = px_all[ev], py_all[ev], pz_all[ev]
        pts, llp = pt_all[ev], llp_all[ev]
        for j in jets:
            if getattr(j, 'llp_frac', None) is None:
                raise RuntimeError("run tag_truth_jets_llp(perf) first")
            idxs = j.constituent_idxs
            lm = llp[idxs]
            if not lm.any():
                j.decay_vtx = None
                j.decay_d3 = 0.0
                j.decay_region = None
                j.flight_eta = j.flight_phi = None
                counts[None] += 1
                continue
            w = pts[idxs][lm]
            vx = float(np.average(px[idxs][lm], weights=w))
            vy = float(np.average(py[idxs][lm], weights=w))
            vz = float(np.average(pz[idxs][lm], weights=w))
            lxy, az = np.hypot(vx, vy), abs(vz)
            if lxy < _R_ECAL and az < _Z_ECAL:
                region = 'tracker'
            elif lxy < _R_OUT and az < _Z_OUT:
                region = 'calo'
            else:
                region = 'beyond'
            j.decay_vtx = (vx, vy, vz)
            j.decay_d3 = float(np.sqrt(vx**2 + vy**2 + vz**2))
            j.decay_region = region
            j.flight_eta = float(np.arcsinh(vz / max(lxy, 1e-9)))
            j.flight_phi = float(np.arctan2(vy, vx))
            counts[region] += 1
    print('tagged truth-jet decay regions:',
          {k if k else 'no-LLP': v for k, v in counts.items()})


def llp_efficiency_by_region(perf, comp='hgpflow', frac=0.9, dr_cut=0.1,
                             pt_min=10, eta_max=2.5, prompt_frac=0.1, verbose=True):
    """Truth-jet-centric matching efficiency split by decay region, scored against
    BOTH candidate references: the jet momentum axis, and the A0 flight direction.

    A truth jet counts as matched if ANY reco jet of the chosen collection lies
    within dr_cut of the reference axis. LLP jets (llp_frac >= frac) are split by
    .decay_region. Below that, jets are split into a CLEAN prompt control
    (llp_frac < prompt_frac, default 0.1 -- matching the section 6.4 convention)
    and a 'mixed' row (prompt_frac <= llp_frac < frac).

    Keeping prompt and mixed separate matters: an earlier version lumped
    everything below `frac` into one 'prompt' row, which absorbed the partially
    displaced jets and understated true prompt performance (72.6% for the blend
    vs the clean control). Do not compare a blended row against section 6.4's
    llp_frac < 0.1 numbers.

    Expectation from the truth-level study (LLP_jet_findings.md section 10):
    'tracker' jets score on the momentum axis; 'calo' jets score much better on
    the flight direction (reference change, 28.4% -> 47.5% at truth level);
    'beyond' jets deposit nothing and should sit near zero on both.

    comp: 'hgpflow' | 'proxy' | 'ppflow'.
    Requires tag_truth_jets_llp(perf) and tag_truth_jets_decay_region(perf).
    Returns {region: {n, eff_mom, eff_flight}} (eff_flight is None for prompt).
    """
    loc = {'hgpflow': (perf.hgpflow_dict, 'jets'),
           'proxy': (perf.hgpflow_dict, 'proxy_jets'),
           'ppflow': (perf.truth_dict, 'ppflow_jets')}
    d, key = loc[comp]
    if key not in d:
        raise RuntimeError(f"{key} not found -- call perf.compute_jets(...) first"
                           + (" (truth file has no pflow_* branches)" if comp == 'ppflow' else ""))
    reco_ev = d[key]

    acc = {}
    for tjets, rjets in zip(perf.truth_dict['truth_jets'], reco_ev):
        for t in tjets:
            if t.pt < pt_min or abs(t.eta) >= eta_max:
                continue
            if t.llp_frac >= frac:
                region = t.decay_region
                if region is None:      # LLP by fraction but no LLP constituent: skip
                    continue
            elif t.llp_frac < prompt_frac:
                region = 'prompt'       # clean control (matches the section 6.4 convention)
            else:
                region = 'mixed'        # partially displaced -- kept OUT of the control
            a = acc.setdefault(region, [0, 0, 0])
            a[0] += 1
            if any(_delta_R(t.eta, t.phi, r.eta, r.phi) < dr_cut for r in rjets):
                a[1] += 1
            # flight direction exists for anything with an LLP constituent (mixed included)
            if getattr(t, 'flight_eta', None) is not None:
                if any(_delta_R(t.flight_eta, t.flight_phi, r.eta, r.phi) < dr_cut for r in rjets):
                    a[2] += 1

    out = {}
    if verbose:
        print(f"--- {comp} | LLP: llp_frac >= {frac} | prompt: llp_frac < {prompt_frac} | "
              f"dR < {dr_cut} | pT > {pt_min}, |eta| < {eta_max} ---")
        print(f"{'region':>10} {'N':>7} {'eff (momentum axis)':>20} {'eff (flight dir)':>18}")
    for region in ('prompt', 'mixed', 'tracker', 'calo', 'beyond'):
        if region not in acc:
            continue
        n, m_mom, m_fl = acc[region]
        out[region] = dict(n=n, eff_mom=m_mom / n if n else np.nan,
                           eff_flight=(m_fl / n if n else np.nan) if region != 'prompt' else None)
        if verbose:
            fl = f"{out[region]['eff_flight']:>17.1%}" if out[region]['eff_flight'] is not None else f"{'--':>17}"
            print(f"{region:>10} {n:>7} {out[region]['eff_mom']:>19.1%} {fl}")
    return out

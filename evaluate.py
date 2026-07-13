"""
Evaluate HGPflow full-pipeline inference results.

Usage:
    python evaluate.py --pred_path <path/to/pred_merged.root> \
                       --ind_threshold 0.48 \
                       --output_dir ./eval_plots

The pred file must come from hgpflow_v2.eval and contain truth_*, proxy_*, and
hgpflow_* branches. No separate truth file is required.
"""
import argparse
import os

import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

CLASS_NAMES = {0: 'ch. had.', 1: 'electron', 2: 'muon', 3: 'neut. had.', 4: 'photon'}
CHARGED_THRESHOLD = 2  # class <= this is charged


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_pred_single(pred_path, ind_threshold):
    tree = uproot.open(pred_path)['event_tree']
    arrays = tree.arrays(library='np')

    ind_mask = np.array([x > ind_threshold for x in arrays['pred_ind']], dtype=object)

    data = {}
    for prefix in ('hgpflow', 'proxy'):
        for var in ('pt', 'eta', 'phi'):
            key = f'{prefix}_{var}'
            data[key] = np.array([x[m] for x, m in zip(arrays[key], ind_mask)], dtype=object)
    data['hgpflow_class'] = np.array(
        [x[m] for x, m in zip(arrays['hgpflow_class'], ind_mask)], dtype=object)

    for var in ('pt', 'eta', 'phi', 'class'):
        data[f'truth_{var}'] = arrays[f'truth_{var}']

    data['event_number'] = arrays['event_number']
    return data


def load_pred(pred_path, ind_threshold):
    """pred_path may be a single file or a glob pattern (multiple pred files are
    concatenated; event numbers are globally unique across split files)."""
    paths = sorted(glob.glob(pred_path)) if any(c in pred_path for c in '*?[') else [pred_path]
    assert len(paths) > 0, f'no files match {pred_path}'

    parts = [_load_pred_single(p, ind_threshold) for p in tqdm(paths, desc='loading pred files')]
    if len(parts) == 1:
        data = parts[0]
    else:
        data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}

    print(f"Loaded {len(data['event_number'])} events from {len(paths)} file(s)")
    return data


def apply_proxy_overwrite(data):
    """
    Stage 2 only corrects neutral kinematics. For charged particles, proxy
    (track-based) eta/phi/pt are better, so overwrite hgpflow with proxy.
    """
    for i in range(len(data['hgpflow_pt'])):
        data['hgpflow_eta'][i] = data['proxy_eta'][i].copy()
        data['hgpflow_phi'][i] = data['proxy_phi'][i].copy()
        ch_mask = data['hgpflow_class'][i] <= CHARGED_THRESHOLD
        data['hgpflow_pt'][i] = data['hgpflow_pt'][i].copy()
        data['hgpflow_pt'][i][ch_mask] = data['proxy_pt'][i][ch_mask]
    return data


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def _delta_r(eta1, eta2, phi1, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def _hung_match_ev(ref_pt, ref_eta, ref_phi, ref_cl,
                   comp_pt, comp_eta, comp_phi, comp_cl):
    empty = {k: np.array([]) for k in
             ('ref_pt', 'ref_eta', 'ref_phi', 'ref_cl',
              'comp_pt', 'comp_eta', 'comp_phi', 'comp_cl')}
    if len(ref_pt) == 0 or len(comp_pt) == 0:
        return empty, np.arange(len(ref_pt)), np.arange(len(comp_pt))

    cost_dpt2 = (ref_pt[:, None] - comp_pt[None, :]) ** 2 / np.where(ref_pt[:, None] > 0, ref_pt[:, None] ** 2, 1.0)
    cost_dr   = _delta_r(ref_eta[:, None], comp_eta[None, :],
                         ref_phi[:, None], comp_phi[None, :])
    cost = np.sqrt(cost_dpt2 + cost_dr ** 2)
    cost = np.where(np.isfinite(cost), cost, 1e9)

    ref_ch  = ref_cl  <= CHARGED_THRESHOLD
    comp_ch = comp_cl <= CHARGED_THRESHOLD

    row_idx = np.array([], dtype=int)
    col_idx = np.array([], dtype=int)
    for is_charged in (True, False):
        rm = ref_ch if is_charged else ~ref_ch
        cm = comp_ch if is_charged else ~comp_ch
        if rm.sum() == 0 or cm.sum() == 0:
            continue
        r_i, c_i = linear_sum_assignment(cost[np.ix_(rm, cm)])
        row_idx = np.concatenate([row_idx, np.where(rm)[0][r_i]])
        col_idx = np.concatenate([col_idx, np.where(cm)[0][c_i]])

    matched = {
        'ref_pt':  ref_pt[row_idx],  'ref_eta':  ref_eta[row_idx],
        'ref_phi': ref_phi[row_idx], 'ref_cl':   ref_cl[row_idx],
        'comp_pt': comp_pt[col_idx], 'comp_eta': comp_eta[col_idx],
        'comp_phi':comp_phi[col_idx],'comp_cl':  comp_cl[col_idx],
    }
    unmatched_truth = np.setdiff1d(np.arange(len(ref_pt)),  row_idx)
    unmatched_pred  = np.setdiff1d(np.arange(len(comp_pt)), col_idx)
    return matched, unmatched_truth, unmatched_pred


def match_all_events(data):
    keys = ('ref_pt', 'ref_eta', 'ref_phi', 'ref_cl',
            'comp_pt', 'comp_eta', 'comp_phi', 'comp_cl')
    all_matched = {k: [] for k in keys}
    unmatched_t_pt, unmatched_t_cl = [], []
    unmatched_p_pt, unmatched_p_cl = [], []

    for i in tqdm(range(len(data['truth_pt'])), desc='Matching events'):
        matched, u_t, u_p = _hung_match_ev(
            data['truth_pt'][i],         data['truth_eta'][i],
            data['truth_phi'][i],        data['truth_class'][i].astype(int),
            data['hgpflow_pt'][i],       data['hgpflow_eta'][i],
            data['hgpflow_phi'][i],      data['hgpflow_class'][i].astype(int),
        )
        for k in keys:
            all_matched[k].append(matched[k])
        unmatched_t_pt.append(data['truth_pt'][i][u_t])
        unmatched_t_cl.append(data['truth_class'][i][u_t].astype(int))
        unmatched_p_pt.append(data['hgpflow_pt'][i][u_p])
        unmatched_p_cl.append(data['hgpflow_class'][i][u_p].astype(int))

    for k in keys:
        all_matched[k] = np.concatenate(all_matched[k])

    return (
        all_matched,
        np.concatenate(unmatched_t_pt), np.concatenate(unmatched_t_cl),
        np.concatenate(unmatched_p_pt), np.concatenate(unmatched_p_cl),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PT_BINS = np.array([1, 2, 5, 10, 20, 40, 80, 200])


def _bin_efficiency(pt, is_matched, bins):
    eff, centers = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pt >= lo) & (pt < hi)
        n_total = mask.sum()
        n_matched = is_matched[mask].sum() if n_total > 0 else 0
        eff.append(n_matched / n_total if n_total > 0 else np.nan)
        centers.append(0.5 * (lo + hi))
    return np.array(centers), np.array(eff)


def plot_pt_residuals(matched, output_dir):
    ref_pt  = matched['ref_pt']
    comp_pt = matched['comp_pt']
    ref_cl  = matched['ref_cl']
    # zero-pt truth entries -> nan (excluded by the isfinite filters below)
    residual = (comp_pt - ref_pt) / np.where(ref_pt > 0, ref_pt, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('pT residuals: (pred - truth) / truth')

    for ax, label, mask in [
        (axes[0], 'Charged (class 0-2)', ref_cl <= CHARGED_THRESHOLD),
        (axes[1], 'Neutral (class 3-4)', ref_cl >  CHARGED_THRESHOLD),
    ]:
        if mask.sum() == 0:
            ax.set_title(f'{label}\n(no matched pairs)')
            continue
        vals = residual[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.set_title(f'{label}\n(no finite residuals)')
            continue
        q1, q99 = np.percentile(vals, 1), np.percentile(vals, 99)
        ax.hist(vals, bins=50, range=(q1, q99), histtype='stepfilled',
                alpha=0.6, color='steelblue', density=True)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        median = np.nanmedian(vals)
        iqr    = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
        ax.set_title(f'{label}\nmedian={median:+.3f}, IQR={iqr:.3f}  (n={mask.sum()})')
        ax.set_xlabel(r'$(p_T^\mathrm{pred} - p_T^\mathrm{truth}) / p_T^\mathrm{truth}$')
        ax.set_ylabel('density')

    fig.tight_layout()
    path = os.path.join(output_dir, 'pt_residuals.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_efficiency(matched, unmatched_t_pt, unmatched_t_cl, output_dir):
    ref_pt = matched['ref_pt']
    ref_cl = matched['ref_cl']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Efficiency vs truth $p_T$')

    for ax, label, charged in [
        (axes[0], 'Charged (class 0-2)', True),
        (axes[1], 'Neutral (class 3-4)', False),
    ]:
        if charged:
            m_pt = ref_pt[ref_cl  <= CHARGED_THRESHOLD]
            u_pt = unmatched_t_pt[unmatched_t_cl <= CHARGED_THRESHOLD]
        else:
            m_pt = ref_pt[ref_cl  >  CHARGED_THRESHOLD]
            u_pt = unmatched_t_pt[unmatched_t_cl >  CHARGED_THRESHOLD]

        all_pt = np.concatenate([m_pt, u_pt])
        is_matched = np.concatenate([np.ones(len(m_pt)), np.zeros(len(u_pt))]).astype(bool)

        centers, eff = _bin_efficiency(all_pt, is_matched, PT_BINS)
        ax.step(np.repeat(PT_BINS, 2)[1:-1], np.repeat(eff, 2),
                color='steelblue', linewidth=1.5, where='mid')
        ax.scatter(centers, eff, color='steelblue', zorder=5)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'truth $p_T$ [GeV]')
        ax.set_ylabel('efficiency')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'efficiency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_fake_rate(matched, unmatched_p_pt, unmatched_p_cl, output_dir):
    comp_pt = matched['comp_pt']
    comp_cl = matched['comp_cl']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fake rate vs predicted $p_T$')

    for ax, label, charged in [
        (axes[0], 'Charged (class 0-2)', True),
        (axes[1], 'Neutral (class 3-4)', False),
    ]:
        if charged:
            m_pt = comp_pt[comp_cl  <= CHARGED_THRESHOLD]
            u_pt = unmatched_p_pt[unmatched_p_cl <= CHARGED_THRESHOLD]
        else:
            m_pt = comp_pt[comp_cl  >  CHARGED_THRESHOLD]
            u_pt = unmatched_p_pt[unmatched_p_cl >  CHARGED_THRESHOLD]

        all_pt = np.concatenate([m_pt, u_pt])
        is_fake = np.concatenate([np.zeros(len(m_pt)), np.ones(len(u_pt))]).astype(bool)

        centers, fr = _bin_efficiency(all_pt, is_fake, PT_BINS)
        ax.step(np.repeat(PT_BINS, 2)[1:-1], np.repeat(fr, 2),
                color='tomato', linewidth=1.5, where='mid')
        ax.scatter(centers, fr, color='tomato', zorder=5)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'pred $p_T$ [GeV]')
        ax.set_ylabel('fake rate')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fake_rate.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_multiplicity(data, matched, unmatched_t_pt, unmatched_p_pt, output_dir):
    n_truth = np.array([len(x) for x in data['truth_pt']])
    n_pred  = np.array([len(x) for x in data['hgpflow_pt']])

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(0, max(n_truth.max(), n_pred.max()) + 2)
    ax.hist(n_truth, bins=bins, histtype='step', label='truth', color='black', linewidth=1.5)
    ax.hist(n_pred,  bins=bins, histtype='step', label='hgpflow', color='steelblue', linewidth=1.5)
    ax.set_xlabel('particles per event')
    ax.set_ylabel('events')
    ax.set_title(f'Particle multiplicity  (mean truth={n_truth.mean():.1f}, pred={n_pred.mean():.1f})')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, 'multiplicity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_class_breakdown(matched, unmatched_t_cl, unmatched_p_cl, output_dir):
    all_classes = sorted(CLASS_NAMES.keys())
    ref_cl  = matched['ref_cl']
    comp_cl = matched['comp_cl']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Matched particles by class')

    # truth composition of matched pairs
    ax = axes[0]
    counts = [((ref_cl == c).sum()) for c in all_classes]
    labels = [CLASS_NAMES[c] for c in all_classes]
    ax.bar(labels, counts, color='steelblue', alpha=0.8)
    ax.set_title('Truth class of matched pairs')
    ax.set_ylabel('count')
    ax.tick_params(axis='x', rotation=20)

    # unmatched truth (misses) vs unmatched pred (fakes) by class
    ax = axes[1]
    miss_counts = [(unmatched_t_cl == c).sum() for c in all_classes]
    fake_counts = [(unmatched_p_cl == c).sum() for c in all_classes]
    x = np.arange(len(all_classes))
    w = 0.35
    ax.bar(x - w/2, miss_counts, w, label='misses (unmatched truth)', color='goldenrod', alpha=0.8)
    ax.bar(x + w/2, fake_counts, w, label='fakes (unmatched pred)',   color='tomato',   alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title('Misses and fakes by class')
    ax.set_ylabel('count')
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, 'class_breakdown.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(matched, unmatched_t_pt, unmatched_t_cl, unmatched_p_pt, unmatched_p_cl):
    ref_cl  = matched['ref_cl']
    comp_cl = matched['comp_cl']
    residual = (matched['comp_pt'] - matched['ref_pt']) / np.where(matched['ref_pt'] > 0, matched['ref_pt'], np.nan)

    n_matched = len(ref_cl)
    n_missed  = len(unmatched_t_pt)
    n_faked   = len(unmatched_p_pt)
    n_truth_total = n_matched + n_missed
    n_pred_total  = n_matched + n_faked

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f"  Truth particles : {n_truth_total}")
    print(f"  Pred  particles : {n_pred_total}")
    print(f"  Matched pairs   : {n_matched}")
    print(f"  Efficiency      : {n_matched / n_truth_total:.3f}  ({n_missed} misses)")
    print(f"  Fake rate       : {n_faked  / n_pred_total :.3f}  ({n_faked} fakes)")
    print()

    print(f"  pT residual (all matched):")
    print(f"    median = {np.nanmedian(residual):+.4f}")
    print(f"    IQR    = {np.nanpercentile(residual,75)-np.nanpercentile(residual,25):.4f}")

    for label, mask in [('Charged', ref_cl <= CHARGED_THRESHOLD),
                         ('Neutral', ref_cl >  CHARGED_THRESHOLD)]:
        if mask.sum() == 0:
            continue
        r = residual[mask]
        iqr = np.nanpercentile(r, 75) - np.nanpercentile(r, 25)
        print(f"  pT residual ({label}, n={mask.sum()}):")
        print(f"    median = {np.nanmedian(r):+.4f},  IQR = {iqr:.4f}")

    print('='*60 + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', '-p', required=True,
                        help='Path to pred_*_merged.root from hgpflow_v2.eval')
    parser.add_argument('--ind_threshold', '-t', type=float, default=0.48,
                        help='Indicator threshold for selecting candidates (default: 0.48)')
    parser.add_argument('--output_dir', '-o', default='eval_plots',
                        help='Directory to save plots (default: ./eval_plots)')
    parser.add_argument('--no_proxy_overwrite', action='store_true',
                        help='Skip proxy overwrite of charged kinematics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"ind_threshold = {args.ind_threshold}")
    data = load_pred(args.pred_path, args.ind_threshold)

    if not args.no_proxy_overwrite:
        print("Applying proxy overwrite for charged kinematics...")
        data = apply_proxy_overwrite(data)

    matched, unmatched_t_pt, unmatched_t_cl, unmatched_p_pt, unmatched_p_cl = \
        match_all_events(data)

    print_summary(matched, unmatched_t_pt, unmatched_t_cl, unmatched_p_pt, unmatched_p_cl)

    plot_pt_residuals(matched, args.output_dir)
    plot_efficiency(matched, unmatched_t_pt, unmatched_t_cl, args.output_dir)
    plot_fake_rate(matched, unmatched_p_pt, unmatched_p_cl, args.output_dir)
    plot_multiplicity(data, matched, unmatched_t_pt, unmatched_p_pt, args.output_dir)
    plot_class_breakdown(matched, unmatched_t_cl, unmatched_p_cl, args.output_dir)

    print(f"All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

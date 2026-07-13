"""
Quantify the energy-weighted cost of missed neutral particles in stage 1.

For real neutral particles (truth indicator > 0.5, not charged), a "miss" is a
predicted indicator below threshold: the particle's entire deposited energy is
lost to downstream reconstruction. This script reports the count-based and
energy-weighted miss rates, overall and binned in truth deposited energy --
i.e. whether the y=-x streak in the tr>0.5;pr<0.5 panel is cosmetic or costly.

Usage:
    python quantify_missed_neutrals.py \
        --run_dir /pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largeqqbartraining4nodev1epoch25 \
        --files "glob.glob('/pscratch/sd/a/agolub/qqbar_events/split_val/*.root')" \
        --n_events 20000
"""
import argparse
import glob
import os
import re
from collections import OrderedDict

import numpy as np
import torch
import yaml


def pick_checkpoint(run_dir):
    """Best epoch=*-val_total_loss=*.ckpt (lowest loss), else last.ckpt."""
    ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', 'epoch=*val_total_loss=*.ckpt'))
    if ckpts:
        def loss_of(p):
            m = re.search(r'val_total_loss=([0-9.]+?)\.ckpt', os.path.basename(p))
            return float(m.group(1)) if m else float('inf')
        return min(ckpts, key=loss_of)
    return os.path.join(run_dir, 'checkpoints', 'last.ckpt')


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='/pscratch/sd/a/agolub/hgpflow_runs/hgpflow_v2/largeqqbartraining4nodev1epoch25',
                    help='training run dir with archived config_v/ms1/t + checkpoints/')
    ap.add_argument('--ckpt', default=None, help='checkpoint path (default: best in run_dir)')
    ap.add_argument('--files', default="glob.glob('/pscratch/sd/a/agolub/qqbar_events/split_val/*.root')",
                    help='file list / glob expression for the dataset')
    ap.add_argument('--n_events', type=int, default=20000, help='examples to evaluate (-1 = all)')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument('--cpu', action='store_true', help='force CPU')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device: {device}')

    config_v = yaml.safe_load(open(os.path.join(args.run_dir, 'config_v.yml')))
    config_ms1 = yaml.safe_load(open(os.path.join(args.run_dir, 'config_ms1.yml')))
    config_t = yaml.safe_load(open(os.path.join(args.run_dir, 'config_t.yml')))
    config_t['device'] = str(device.type)
    config_t['lsa_num_threads'] = 1

    # ── model ────────────────────────────────────────────────────────────
    from hgpflow_v2.models.hgpflow_model import HGPFlowModel
    from hgpflow_v2.utility.helper_dicts import class_mass_dict
    from hgpflow_v2.utility.metrics import Metrics
    from hgpflow_v2.dataset.dataset import get_dataloader

    ckpt_path = args.ckpt or pick_checkpoint(args.run_dir)
    print(f'checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = OrderedDict((k[4:], v) for k, v in ckpt['state_dict'].items() if k.startswith('net.'))

    model = HGPFlowModel(config_v, config_ms1, None, class_mass_dict)
    model.load_state_dict(state)
    model.eval().to(device)

    metrics = Metrics(config_t, 0)

    # ── data (exact-node-count batches: no padding, no node_valid needed) ─
    ds_kwargs = {'filename': args.files, 'config_v': config_v,
                 'reduce_ds': args.n_events, 'compute_incidence': True}
    sampler_kwargs = {'config_v': config_v, 'batch_size': args.batch_size,
                      'remove_idxs': True, 'apply_cells_threshold': False,
                      'n_cells_threshold': 10_000}
    loader = get_dataloader(config_v['dataset_type'], ds_kwargs, sampler_kwargs,
                            {'num_workers': 4, 'pin_memory': device.type == 'cuda'})

    # ── inference + matching ─────────────────────────────────────────────
    truth_dep_all, pred_ind_all, keep_all = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = to_device(batch, device)

            node_feat = model.node_prep_model(batch)
            preds_list, _ = model.hg_model(node_feat, batch['node']['is_track'].squeeze(-1).bool())
            pred_inc, pred_ind_logit, pred_is_charged = preds_list[-1][-1]

            # Hungarian match to order predicted slots against truth slots
            _, _, indices = metrics.LAP_loss_single(
                (pred_inc, pred_ind_logit, pred_is_charged),
                (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
                node_is_track=batch['node']['is_track'])

            pred_ind = torch.sigmoid(pred_ind_logit).squeeze(-1)
            pred_ind_ordered = torch.gather(pred_ind, 1, indices)

            topo_e_raw = batch['topo']['e_raw'].unsqueeze(1)
            truth_dep = (batch['incidence_truth'] * topo_e_raw).sum(2)  # (b, ne)

            # real neutral particles only
            keep = (batch['indicator_truth'] > 0.5) & (~batch['particle']['is_charged'].bool())

            truth_dep_all.append(truth_dep[keep].cpu().numpy())
            pred_ind_all.append(pred_ind_ordered[keep].cpu().numpy())

            if (bi + 1) % 20 == 0:
                print(f'  {bi+1} batches...')

    truth_dep = np.concatenate(truth_dep_all)
    pred_ind = np.concatenate(pred_ind_all)
    total_e = truth_dep.sum()
    n = len(truth_dep)
    print(f'\nreal neutral particles: {n}   total truth dep E: {total_e:.1f} GeV')
    print(f'dep E percentiles [50/90/99/max]: '
          f'{np.percentile(truth_dep, 50):.2f} / {np.percentile(truth_dep, 90):.2f} / '
          f'{np.percentile(truth_dep, 99):.2f} / {truth_dep.max():.1f} GeV')

    # ── overall miss rates ───────────────────────────────────────────────
    print(f'\n{"threshold":>10} {"miss rate (count)":>18} {"miss rate (energy)":>19}')
    for th in args.thresholds:
        missed = pred_ind < th
        print(f'{th:>10.2f} {missed.mean():>17.3%} {truth_dep[missed].sum()/total_e:>18.3%}')

    # ── energy-binned miss rates at each threshold ───────────────────────
    bins = np.array([0, 0.5, 1, 2, 5, 10, 20, 50, np.inf])
    for th in args.thresholds:
        missed = pred_ind < th
        print(f'\n-- threshold {th} --')
        print(f'{"E bin [GeV]":>14} {"N":>9} {"miss (count)":>13} {"miss (energy)":>14} {"E lost [GeV]":>13}')
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (truth_dep >= lo) & (truth_dep < hi)
            if m.sum() == 0:
                continue
            e_bin = truth_dep[m].sum()
            e_lost = truth_dep[m & missed].sum()
            label = f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>{lo:g}'
            print(f'{label:>14} {m.sum():>9d} {missed[m].mean():>12.3%} '
                  f'{e_lost/e_bin:>13.3%} {e_lost:>13.1f}')

    print('\nInterpretation: the "miss rate (energy)" column at your operating threshold '
          'is the fraction of neutral calorimeter energy stage 2 will never see. '
          'A few % total, concentrated in the lowest bins, is cosmetic; '
          'double-digit loss in the >5 GeV bins is worth attention.')


if __name__ == '__main__':
    main()

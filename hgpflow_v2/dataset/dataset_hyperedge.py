import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import gc
import glob


class PflowDatasetHyperedge(Dataset):

    def __init__(self, filename, ind_threshold, reduce_ds=-1, fix_candidate_context=False):
        '''
        fix_candidate_context=False reproduces the published (paper) behaviour exactly:
            dataset length = segment count (only the first n_segments flattened
            candidates are ever sampled) and __getitem__ pairs candidate idx with
            node_feat_sum of SEGMENT idx (a different index space -> misaligned).
        fix_candidate_context=True indexes over ALL candidates and pairs each with
            its own segment's node_feat_sum via a candidate->segment map.
        Keep False for paper-faithful replication; True for the corrected variant (A/B).
        '''
        super().__init__()

        self.filename = filename
        self.ind_threshold = ind_threshold
        self.reduce_ds = reduce_ds
        self.fix_candidate_context = fix_candidate_context
        assert reduce_ds == -1 or (isinstance(reduce_ds, int) and reduce_ds > 0), \
            "reduce_ds must be -1 or a positive integer"

        if not isinstance(filename, list):
            if ('[' in filename and ']' in filename) or ('glob.glob' in filename):
                filename = eval(filename)
            else:
                filename = [filename]

        self.branches_to_read = [
            'ch_proxy_kin', 'neut_proxy_kin', 'proxy_is_charged',
            'e_t', 'inc_times_node_feat', 'node_feat_sum', 'proxy_em_frac', 'pred_ind',
            'truth_pt', 'truth_eta', 'truth_phi', 'truth_ke', 'truth_class', 'truth_is_charged', 'truth_ind',
            'truth_pt_raw', 'truth_eta_raw', 'truth_ke_raw', 'proxy_pt_raw', 'proxy_ke_raw', 'proxy_eta_raw', 'proxy_phi'
        ]

        self.vars_to_reshape = ['ch_proxy_kin', 'neut_proxy_kin', 'e_t', 'inc_times_node_feat']

        # Per-file pipeline: read -> reshape -> mask -> flatten -> numpy, one file at a
        # time. The old order (concatenate ALL files as awkward arrays, then reshape/
        # mask/flatten each with a full copy) peaked at ~3-4x the final size and
        # OOM-killed multi-rank runs on the full 305-file dataset.
        chunks = {var: [] for var in self.branches_to_read}
        seg_idx_chunks = []
        n_segments = 0
        for fn_i, fn in enumerate(filename):
            f = uproot.open(fn)
            tree = f['event_tree']

            n_events_fni = tree.num_entries
            if (reduce_ds != -1):
                n_events_fni = min(n_events_fni, reduce_ds)
                reduce_ds -= n_events_fni
            entry_stop = n_events_fni

            file_dict = {}
            for var in tqdm(self.branches_to_read, desc=f'reading file {fn_i+1}/{len(filename)}'):
                file_dict[var] = tree[var].array(library='ak', entry_start=0, entry_stop=entry_stop)

            # reshape flat per-event vectors -> (n_slots, dim)
            _n_particles = ak.to_numpy(ak.count(file_dict['truth_pt'], axis=1))
            for k in self.vars_to_reshape:
                _dim = len(file_dict[k][0]) // _n_particles[0]
                file_dict[k] = ak.unflatten(file_dict[k], _dim, axis=1)

            # pred-indicator mask (node_feat_sum stays per segment)
            ind_mask = file_dict['pred_ind'] > self.ind_threshold
            for k in self.branches_to_read:
                if k != 'node_feat_sum':
                    file_dict[k] = file_dict[k][ind_mask]

            # candidate -> segment map for this file (global segment ids)
            n_cand_per_seg = ak.to_numpy(ak.num(file_dict['pred_ind'], axis=1))
            seg_idx_chunks.append(
                np.repeat(np.arange(len(n_cand_per_seg)) + n_segments, n_cand_per_seg))
            n_segments += len(n_cand_per_seg)

            # flatten to the final numpy form
            for k in self.branches_to_read:
                if k == 'node_feat_sum':
                    chunks[k].append(ak.to_numpy(file_dict[k]))
                else:
                    chunks[k].append(ak.to_numpy(ak.flatten(file_dict[k], axis=1)))
            del file_dict
            gc.collect()

            if reduce_ds == 0:
                break

        # concatenate one key at a time, freeing chunks as we go: keeps the peak at
        # ~(final size + one array) instead of 2x final (chunks + result all alive)
        self.data_dict = {}
        for k in list(chunks.keys()):
            self.data_dict[k] = np.concatenate(chunks.pop(k), axis=0)
            gc.collect()
        self.seg_idx = np.concatenate(seg_idx_chunks)

        if self.fix_candidate_context:
            self.n_events = len(self.data_dict['pred_ind'])  # all candidates
            assert len(self.seg_idx) == self.n_events
        else:
            self.n_events = len(self.data_dict['node_feat_sum'])  # historical: segment count

        gc.collect()
        print(f'\ndataset loaded. Number of examples: {self.n_events} '
              f'(fix_candidate_context={self.fix_candidate_context}; '
              f'{len(self.seg_idx)} candidates / {n_segments} segments)\n')


    def __getitem__(self, idx):
        return_dict = {}
        for k, v in self.data_dict.items():
            if k == 'node_feat_sum' and self.fix_candidate_context:
                return_dict[k] = torch.tensor(v[self.seg_idx[idx]])  # own segment's context
            else:
                return_dict[k] = torch.tensor(v[idx])
        return return_dict


    def __len__(self):
        return self.n_events
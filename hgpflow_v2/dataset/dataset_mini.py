import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm
import glob

import torch
from torch.utils.data import Dataset, Sampler

from ..utility.var_transformation import VarTransformation 
from ..utility.helper_dicts import pdgid_class_dict


class PflowDatasetMini(Dataset):

    def __init__(self, filename, config_v, reduce_ds=-1, compute_incidence=True,
                 shard_files_by_rank=False):
        '''
        Args:
            filename: str | list of str | str to eval to get list of str
            shard_files_by_rank: under DDP, load only this rank's slice of the file
                list (1/world of the load time and RAM). The batch sampler must then
                skip rank-striding since the data itself is disjoint (it reads the
                resulting `files_sharded` attribute via get_dataloader).
        '''

        super().__init__()
	    
        self.config_v = config_v
        self.reduce_ds = reduce_ds
        assert reduce_ds == -1 or (isinstance(reduce_ds, int) and reduce_ds > 0), \
            "reduce_ds must be -1 or a positive integer"
        self.compute_incidence = compute_incidence
        self.max_particles = config_v['max_particles']

        self.init_var_list()
        self.pdgid_to_class = pdgid_class_dict

        if not isinstance(filename, list):
            if ('[' in filename and ']' in filename) or ('glob.glob' in filename):
                filename = eval(filename)
            else:
                filename = [filename]

        self.files_sharded = False
        if shard_files_by_rank and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            if world > 1 and len(filename) >= world:
                filename = sorted(filename)[rank::world]  # sorted: identical order on every rank
                self.files_sharded = True
                if reduce_ds > 0:  # interpret reduce_ds as a GLOBAL cap -> split across ranks
                    reduce_ds = max(1, reduce_ds // world)
                print(f'[rank {rank}] file-sharded dataset: {len(filename)} files, reduce_ds={reduce_ds}')
            elif world > 1:
                print(f'\033[96m[rank {rank}] only {len(filename)} file(s) for {world} ranks; '
                      f'loading all files, sampler will stride batches instead\033[0m')

        self.data_dict = {}; self.n_events = 0
        for fn_i, fn in enumerate(filename):
            f = uproot.open(fn)
            tree = f['EventTree']

            n_events_fni = tree.num_entries
            if (reduce_ds != -1):
                n_events_fni = min(n_events_fni, reduce_ds)
                reduce_ds -= n_events_fni

            self.n_events += n_events_fni
            entry_stop = n_events_fni

            # read the data
            for var in tqdm(self.branches_to_read, desc=f'reading file {fn_i+1}/{len(filename)}'):
                v_name = self.branches_rename[var] if var in self.branches_rename else var
                if fn_i == 0:
                    self.data_dict[v_name] = []

                branch_array = tree[var].array(library='ak', entry_start=0, entry_stop=entry_stop)
                self.data_dict[v_name].append(branch_array)

            if reduce_ds == 0:
                break

        # stack the data (merging)
        for v_name in self.data_dict.keys():
            self.data_dict[v_name] = ak.concatenate(self.data_dict[v_name], axis=0)

        # create the transform dict
        self.transform_dicts = {}
        for k, v in config_v['transformation_dict'].items():
            self.transform_dicts[k] = VarTransformation(v)

        # process data
        self.process_data()

        # NaN check
        for k, v in self.data_dict.items():
            if np.sum(np.isnan(v)) > 0:
                # print in red
                print(f'\033[91m{k} has {np.sum(np.isnan(v))} NaNs\033[0m')

		# needed for batch sampling
        self.n_tracks = ak.to_numpy(ak.count(self.data_dict['track_eta'], axis=1))
        self.n_topos = ak.to_numpy(ak.count(self.data_dict['topo_eta'], axis=1))
        self.n_nodes = self.n_tracks + self.n_topos
        self.n_particles = ak.to_numpy(ak.count(self.data_dict['particle_pt'], axis=1))

        print(f'\ndataset loaded. Number of examples: {self.n_events}\n')


    def init_var_list(self):
        self.branches_to_read = []
        for k, v in self.config_v['data_loading']['branches_to_read'].items():
            self.branches_to_read += v
        self.branches_rename = self.config_v['data_loading']['branches_rename']

        self.track_feat0_vars = self.config_v['features']['track_feat0_vars']
        self.topo_feat0_vars  = self.config_v['features']['topo_feat0_vars']
        self.track_skip0_vars = self.config_v['features']['node_skip_feat0_vars']['track']
        self.topo_skip0_vars  = self.config_v['features']['node_skip_feat0_vars']['topo']

        self.particle_vars = self.config_v['getitem_retun']['particle_vars']
        self.event_vars = self.config_v['getitem_retun']['event_vars']
        self.additional_vars = self.config_v['getitem_retun']['additional_vars']

    def process_data(self):

        # adding sin and cos of phis
        tqdm_obj = tqdm(self.config_v['data_processing']['vars_sin_cos'], desc='adding sin-cos phi')
        for var in tqdm_obj:
            new_var = var.replace('phi', 'cosphi')
            self.data_dict[new_var] = np.cos(self.data_dict[var])

            new_var = var.replace('phi', 'sinphi')
            self.data_dict[new_var] = np.sin(self.data_dict[var])

            if var in self.config_v['data_processing']['vars_sin_cos_og_delete']:
                del self.data_dict[var]

        # transformations
        tqdm_obj = tqdm(self.config_v['data_processing']['vars_to_transform'].items(), desc='transforming vars')
        for var, trans_name in tqdm_obj:
            new_name = var.replace('_raw', '')
            self.data_dict[new_name] = self.transform_dicts[trans_name].forward(self.data_dict[var])


        # # particle class labels
        # self.data_dict['particle_class'] = []
        # for idx in tqdm(range(self.n_events), desc='particle class'):

        #     particle_class_ev = np.array([self.pdgid_to_class[x] for x in self.data_dict['particle_pdgid'][idx]])

        #     track_particle_idx_ev = self.data_dict['track_particle_idx'][idx]
        #     track_particle_idx_ev = track_particle_idx_ev[track_particle_idx_ev >= 0] # in inference, we have -9999 for no associated particle
        #     trackless_particle_mask_ev = np.ones_like(particle_class_ev, dtype=np.bool)
        #     trackless_particle_mask_ev[track_particle_idx_ev] = False

        #     trackless_chhad_and_e_mask_ev = trackless_particle_mask_ev & (particle_class_ev <= 1)
        #     trackless_muon_mask_ev = trackless_particle_mask_ev & (particle_class_ev == 2)

        #     # trackless ch hads and es become nu hads and photons (+3)
        #     particle_class_ev[trackless_chhad_and_e_mask_ev] += 3

        #     # trackless muons become neutral hadrons
        #     particle_class_ev[trackless_muon_mask_ev] = 3

        #     self.data_dict['particle_class'].append(particle_class_ev)
        # # self.data_dict['particle_class'] = np.array(self.data_dict['_particle_class'], dtype=object)
        # self.data_dict['particle_class'] = ak.Array(self.data_dict['particle_class'])





        flat_particle_pdgid = ak.flatten(self.data_dict['particle_pdgid'])
        flat_particle_class = ak.Array([self.pdgid_to_class[x] for x in flat_particle_pdgid])

        flat_particle_track_idx = ak.flatten(self.data_dict['particle_track_idx'])
        flat_trackless_particle_mask = ak.where(flat_particle_track_idx >= 0, False, True)

        # trackless ch hads and es become nu hads and photons (+3)
        flat_trackless_chhad_and_e_mask = flat_trackless_particle_mask & (flat_particle_class <= 1)
        flat_particle_class = ak.where(flat_trackless_chhad_and_e_mask, flat_particle_class + 3, flat_particle_class)

        # trackless muons become neutral hadrons
        flat_trackless_muon_mask = flat_trackless_particle_mask & (flat_particle_class == 2)
        flat_particle_class = ak.where(flat_trackless_muon_mask, 3, flat_particle_class)

        # Unflatten the arrays back to their original structure
        particle_class = ak.unflatten(flat_particle_class, ak.num(self.data_dict['particle_pdgid']))
        self.data_dict['particle_class'] = particle_class




        # em fraction
        if 'topo_em_frac' not in self.data_dict:
            self.data_dict['topo_em_frac'] = self.data_dict['topo_ecal_e_raw'] / \
                (self.data_dict['topo_ecal_e_raw'] + self.data_dict['topo_hcal_e_raw'] + 1e-8)


    def get_incidence_matrix(self, idx):
        incidence_matrix = np.zeros((self.max_particles, self.n_nodes[idx]))

        # add tracks
        part_idxs = self.data_dict['track_particle_idx'][idx]
        track_idxs = np.arange(self.n_tracks[idx])
        incidence_matrix[part_idxs, track_idxs] = 1.0

        # add topos
        topo_idxs = self.data_dict['topo2particle_topo_idx'][idx]
        part_idxs = self.data_dict['topo2particle_particle_idx'][idx]
        part_es = self.data_dict['topo2particle_energy'][idx]
        incidence_matrix[part_idxs, topo_idxs + self.n_tracks[idx]] = part_es

        # check for TC w/o associated particles
        if (incidence_matrix.sum(axis=0) == 0).any():
            noisy_cols = np.where(incidence_matrix.sum(axis=0) == 0)[0]
            fake_rows  = np.arange(len(noisy_cols)) + self.n_particles[idx]

            # check for indices greater than config_v['max_particles']
            if not (fake_rows < self.max_particles).all():
                print(f'Warning: fake_rows go beyond maximum ({self.max_particles}) particles in event {self.data_dict["event_number"][idx]}. Dropping them!')
                noisy_cols = noisy_cols[fake_rows < self.max_particles]
                fake_rows = fake_rows[fake_rows < self.max_particles]
            incidence_matrix[fake_rows, noisy_cols] = 1.0

        # normalize
        incidence_matrix = incidence_matrix / np.clip(incidence_matrix.sum(axis=0, keepdims=True), a_min=1e-6, a_max=None)

        return incidence_matrix


    def __getitem__(self, idx):

        # tracks feat0
        track_feat0 = []; track_dict = {}
        for var in self.track_feat0_vars:
            v_name = 'track_' + var
            track_feat0.append(torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx])))
        track_dict['feat0'] = torch.stack(track_feat0, dim=-1)

        # topos feat0
        topo_feat0 = []; topo_dict = {}
        for var in self.topo_feat0_vars:
            v_name = 'topo_' + var
            topo_feat0.append(torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx])))
        topo_dict['feat0'] = torch.stack(topo_feat0, dim=-1).float() # float conversion (should take care of it in data processing)

        # node skip
        node_dict = {}
        n_skip_feat0 = len(self.config_v['features']['node_skip_feat0_vars']['track'])
        node_skip_feat0 = torch.zeros(self.n_nodes[idx], n_skip_feat0, dtype=torch.float32)
        for vi, var in enumerate(self.track_skip0_vars):
            if var != None:
                v_name = 'track_' + var
                node_skip_feat0[:self.n_tracks[idx], vi] = torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx]))
        for vi, var in enumerate(self.topo_skip0_vars):
            if var != None:
                v_name = 'topo_' + var
                node_skip_feat0[self.n_tracks[idx]:, vi] = torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx]))
        node_dict['skip_feat0'] = node_skip_feat0

        # particles (zero padded to be of max_particles length)
        particle_dict = {}
        for var in self.particle_vars:
            v_name = 'particle_' + var
            value = torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx]))
            particle_dict[var] = torch.zeros(self.max_particles, dtype=value.dtype)
            if 'class' in var:
                particle_dict[var] += 99 # making sure the code crashes if we try to learn these
            particle_dict[var][:len(value)] = value

        particle_dict['mask'] = torch.zeros(self.max_particles).bool()
        particle_dict['mask'][:len(value)] = True
        particle_dict['is_charged'] = (particle_dict['class'] <= 2).bool() * particle_dict['mask']

        # incidence matrix (self.max_particles, n_nodes), indicator
        incidence_matrix = None
        if self.compute_incidence:
            incidence_matrix = self.get_incidence_matrix(idx)
            incidence_matrix = torch.from_numpy(incidence_matrix).float()

        indicator = torch.zeros(self.max_particles)
        indicator[particle_dict['class'] <= 4] = 1.0

        # additional vars added to track, cell, topo
        for var in self.additional_vars['track']:
            v_name = 'track_' + var
            track_dict[var] = torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx]))
        for var in self.additional_vars['topo']:
            v_name = 'topo_' + var
            topo_dict[var] = torch.from_numpy(ak.to_numpy(self.data_dict[v_name][idx]))


        return_dict = {
            'track_dict': track_dict,
            'topo_dict': topo_dict,
            'node_dict': node_dict,
            'particle_dict': particle_dict,
            'incidence': incidence_matrix,
            'indicator': indicator,
            'idx': idx,
            'event_number': self.data_dict['event_number'][idx]
        }

        return return_dict


    def __len__(self):
        return self.n_events


    

def collate_fn_mini(samples):

    bs = len(samples)

    # num node computation
    sample0 = samples[0]
    batch_num_tracks = [len(x['track_dict']['feat0']) for x in samples]
    batch_num_topos = [len(x['topo_dict']['feat0']) for x in samples]
    batch_num_nodes = [x + y for x, y in zip(batch_num_tracks, batch_num_topos)]
    max_num_nodes = max(batch_num_nodes)

    # mixed node counts are allowed: shorter samples are zero-padded in the middle
    # (tracks kept at the front, topos at the back) and marked via is_track/is_topo.

    # create batched dicts with zeros
    batched_track_dict = {}
    for name, value in sample0['track_dict'].items():
        batched_track_dict[name] = torch.zeros(bs, max_num_nodes, *value.shape[1:], dtype=value.dtype)

    batched_topo_dict = {}
    for name, value in sample0['topo_dict'].items():
        batched_topo_dict[name] = torch.zeros(bs, max_num_nodes, *value.shape[1:], dtype=value.dtype)

    batched_node_dict = {}
    batched_node_dict['is_track'] = torch.zeros(bs, max_num_nodes, dtype=torch.bool)
    batched_node_dict['is_topo'] = torch.zeros(bs, max_num_nodes, dtype=torch.bool)

    # masks
    batched_track_dict['mask'] = torch.zeros(bs, max_num_nodes, dtype=torch.bool)
    batched_topo_dict['mask'] = torch.zeros(bs, max_num_nodes, dtype=torch.bool)

    # fill in the values (track, cell, topo, edges)
    for i, sample in enumerate(samples):
        if len(sample['track_dict']['feat0']) > 0:
            for name, value in sample['track_dict'].items():
                batched_track_dict[name][i, :batch_num_tracks[i]] = value
            batched_track_dict['mask'][i, :batch_num_tracks[i]] = True

        if len(sample['topo_dict']['e_raw']) > 0:
            for name, value in sample['topo_dict'].items():
                batched_topo_dict[name][i, -batch_num_topos[i]:] = value
            batched_topo_dict['mask'][i, -batch_num_topos[i]:] = True

        batched_node_dict['is_track'][i, :batch_num_tracks[i]] = True
        batched_node_dict['is_topo'][i, -batch_num_topos[i]:] = True

    # batched node feat (per-node -> node-dim padded: tracks at front, topos at back)
    for name in sample0['node_dict'].keys():
        feat_shape = sample0['node_dict'][name].shape[1:]
        batched = torch.zeros(bs, max_num_nodes, *feat_shape, dtype=sample0['node_dict'][name].dtype)
        for i, sample in enumerate(samples):
            val = sample['node_dict'][name]  # (n_tracks_i + n_topos_i, *feat)
            nt, ntp = batch_num_tracks[i], batch_num_topos[i]
            if nt > 0:
                batched[i, :nt] = val[:nt]
            if ntp > 0:
                batched[i, max_num_nodes - ntp:] = val[nt:nt + ntp]
        batched_node_dict[name] = batched

    # batched particle feat
    batched_particle_dict = {}
    for name in sample0['particle_dict'].keys():
        batched_particle_dict[name] = torch.stack([x['particle_dict'][name] for x in samples])

    # incidence: pad each (max_particles, n_nodes) matrix to max_num_nodes, aligning
    # columns with the node layout (tracks at front, topos at back, padding in middle).
    incidence_truth = None
    if samples[0]['incidence'] is not None:
        max_particles = samples[0]['incidence'].shape[0]
        incidence_truth = torch.zeros(
            bs, max_particles, max_num_nodes, dtype=samples[0]['incidence'].dtype)
        for i, sample in enumerate(samples):
            inc = sample['incidence']  # (max_particles, n_tracks_i + n_topos_i)
            nt, ntp = batch_num_tracks[i], batch_num_topos[i]
            if nt > 0:
                incidence_truth[i, :, :nt] = inc[:, :nt]
            if ntp > 0:
                incidence_truth[i, :, max_num_nodes - ntp:] = inc[:, nt:nt + ntp]
    indicator_truth = torch.stack([x['indicator'] for x in samples]) # .bool()

    batched_dict = {
        'track': batched_track_dict,
        'topo': batched_topo_dict,
        'node': batched_node_dict,
        
        'particle': batched_particle_dict,
        'incidence_truth': incidence_truth,
        'indicator_truth': indicator_truth,
        'idx': np.array([x['idx'] for x in samples]),
        'event_number': np.array([x['event_number'] for x in samples])
    }

    return batched_dict



class PflowSamplerMini(Sampler):
    def __init__(self, n_nodes_array, batch_size, remove_idxs=[], length_grouped=False,
                 data_already_sharded=False):
        """
        :param n_nodes_array: array of the number of nodes (tracks + topos)
        :param batch_size: batch size
        :param length_grouped: if False (default), each batch holds examples with an
            identical node count (original behaviour). If True, examples are sorted by
            node count and chunked into batch_size groups so batches fill to batch_size
            with only small padding -- lifts the effective-batch cap (needs the model /
            loss node masking). Meant for training; inference keeps the default.
        :param data_already_sharded: the dataset holds a per-rank file shard, so do NOT
            stride batches across ranks; instead truncate every rank to the global
            minimum batch count (one collective here) so DDP steps stay in lock-step.
        """
        super().__init__()

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.drop_last = False

        self.index_to_batch = {}
        running_idx = -1

        if length_grouped:
            # sort by node count, chunk into batch_size groups (minimal padding, full batches)
            valid_idx = np.setdiff1d(np.arange(len(n_nodes_array)), remove_idxs)
            order = valid_idx[np.argsort(n_nodes_array[valid_idx], kind='stable')]
            for start in range(0, len(order), self.batch_size):
                running_idx += 1
                self.index_to_batch[running_idx] = order[start:start + self.batch_size]
        else:
            # original: one exact node count per batch
            for n_nodes_i in set(n_nodes_array):
                idxs = np.setdiff1d(np.where(n_nodes_array == n_nodes_i)[0], remove_idxs)
                for i in range(0, len(idxs), self.batch_size):
                    running_idx += 1
                    self.index_to_batch[running_idx] = idxs[i:i + self.batch_size]

        self.n_batches = running_idx + 1

        # data_already_sharded: equalize the per-rank batch count once, up front
        # (uneven counts would deadlock DDP at the end of the epoch)
        self.data_already_sharded = data_already_sharded
        self._n_batches_equal = self.n_batches
        if data_already_sharded:
            rank, world_size = self._dist_info()
            if world_size > 1:
                counts = [None] * world_size
                torch.distributed.all_gather_object(counts, self.n_batches)
                self._n_batches_equal = min(counts)
                if rank == 0 and self._n_batches_equal < max(counts):
                    print(f'[sampler] equalized batches/rank to {self._n_batches_equal} '
                          f'(per-rank counts: {counts})')

    @staticmethod
    def _dist_info():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __len__(self):
        if self.data_already_sharded:
            return self._n_batches_equal
        _, world_size = self._dist_info()
        if world_size > 1:
            return self.n_batches // world_size
        return self.n_batches

    def __iter__(self):
        # epoch-seeded so every DDP rank generates the SAME permutation; ranks then
        # take disjoint strides of it. (Also makes single-GPU batch order reproducible.)
        self._epoch = getattr(self, '_epoch', -1) + 1
        rng = np.random.default_rng(12345 + self._epoch)
        batch_order = rng.permutation(np.arange(self.n_batches))

        rank, world_size = self._dist_info()
        if self.data_already_sharded:
            # data is disjoint per rank already; just hold every rank to the same count
            batch_order = batch_order[:self._n_batches_equal]
        elif world_size > 1:
            # equal batch count per rank (drop the remainder) so DDP doesn't deadlock
            n_even = (len(batch_order) // world_size) * world_size
            batch_order = batch_order[:n_even][rank::world_size]

        if rank == 0:
            # write to single np file both batch_order and self.index_to_batch, will load them in python later
            np.savez('batch_order.npz', batch_order=batch_order, index_to_batch=self.index_to_batch)

        for i in batch_order:
            yield self.index_to_batch[i]

import torch
import time
import random
from contextlib import contextmanager
from ..utility import misc
from ..dataset.dataset import get_dataloader

# Miscellaneous
from numpy.random import default_rng
SEED = 123456
RNG = default_rng(SEED)


class HGPFLightningIR:
    def __init__(self, parent_lightning, config_v, config_ms1, config_ms2, config_t):        
        self.config_v = config_v
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2 # this is None
        self.config_t = config_t
        self._parent = parent_lightning

        self.T_TOTAL = self.config_ms1['hg_model']['T_TOTAL']
        self.T_BPTT  = self.config_ms1['hg_model']['T_BPTT']
        self.N_BPTT  = self.config_ms1['hg_model']['N_BPTT']

        self._parent.automatic_optimization = False
        self.random_bptt = config_t.get('random_bptt', False)
        if not self.random_bptt:
            self.sampler = misc.IntegerPartitionSampler(self.T_TOTAL-self.T_BPTT*self.N_BPTT, self.N_BPTT, RNG)


    def train_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_train'],
            'config_v': self.config_v,
            'reduce_ds': self.config_t['reduce_ds_train'],
            'compute_incidence': True,
            'shard_files_by_rank': self.config_t.get('shard_files_by_rank', True)}
        
        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.config_t['batchsize_train'],
            'remove_idxs': True,
            'apply_cells_threshold': self.config_t.get('apply_cells_threshold', False),
            'n_cells_threshold': self.config_t.get('n_cells_threshold', 10_000),
            'length_grouped': self.config_t.get('length_grouped_batching', False)}

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def val_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_val'],
            'config_v': self.config_v,
            'reduce_ds': self.config_t['reduce_ds_val'],
            'compute_incidence': True,
            'shard_files_by_rank': self.config_t.get('shard_files_by_rank', True)}

        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.config_t['batchsize_val'],
            'remove_idxs': True,
            'apply_cells_threshold': self.config_t.get('apply_cells_threshold', False),
            'n_cells_threshold': self.config_t.get('n_cells_threshold', 10_000)}

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def get_t_backprops(self, last_only=False):
        if last_only:
            return [False] * (self.T_TOTAL - 1) + [True]

        if self.random_bptt:
            bptt_list = [False] * (self.T_TOTAL - self.T_BPTT) + [True] * (self.T_BPTT - 1)
            random.shuffle(bptt_list)
            bptt_list.append(True)
            return [bptt_list]

        else:
            t_pre = self.sampler()
            bptt_lists = []
            for t in t_pre:
                bptt_list = [False] * t + [True] * (self.T_BPTT)
                bptt_lists.append(bptt_list)
            return bptt_lists

    
    @contextmanager
    def _timed(self, name):
        """Wall-clock timer that brackets a region with cuda.synchronize so GPU
        (async) work is attributed correctly. No-op unless time_training_step is set."""
        if not self.config_t.get('time_training_step', False):
            yield
            return
        if not hasattr(self, '_timers'):
            self._timers = {}
        sync = torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if sync:
            torch.cuda.synchronize()
        self._timers[name] = self._timers.get(name, 0.0) + (time.perf_counter() - t0)

    def _report_timers(self, batch_idx, every=20):
        if not self.config_t.get('time_training_step', False):
            return
        if getattr(getattr(self._parent, 'trainer', None), 'global_rank', 0) != 0:
            return
        if (batch_idx + 1) % every != 0 or not getattr(self, '_timers', None):
            return
        total = sum(self._timers.values())
        print(f"\n[timers] after {batch_idx+1} steps  (total timed {total:.2f}s)")
        for k, v in sorted(self._timers.items(), key=lambda x: -x[1]):
            print(f"    {k:18s}: {v:7.3f}s  ({100*v/total:4.1f}%)  {1000*v/(batch_idx+1):6.1f} ms/step")

    def training_step(self, batch, batch_idx):

        bs = batch['incidence_truth'].size(0)
        opt = self._parent.optimizers()
        opt.zero_grad()

        # full-model grad-norm is only used as a logged metric; compute it only on log steps
        self._parent._grad_norm_due = (self._parent.comet_logger is not None) and \
            (batch_idx % self.config_t.get('train_log_every_n_steps', 1) == 0)

        node_valid = (batch['node']['is_track'] | batch['node']['is_topo']).bool()

        with self._timed('fwd_setup'):
            node_feat = self._parent.net.node_prep_model(batch)
            e_t, v_t, i_t, track_eye, ch_mask_from_tracks = self._parent.net.hg_model.model.get_initial(
                node_feat, batch['node']['is_track'].bool(), node_valid=node_valid)

        if 'hg_model' in self.config_t['train_components']:
            bptt_lists = self.get_t_backprops()

            loss_per_upd = []; loss_comps = []
            for t, bptt_list in enumerate(bptt_lists):
                with self._timed('refine_fwd'):
                    preds_list, (e_t, v_t, i_t) = self._parent.net.hg_model.model.refine(
                        node_feat, e_t, v_t, i_t, batch['node']['is_track'].squeeze(-1).bool(),
                        track_eye, ch_mask_from_tracks, t_backprops=bptt_list, node_valid=node_valid)

                with self._timed('lap_match'):
                    loss, loss_components, _ = self._parent.metrics.LAP_loss_multi(
                        preds_list,
                        (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
                        node_is_track=batch['node']['is_track'], node_valid=node_valid)

                with self._timed('backward_opt'):
                    self._parent.manual_backward(loss)
                    if not self._parent.automatic_optimization:
                        torch.nn.utils.clip_grad_norm_(self._parent.net.parameters(), 1.0)

                    opt.step()
                    opt.zero_grad()

                loss_per_upd.append(loss.detach().cpu().numpy())
                loss_comps.append(loss_components)

                if t < len(bptt_lists)-1:
                    node_feat = node_feat.detach()
                    e_t, v_t, i_t = e_t.detach(), v_t.detach(), i_t.detach()

            if self._parent.comet_logger is not None and \
                    batch_idx % self.config_t.get('train_log_every_n_steps', 1) == 0:
                logs = {}
                for k in loss_comps[0].keys():
                    logs[f'train/{k}'] = sum([l[k] for l in loss_comps]) / len(loss_comps)
                logs['train/loss_to_optimize_on'] = sum(loss_per_upd)/len(loss_per_upd)
                logs['grad_2.0_norm_total'] =  max(self._parent.norms2store) if self._parent.norms2store else 0.0

                # global_step is incremented by 2 in the lightning module, and it's already done (-2)
                self._parent.comet_logger.log_metrics(logs, step=self._parent.global_step//2 - 1)

                self._parent.norms2store = []

            self._report_timers(batch_idx)

        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):

        loss_to_optimize_on = 0

        node_feat = self._parent.net.node_prep_model(batch)
        preds_list, _ = self._parent.net.hg_model(
            node_feat, batch['node']['is_track'].squeeze(-1).bool())

        refiner_loss, loss_components, indices = self._parent.metrics.LAP_loss_single(
            preds_list[-1][-1], 
            (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
            node_is_track=batch['node']['is_track'])

        if 'hg_model' in self.config_t['train_components']:
            loss_to_optimize_on += refiner_loss.item()
            log_dict = {}
            for k, v in loss_components.items():
                log_dict[f'val/{k}'] = v

            inc_pred, ind_pred_logit, pred_is_charged = preds_list[-1][-1]

            self._parent.validation_step_perf.run_on_batch(
                inc_pred, ind_pred_logit, pred_is_charged, indices, batch)
                
        if 'kinematics' in self.config_t['train_components']:
            raise NotImplementedError
        
        log_dict['val_total_loss'] = loss_to_optimize_on
        self._parent.validation_step_perf.append_to_log_dicts(log_dict)

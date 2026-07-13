import comet_ml

# remove local paths, so that we don't use any
import sys, os
paths = sys.path
for p in paths:
    if '.local' in p:
            paths.remove(p)

import argparse
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_path_var', '-cv', type=str, required=False)
argparser.add_argument('--config_path_model_stage1', '-cms1', type=str, required=False)
argparser.add_argument('--config_path_model_stage2', '-cms2', type=str, required=False)
argparser.add_argument('--config_path_train', '-ct', type=str, required=True)
argparser.add_argument('--exp_key', '-ekey', type=str, required=False)
argparser.add_argument('--debug_mode', '-d', action='store_true')
argparser.add_argument('--precision', '-p', type=str, required=False, default='medium')
argparser.add_argument('--gpu', '-g', type=str, required=False, default='0')
argparser.add_argument('--resume_from_checkpoint', '-resume', type=str, required=False, default=None)

args = argparser.parse_args()
config_path_v = args.config_path_var
config_path_ms1 = args.config_path_model_stage1
config_path_ms2 = args.config_path_model_stage2
config_path_t = args.config_path_train
debug_mode = args.debug_mode
exp_key = args.exp_key
precision = args.precision

assert config_path_t is not None
stage1_condition = (config_path_v is not None) and (config_path_ms1 is not None) and (config_path_ms2 is None)
stage2_condition = (config_path_v is None) and (config_path_ms1 is None) and (config_path_ms2 is not None)
assert stage1_condition or stage2_condition, \
    "stage1 and stage2 are mutually exclusive\n" + \
    "stage1: need exactly --config_path_var (-cv), --config_path_model_stage1 (-cms1)\n" + \
    "stage2: need exactly --config_path_model_stage2 (-cms2)"
    

# need to set CUDA_VISIBLE_DEVICES before importing torch.
# Only pin it for single-GPU runs: under DDP each rank must see its own GPU
# (assignment is handled by Lightning / SLURM).
import yaml
with open(config_path_t, 'r') as fp:
    _cfg_t_peek = yaml.safe_load(fp)
if _cfg_t_peek.get('num_devices', 1) == 1 and _cfg_t_peek.get('num_nodes', 1) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.system('nvidia-smi')

import yaml, glob, random, string, shutil
import torch

from .lightnings.hgpf_lightning import HGPFLightning
from .utility.comet_helper import CometLoggerCustom
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment

# In interactive salloc sessions Lightning refuses to auto-detect SLURM (job name
# "interactive"/"bash"), so each srun task would become its own DDP launcher and
# spawn a private 4-process world. Force the SLURM env whenever srun gave us >1 task.
def _ddp_plugins(config_t):
    is_ddp = config_t.get('num_devices', 1) > 1 or config_t.get('num_nodes', 1) > 1
    if is_ddp and int(os.environ.get('SLURM_NTASKS', '1')) > 1:
        return [SLURMEnvironment(auto_requeue=False)]
    return None

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# The val plot-summary metrics are deliberately logged rank-locally (syncing them
# deadlocks: whether a rank logs them depends on which val batches it saw), so
# silence Lightning's blanket "use sync_dist=True" recommendation.
import warnings
warnings.filterwarnings('ignore', message='.*It is recommended to use.*sync_dist=True.*')

# set precision
torch.set_float32_matmul_precision(precision)

# configs
with open(config_path_t, 'r') as fp:
    config_t = yaml.safe_load(fp)

# CLI override lets chained jobs resume without editing the config each time
if args.resume_from_checkpoint is not None:
    config_t['resume_from_checkpoint'] = args.resume_from_checkpoint

if stage1_condition:
    print("\033[96mTraining stage 1\033[00m")
    with open(config_path_v, 'r') as fp:
        config_v = yaml.safe_load(fp)
    with open(config_path_ms1, 'r') as fp:
        config_ms1 = yaml.safe_load(fp)
    config_ms2 = None
else:
    print("\033[96mTraining stage 2 with frozen stage 1\033[00m")
    with open(config_path_ms2, 'r') as fp:
        config_ms2 = yaml.safe_load(fp)

    # get config_v and config_ms1 from frozen stage1
    config_path_v = config_t['config_path_v']
    with open(config_path_v, 'r') as fp:
        config_v = yaml.safe_load(fp)
    config_path_ms1 = config_t['config_path_ms1']
    with open(config_path_ms1, 'r') as fp:
        config_ms1 = yaml.safe_load(fp)

# model
lightning_model = HGPFLightning(config_v, config_ms1, config_ms2, config_t)

# DDP: derive the deterministic exp_key BEFORE the ModelCheckpoint below, so the
# checkpoint dirpath is set (all ranks agree on the key via the SLURM job/step ids)
if exp_key is None and not debug_mode and \
        (config_t.get('num_devices', 1) > 1 or config_t.get('num_nodes', 1) > 1):
    job_id = os.environ.get('SLURM_JOB_ID', '0')
    step_id = os.environ.get('SLURM_STEP_ID', '0')
    exp_key = f'{config_t["run_name"]}xjob{job_id}s{step_id}'
    exp_key = ''.join(c for c in exp_key.lower() if c.isalnum())
    exp_key = (exp_key + 'x' * 32)[:max(32, len(exp_key))]

# for saving checkpoints for best 3 models (according to val loss) and last epoch
checkpoint_callback = ModelCheckpoint(
    monitor='val_total_loss',
    mode='min',
    every_n_train_steps=0,
    every_n_epochs=1,
    train_time_interval=None,
    save_top_k=3,
    save_last= True,
    # deterministic dir (keyed by exp_key) so chained jobs can find/resume it;
    # None keeps the old logger-derived path when no exp_key is given.
    dirpath=(f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}/checkpoints'
             if exp_key is not None else None),
    filename='{epoch}-{val_total_loss:.4f}')


if debug_mode:
    trainer = Trainer(
        max_epochs = config_t['num_epochs'],
        accelerator = config_t['device'],
        devices = config_t['num_devices'],
        default_root_dir = config_t["base_root_dir"],
        callbacks = [checkpoint_callback],
        check_val_every_n_epoch = config_t['eval_every_n_epoch'],
        gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
        profiler = config_t.get('profiler', None),
        precision = config_t.get('trainer_precision', '32-true'),
        # find_unused_parameters: BPTT chunks detach node_feat, so node-prep params
        # produce no grad in later manual_backward calls (plus a couple of never-used norms)
        # stage 1 (hg_model) needs find_unused: BPTT chunks detach node_feat so some params
        # get no grad in later backwards. Stage 2 (hyperedge) uses every trainable param.
        strategy = (('ddp_find_unused_parameters_true' if 'hg_model' in config_t['train_components'] else 'ddp')
                    if (config_t.get('num_devices', 1) > 1 or config_t.get('num_nodes', 1) > 1) else 'auto'),
        num_nodes = config_t.get('num_nodes', 1),
        # stage 1 (hg_model): custom PflowSamplerMini shards batches by rank itself -> False.
        # stage 2 (hyperedge): plain DataLoader -> let Lightning inject DistributedSampler.
        use_distributed_sampler = ('hg_model' not in config_t['train_components']),
        plugins = _ddp_plugins(config_t),
    )

else:
    _is_ddp = config_t.get('num_devices', 1) > 1 or config_t.get('num_nodes', 1) > 1
    _rank0 = int(os.environ.get('SLURM_PROCID', os.environ.get('LOCAL_RANK', '0'))) == 0

    if exp_key is None:
        # (DDP keys are derived earlier, before the ModelCheckpoint; this is single-GPU)
        exp_key  = f'{config_t["run_name"]}xxx'
        exp_key += ''.join(random.choices(string.ascii_lowercase + string.digits, k=32-len(exp_key)))

    # archive the configs in the run dir (rank 0 only under DDP)
    if _rank0:
        dst = f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}'
        Path(dst).mkdir(parents=True, exist_ok=True)

        new_config_path_t = os.path.join(dst, 'config_t.yml')
        shutil.copyfile(config_path_t, new_config_path_t)

        new_config_path_v = os.path.join(dst, 'config_v.yml')
        shutil.copyfile(config_path_v, new_config_path_v)

        new_config_path_ms1 = os.path.join(dst, 'config_ms1.yml')
        shutil.copyfile(config_path_ms1, new_config_path_ms1)

        if config_path_ms2 is not None:
            new_config_path_ms2 = os.path.join(dst, 'config_ms2.yml')
            shutil.copyfile(config_path_ms2, new_config_path_ms2)

    comet_logger = CometLoggerCustom(
        api_key = os.environ["COMET_API_KEY"],
        project_name = config_t["project_name"], # hgpflow_v2
        workspace = os.environ["COMET_WORKSPACE"], # user_name
        experiment_name = config_t["run_name"],
        experiment_key_custom = exp_key
    )

    lightning_model.set_comet_logger(comet_logger)
    comet_logger.experiment.log_asset(config_path_v, file_name='config_v')
    comet_logger.experiment.log_asset(config_path_t, file_name='config_t')
    comet_logger.experiment.log_asset(config_path_ms1, file_name='config_ms1')
    if config_path_ms2 is not None:
        comet_logger.experiment.log_asset(config_path_ms2, file_name='config_ms2')

    comet_logger.experiment.log_parameter('ekey', exp_key)
    comet_logger.experiment.log_parameter(
        'experiment_path', f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}')

    # log files
    dirs2log = ['.', 'models', 'utility']
    for d in dirs2log:
        all_files = glob.glob(f'{d}/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath, file_name=f'{d}/{fpath}')

    trainer = Trainer(
        max_epochs = config_t['num_epochs'],
        accelerator = config_t['device'],
        devices = config_t['num_devices'],
        default_root_dir = config_t["base_root_dir"],
        callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=100)],
        check_val_every_n_epoch = config_t['eval_every_n_epoch'],
        log_every_n_steps = 1,
        logger = comet_logger,
        gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
        profiler = config_t.get('profiler', None),
        precision = config_t.get('trainer_precision', '32-true'),
        strategy = (('ddp_find_unused_parameters_true' if 'hg_model' in config_t['train_components'] else 'ddp')
                    if _is_ddp else 'auto'),  # see debug-branch comment
        num_nodes = config_t.get('num_nodes', 1),
        # stage 1 (hg_model): custom PflowSamplerMini shards batches by rank itself -> False.
        # stage 2 (hyperedge): plain DataLoader -> let Lightning inject DistributedSampler.
        use_distributed_sampler = ('hg_model' not in config_t['train_components']),
        plugins = _ddp_plugins(config_t),
    )

# run trainer
trainer.fit(lightning_model, ckpt_path=config_t['resume_from_checkpoint'])

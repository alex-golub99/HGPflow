import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger
from typing import TYPE_CHECKING, Any, Optional, Union
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
import os
if TYPE_CHECKING:
    from comet_ml import ExistingExperiment, Experiment, OfflineExperiment
import warnings


from typing import Optional, Any
from pytorch_lightning.loggers import CometLogger
from comet_ml import Experiment, ExistingExperiment, OfflineExperiment

class CometLoggerCustom(CometLogger):
    """
    Custom CometLogger that supports manually setting experiment keys,
    works with latest comet_ml and Lightning.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        experiment_key_custom: Optional[str] = None,  # custom addition
        online: bool = True,  # use online=False for offline
        offline_dir: Optional[str] = None,  # directory for offline logging
        **kwargs: Any,
    ):
        # normalize legacy kwarg spellings from call sites (project_name/experiment_name)
        # so they don't collide with comet_ml's deprecation handling
        project = project or kwargs.pop('project_name', None)
        name = name or kwargs.pop('experiment_name', None)
        super().__init__(
            api_key=api_key,
            project=project,
            name=name,
            online=online,
            **kwargs
        )
        self._offline_dir = offline_dir
        self._experiment_key_custom = experiment_key_custom
        self._exp_name_custom = name  # PL may not store name as an attribute

    @property
    @rank_zero_experiment
    def experiment(self):
        """
        Returns the underlying Comet Experiment object.
        Creates it if it doesn't exist yet.
        Guarded by rank_zero_experiment so only rank 0 logs under DDP.
        """
        if self._experiment is not None and getattr(self._experiment, "alive", True):
            return self._experiment

        # Decide which type of experiment to create
        from comet_ml import Experiment, ExistingExperiment, OfflineExperiment

        # attribute names differ across pytorch_lightning versions (api_key vs _api_key,
        # _project vs _project_name); unknown init kwargs also end up in self._kwargs,
        # so resolve everything explicitly and do NOT splat _kwargs into Experiment().
        _kwargs = dict(getattr(self, '_kwargs', {}) or {})
        _api_key = getattr(self, 'api_key', None) or getattr(self, '_api_key', None) \
            or os.environ.get('COMET_API_KEY')
        _project = getattr(self, '_project', None) or getattr(self, '_project_name', None) \
            or _kwargs.pop('project_name', None)
        _workspace = getattr(self, '_workspace', None) or _kwargs.pop('workspace', None)
        _exp_name = getattr(self, '_exp_name_custom', None) \
            or _kwargs.pop('experiment_name', None) or getattr(self, '_name', None)

        if self._online:
            exp_kwargs = dict(api_key=_api_key, project_name=_project, workspace=_workspace)
            if self._experiment_key_custom is not None:
                exp_kwargs['experiment_key'] = self._experiment_key_custom
            self._experiment = Experiment(**exp_kwargs)
        else:
            self._experiment = OfflineExperiment(
                offline_directory=self._offline_dir,
                project_name=_project,
                workspace=_workspace,
            )

        # Set experiment name if provided
        if _exp_name:
            self._experiment.set_name(_exp_name)

        # Log that Lightning created this experiment
        self._experiment.log_other("Created from", "pytorch-lightning")
        return self._experiment


def save_plot(fig, name, comet_logger=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    if comet_logger is not None:
        fig.canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(int(h), int(w), 4)[:, :, :3]
        comet_logger.experiment.log_image(
            image_data=image,
            name=name,
            overwrite=False, 
            image_format="png",
        )
    else:
        fig.savefig(f'plot_dump/{name}.png', bbox_inches='tight')
    plt.close(fig)


def log_parameters(comet_logger, config, prefix=''):
    for k, v in config.items():
        if isinstance(v, dict):
            log_parameters(comet_logger, v, prefix=f'{prefix}{k}.')
        else:
            comet_logger.experiment.log_parameter(f'{prefix}{k}', v)
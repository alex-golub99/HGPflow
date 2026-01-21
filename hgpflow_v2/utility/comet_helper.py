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
        super().__init__(
            api_key=api_key,
            project=project,
            name=name,
            online=online,
            **kwargs
        )
        self._offline_dir = offline_dir
        self._experiment_key_custom = experiment_key_custom

    @property
    def experiment(self):
        """
        Returns the underlying Comet Experiment object.
        Creates it if it doesn't exist yet.
        """
        if self._experiment is not None and getattr(self._experiment, "alive", True):
            return self._experiment

        # Decide which type of experiment to create
        from comet_ml import Experiment, ExistingExperiment, OfflineExperiment

        if self.online:
            if self._experiment_key_custom is not None:
                # Create experiment with custom key
                self._experiment = Experiment(
                    api_key=self.api_key,
                    project_name=self._project,
                    experiment_key=self._experiment_key_custom,
                    **self._kwargs
                )
            else:
                self._experiment = Experiment(
                    api_key=self.api_key,
                    project_name=self._project,
                    **self._kwargs
                )
        else:
            self._experiment = OfflineExperiment(
                offline_directory=self._offline_dir,
                project_name=self._project,
                **self._kwargs
            )

        # Set experiment name if provided
        if getattr(self, "_name", None):
            self._experiment.set_name(self._name)

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
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)
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
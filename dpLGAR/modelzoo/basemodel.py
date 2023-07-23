from typing import Dict, Optional, Callable

from omegaconf import DictConfig
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: DictConfig):
        super(BaseModel, self).__init__()
        self.cfg = cfg

        self.alpha = torch.tensor(0.0)
        self.n = torch.tensor(0.0)
        self.ksat = torch.tensor(0.0)

        self._set_parameters()


    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model output and potentially any intermediate states and activations as a dictionary.
        """
        raise NotImplementedError

    def _set_parameters(self):
        """Sets the module parameters"""
        raise NotImplementedError
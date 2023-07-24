import logging
import warnings

from omegaconf import DictConfig
import torch.nn as nn

from dpLGAR.modelzoo.base_lgar import BaseLGAR

log = logging.getLogger(__name__)


def get_model(cfg: DictConfig) -> nn.Module:
    """Get model object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """

    if cfg.modelzoo.model.lower() == "base":
        model = BaseLGAR(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.modelzoo.model} not implemented or not linked in `get_model()`")

    return model

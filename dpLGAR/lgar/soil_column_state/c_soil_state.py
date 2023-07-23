import logging

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor

from dpLGAR.lgar.soil_column_state.base_state import BaseState

log = logging.getLogger(__name__)


class CSoilState(BaseState):
    def __init__(self, cfg: DictConfig, ponded_depth_max) -> None:
        super(CSoilState).__init__(cfg=cfg)
        s
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

log = logging.getLogger("models.physics.layers.Layer")


class Layer:
    def __init__(self, cfg, global_params, c, alpha, n, ksat, is_top=False):
        """
        A layer of soil. Each soil layer can have many wetting fronts and several properties
        :param cfg:
        :param global_params:
        :param c:
        :param alpha:
        :param n:
        :param ksat:
        :param is_top:
        """
        super().__init__()
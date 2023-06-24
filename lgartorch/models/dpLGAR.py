from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.data.utils import read_test_params

log = logging.getLogger("models.dpLGAR")


class dpLGAR(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(dpLGAR, self).__init__()

        # Setting parameters
        alpha_, n_, ksat_ = read_test_params(cfg)
        self.alpha = nn.ParameterList([])
        self.n = nn.ParameterList([])
        self.ksat = nn.ParameterList([])
        for i in range(alpha_.shape[0]):
            self.alpha.append(nn.Parameter(alpha_[i]))
            self.n.append(nn.Parameter(n_[i]))
            self.ksat.append(nn.Parameter(ksat_[i]))

        # Creating initial soil Layers

    def forward(self, x, c) -> Tensor:
        """
        The forward function to model Precip/PET through LGAR functions
        :param x: Precip and PET forcings
        :param c: Soil Attributes
        :return:
        """
        pass

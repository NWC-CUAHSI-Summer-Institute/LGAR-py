"""A class to hold the Multilayer Perceptron Model to estimate river parameters"""
import logging
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Sigmoid, Linear

from utils.transform import to_physical

log = logging.getLogger("graphs.MLP")


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        The Multilayer Perceptron Model (MLP) which learns values
        of n and q_spatial from downstream discharge

        args:
        - cfg: The DictConfig object that houses global variables
        """
        super().__init__()
        self.cfg = cfg
        input_size = self.cfg.models.mlp.input_size
        hidden_size = self.cfg.models.mlp.hidden_size
        output_size = self.cfg.models.mlp.output_size
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, hidden_size)
        self.lin3 = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self.alpha_range = [self.cfg.models.range.lb[0], self.cfg.models.range.ub[0]]
        self.n_range = [self.cfg.models.range.lb[1], self.cfg.models.range.ub[1]]
        self.ksat_range = [self.cfg.models.range.lb[2], self.cfg.models.range.ub[2]]
        self.ponded_max_range = [self.cfg.models.range.lb[3], self.cfg.models.range.ub[3]]
        self.theta_e_range = [self.cfg.models.range.lb[4], self.cfg.models.range.ub[4]]
        self.theta_r_range = [self.cfg.models.range.lb[5], self.cfg.models.range.ub[5]]

    def forward(self, c: Tensor):
        """
        The forward run function for the MLP

        arguments:
        - c: Normalized attributes inputs going into the MLP Layers

        returns:
        - n: (nn.Parameter()) Manning's roughness estimates
        - q_spatial: (nn.Parameter())
        """
        l1 = self.lin1(c)
        l2 = self.lin2(l1)
        l3 = self.lin3(l2)
        out = self.sigmoid(l3)
        c_transpose = out.transpose(0, 1)
        alpha = to_physical(c_transpose[0], self.alpha_range)
        n = to_physical(c_transpose[1], self.n_range)
        ksat = to_physical(c_transpose[2], self.ksat_range)
        ponded_max_depth = to_physical(c_transpose[3], self.ponded_max_range)
        theta_e = to_physical(c_transpose[4], self.theta_e_range)
        theta_r = to_physical(c_transpose[5], self.theta_r_range)
        return alpha, n, ksat, ponded_max_depth, theta_e, theta_r
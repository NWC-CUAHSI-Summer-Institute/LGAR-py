"""A class to hold the Multilayer Perceptron Model to estimate river parameters"""
import logging
from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn
from torch.nn import Sigmoid, Linear

from dpLGAR.training.utils import to_physical

log = logging.getLogger("graphs.MLP")


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig, input_data) -> None:
        """
        The Multilayer Perceptron Model (MLP) which learns values
        of n and q_spatial from downstream discharge

        args:
        - cfg: The DictConfig object that houses global variables
        """
        super().__init__()
        self.cfg = cfg
        batch_size, num_features, feature_dim = input_data.size()
        input_size = batch_size * feature_dim
        hidden_size = self.cfg.models.mlp.hidden_size
        output_size = self.cfg.models.mlp.output_size
        self.layers = nn.Sequential(
            Linear(input_size, hidden_size),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, output_size),
            Sigmoid(),
        )
        self.ponded_depth_lin = nn.Linear(6, 1)
        self.alpha_range = [self.cfg.models.range.lb[0], self.cfg.models.range.ub[0]]
        self.n_range = [self.cfg.models.range.lb[1], self.cfg.models.range.ub[1]]
        self.ksat_range = [self.cfg.models.range.lb[2], self.cfg.models.range.ub[2]]
        self.theta_e_range = [self.cfg.models.range.lb[3], self.cfg.models.range.ub[3]]
        self.theta_r_range = [self.cfg.models.range.lb[4], self.cfg.models.range.ub[4]]
        self.ponded_max_range = [self.cfg.models.range.lb[5], self.cfg.models.range.ub[5]]


    def forward(self, x: Tensor):
        """
        The forward run function for the MLP

        arguments:
        - x: Normalized attributes inputs going into the MLP Layers

        returns:
        - n: (nn.Parameter()) Manning's roughness estimates
        - q_spatial: (nn.Parameter())
        """
        batch_size, num_features, feature_dim = x.size()
        x = x.permute(1, 0, 2).reshape(num_features, -1)
        out = self.layers(x)
        ponded_depth_out = self.ponded_depth_lin(out[5]).squeeze()
        x_transpose = out.transpose(0, 1)
        alpha = to_physical(x_transpose[0], self.alpha_range)
        n = to_physical(x_transpose[1], self.n_range)
        ksat = to_physical(x_transpose[2], self.ksat_range)
        theta_e = to_physical(x_transpose[3], self.theta_e_range)
        theta_r = to_physical(x_transpose[4], self.theta_r_range)
        ponded_max_depth = to_physical(ponded_depth_out, self.ponded_max_range)
        return alpha, n, ksat, theta_e, theta_r, ponded_max_depth
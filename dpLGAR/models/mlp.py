from functools import partial
import logging

from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import torch.nn as nn

from dpLGAR.models import Initialization

log = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(MLP, self).__init__()
        self.cfg = cfg
        input_size = len(self.cfg.data.varC) * 1000  # Num_samples
        hidden_size = self.cfg.models.hidden_size * 1000  # Num samples
        output_size = 5 * len(self.cfg.data.layer_thickness) + 1
        self.Initialization = Initialization(self.cfg)
        self.m1 = nn.Flatten(start_dim=0, end_dim=-1)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def initialize_weights(self):
        func = self.Initialization.get()
        init_func = partial(self._initialize_weights, func=func)
        self.apply(init_func)

    def _denormalize(self, idx: int, param: torch.Tensor) -> torch.Tensor:
        value_range = self.cfg.models.transformations[idx]
        output = (param * (value_range[1] - value_range[0])) + value_range[0]
        return output

    @staticmethod
    def _initialize_weights(m, func):
        if isinstance(m, nn.Linear):
            func(m.weight)

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        log.info("Running MLP forward")
        _x = self.m1(inputs)
        x = self.layers(_x)
        alpha = self._denormalize(0, x[0:3])
        n = self._denormalize(1, x[3:6])
        ksat = self._denormalize(2, x[6:9])
        theta_e = self._denormalize(3, x[9:12])
        theta_r = self._denormalize(4, x[12:15])
        ponded_depth_max = self._denormalize(3, x[-1])
        return alpha, n, ksat, theta_e, theta_r, ponded_depth_max

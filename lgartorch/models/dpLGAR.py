from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
import torch.nn as nn

log = logging.getLogger("models.dpLGAR")


class dpLGAR(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(dpLGAR, self).__init__()

        # Setting parameters
        alpha_, n_, ksat_ = self.read_test_params(cfg)
        self.alpha = nn.ParameterList([])
        self.n = nn.ParameterList([])
        self.ksat = nn.ParameterList([])
        for i in range(alpha_.shape[0]):
            self.alpha.append(nn.Parameter(alpha_[i]))
            self.n.append(nn.Parameter(n_[i]))
            self.ksat.append(nn.Parameter(ksat_[i]))


    def forward(self, x, c):
        """
        The forward function to model Precip/PET through LGAR functions
        :param x: Precip and PET forcings
        :param c: Soil Attributes
        :return:
        """
        pass

    def read_test_params(self, cfg):
        alpha_test_params = torch.tensor(
            [
                0.0100,
                0.0200,
                0.0100,
                0.0300,
                0.0400,
                0.0300,
                0.0200,
                0.0300,
                0.0100,
                0.0200,
                0.0100,
                0.0100,
                0.0031,
                0.0083,
                0.0037,
                0.0096,
                0.0053,
                0.0045,
            ],
            device=cfg.device,
        )
        n_test_params = torch.tensor(
            [
                1.2500,
                1.4200,
                1.4700,
                1.7500,
                3.1800,
                1.2100,
                1.3300,
                1.4500,
                1.6800,
                1.3200,
                1.5200,
                1.6600,
                1.6858,
                1.2990,
                1.6151,
                1.3579,
                1.5276,
                1.4585,
            ],
            device=cfg.device,
        )

        k_sat_test_params = torch.tensor(
            [
                6.1200e-01,
                3.3480e-01,
                5.0400e-01,
                4.3200e00,
                2.6640e01,
                4.6800e-01,
                5.4000e-01,
                1.5840e00,
                1.8360e00,
                4.3200e-01,
                4.6800e-01,
                7.5600e-01,
                4.5000e-01,
                7.0000e-02,
                4.5000e-01,
                7.0000e-02,
                2.0000e-02,
                2.0000e-01,
            ],
            device=cfg.device,
        )

        return alpha_test_params, n_test_params, k_sat_test_params

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
                0.01,
                0.02,
                0.01,
                0.03,
                0.04,
                0.03,
                0.02,
                0.03,
                0.01,
                0.02,
                0.01,
                0.01,
                0.0031297,
                0.0083272,
                0.0037454,
                0.009567,
                0.005288,
                0.004467,
            ],
            device=cfg.device,
        )
        n_test_params = torch.tensor(
            [
                1.25,
                1.42,
                1.47,
                1.75,
                3.18,
                1.21,
                1.33,
                1.45,
                1.68,
                1.32,
                1.52,
                1.66,
                1.6858,
                1.299,
                1.6151,
                1.3579,
                1.5276,
                1.4585,
            ],
            device=cfg.device,
        )

        k_sat_test_params = torch.tensor(
            [
                0.612,
                0.3348,
                0.504,
                4.32,
                26.64,
                0.468,
                0.54,
                1.584,
                1.836,
                0.432,
                0.468,
                0.756,
                0.45,
                0.07,
                0.45,
                0.07,
                0.02,
                0.2,
            ],
            device=cfg.device,
        )

        return alpha_test_params, n_test_params, k_sat_test_params

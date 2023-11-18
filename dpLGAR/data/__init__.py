from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class Scaler(ABC):
    """
    Abstract base class for handling normalization.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.basin_area = None
        self.mean_prep = None
        self.stat_dict = {}

    def initialize(self, data):
        self.basin_area = data.areas.reshape(-1, 1)
        self.mean_prep = data.p_mean.reshape(-1, 1)
        self.read()

    def read(self):
        stat_dict_path = Path(self.cfg.scaler.stat_dict)
        with stat_dict_path.open("r") as fp:
            self.stat_dict = json.load(fp)

    def write(self):
        stat_dict_path = Path(self.cfg.scaler.stat_dict)
        with stat_dict_path.open("w") as fp:
            json.dump(self.stat_dict, fp, indent=4)

    def calStatgamma(self, x):
        """
        for daily streamflow and precipitation
        :param x: inputs
        :return:
        """
        a = x.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        b = np.log10(
            np.sqrt(b) + 0.1
        )  # do some tranformation to change gamma characteristics
        p10 = np.percentile(b, 10).astype(float)
        p90 = np.percentile(b, 90).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        if std < 0.001:
            std = 1
        return [p10, p90, mean, std]

    def calStatbasinnorm(self, x):
        """
        for daily streamflow normalized by basin area and precipitation
        :param x:
        :return:
        """
        temparea = np.tile(self.basin_area, (1, x.shape[1]))
        tempprep = np.tile(self.mean_prep, (1, x.shape[1]))
        flowua = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10**6)) * (tempprep * 10 ** (-3))
        )  # unit (m^3/day)/(m^3/day)
        a = flowua.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        b = np.log10(
            np.sqrt(b) + 0.1
        )  # do some tranformation to change gamma characteristics plus 0.1 for 0 values
        p10 = np.percentile(b, 10).astype(float)
        p90 = np.percentile(b, 90).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        if std < 0.001:
            std = 1
        return [p10, p90, mean, std]

    def calStat(self, x):
        """

        :param x: inputs
        :return:
        """
        a = x.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        p10 = np.percentile(b, 10).astype(float)
        p90 = np.percentile(b, 90).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        if std < 0.001:
            std = 1
        return [p10, p90, mean, std]

    def transNorm(self, x, varLst, toNorm):
        if type(varLst) is str:
            varLst = [varLst]
        out = np.zeros(x.shape)
        for k in range(len(varLst)):
            var = varLst[k]
            stat = self.stat_dict[var]
            if toNorm is True:
                if len(x.shape) == 3:
                    if var == "prcp" or var == "usgsFlow":
                        x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
                elif len(x.shape) == 2:
                    if var == "prcp" or var == "usgsFlow":
                        x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
            else:
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                    if var == "prcp" or var == "usgsFlow":
                        temptrans = np.power(10, out[:, :, k]) - 0.1
                        temptrans[temptrans < 0] = 0  # set negative as zero
                        out[:, :, k] = (temptrans) ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[3] + stat[2]
                    if var == "prcp" or var == "usgsFlow":
                        temptrans = np.power(10, out[:, k]) - 0.1
                        temptrans[temptrans < 0] = 0
                        out[:, k] = (temptrans) ** 2
        return out

    def basinNorm(self, x, toNorm):
        nd = len(x.shape)
        if nd == 3 and x.shape[2] == 1:
            x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
        temparea = np.tile(self.basin_area, (1, x.shape[1]))
        tempprep = np.tile(self.mean_prep, (1, x.shape[1]))
        if toNorm is True:
            flow = (x * 0.0283168 * 3600 * 24) / (
                    (temparea * (10 ** 6)) * (tempprep * 10 ** (-3))
            )  # (m^3/day)/(m^3/day)
        else:

            flow = (
                    x
                    * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))
                    / (0.0283168 * 3600 * 24)
            )
        if nd == 3:
            flow = np.expand_dims(flow, axis=2)
        return flow

    def transNormbyDic(self, x, varLst, toNorm):
        if type(varLst) is str:
            varLst = [varLst]
        out = np.zeros(x.shape)
        for k in range(len(varLst)):
            var = varLst[k]
            stat = self.stat_dict[var]
            if toNorm is True:
                if len(x.shape) == 3:
                    if var in [
                        "prcp",
                        "usgsFlow",
                        "Precip",
                        "runoff",
                        "Runoff",
                        "Runofferror",
                    ]:
                        temp = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                        out[:, :, k] = (temp - stat[2]) / stat[3]
                    else:
                        out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
                elif len(x.shape) == 2:
                    if var in [
                        "prcp",
                        "usgsFlow",
                        "Precip",
                        "runoff",
                        "Runoff",
                        "Runofferror",
                    ]:
                        temp = np.log10(np.sqrt(x[:, k]) + 0.1)
                        out[:, k] = (temp - stat[2]) / stat[3]
                    else:
                        out[:, k] = (x[:, k] - stat[2]) / stat[3]
            else:
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                    if var in [
                        "prcp",
                        "usgsFlow",
                        "Precip",
                        "runoff",
                        "Runoff",
                        "Runofferror",
                    ]:
                        temptrans = np.power(10, out[:, :, k]) - 0.1
                        temptrans[temptrans < 0] = 0  # set negative as zero
                        out[:, :, k] = (temptrans) ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[3] + stat[2]
                    if var in [
                        "prcp",
                        "usgsFlow",
                        "Precip",
                        "runoff",
                        "Runoff",
                        "Runofferror",
                    ]:
                        temptrans = np.power(10, out[:, k]) - 0.1
                        temptrans[temptrans < 0] = 0
                        out[:, k] = (temptrans) ** 2
        return out

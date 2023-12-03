import hydra
import logging
from omegaconf import DictConfig
import random
import sys
import time

sys.path.insert(0, '/Users/taddbindas/projects/soils_work/lgar-py/dpLGAR/plugins/')

import numpy as np
import torch

from dpLGAR.agents.MLP_LGAR import Agent
# from dpLGAR.plugins import HybridConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    _set_seed(cfg)
    start = time.perf_counter()
    # hybrid_cfg = HybridConfig(cfg)
    agent = Agent(cfg)
    agent.run()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


def _set_seed(cfg):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(1)
    random.seed(0)


if __name__ == "__main__":
    main()

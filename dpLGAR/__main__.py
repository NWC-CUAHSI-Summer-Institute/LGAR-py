import hydra
import logging
from omegaconf import DictConfig
import sys
import time

sys.path.insert(0, '/Users/taddbindas/projects/soils_work/lgar-py/dpLGAR/plugins/')

from dpLGAR.agents.MLP_LGAR import Agent
# from dpLGAR.plugins import HybridConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    # hybrid_cfg = HybridConfig(cfg)
    agent = Agent(cfg)
    agent.run()
    agent.finalize()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()

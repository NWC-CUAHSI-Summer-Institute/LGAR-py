import hydra
import logging
from omegaconf import DictConfig
import time

from dpLGAR.agents.DifferentiableLGAR import DifferentiableLGAR
from dpLGAR.plugins import neural_hydrology_config_adapter

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    hybrid_cfg = neural_hydrology_config_adapter(cfg)
    agent = DifferentiableLGAR(hybrid_cfg)
    agent.run()
    agent.finalize()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()

import hydra
import logging
from omegaconf import DictConfig
import os
import time

from dpLGAR.agents.DataParallelLGAR import DataParallelLGAR
from dpLGAR.agents.SingleBasinRun import LGARAgent
from dpLGAR.agents.SyntheticAgent import SyntheticAgent


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    try:
        # Running the code in Parallel
        # Pulling the LOCAL_RANK that is generated by torchrun
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        agent = DataParallelLGAR(cfg)
    except KeyError:
        # There is no Data Parallel in use
        cfg.local_rank = 0
        cfg.nproc = 1
        agent = LGARAgent(cfg)
        # agent = SyntheticAgent(cfg)
    agent.run()
    agent.finalize()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()

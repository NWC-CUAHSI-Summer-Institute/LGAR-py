import logging
import time

import hydra
from omegaconf import DictConfig

from dpLGAR.training.basetrainer import BaseTrainer


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    if cfg.mode == "train":
        start_train(cfg)
    end = time.perf_counter()
    log.info(f"Run took : {(end - start):.6f} seconds")


def start_train(cfg: DictConfig) -> None:
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

if __name__ == "__main__":
    main()

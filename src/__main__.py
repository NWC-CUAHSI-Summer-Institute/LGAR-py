import hydra
import logging
from omegaconf import DictConfig

from LGARBmi import LGARBmi
from tests import sanity_checks

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if sanity_checks.soil_types(cfg):
        raise sanity_checks.DataError

    lgar_bmi = LGARBmi()
    lgar_bmi.initialize(str(cfg))
    # TODO SET VARS
    lgar_bmi.update_until(lgar_bmi.nsteps)
    lgar_bmi.finalize()



if __name__ == "__main__":
    main()
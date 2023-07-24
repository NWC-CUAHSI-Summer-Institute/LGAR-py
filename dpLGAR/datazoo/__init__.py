import logging

from omegaconf import DictConfig

from dpLGAR.datazoo.basedataset import BaseDataset
from dpLGAR.datazoo.phillipsburg import Phillipsburg

log = logging.getLogger(__name__)


def get_dataset(cfg: DictConfig,
                is_train: bool,
                period: str,
                basin: str = None) -> BaseDataset:
    """Get data set instance, depending on the run configuration.

    Currently implemented datasets are 'caravan', 'camels_aus', 'camels_br', 'camels_cl', 'camels_gb', 'camels_us', and
    'hourly_camels_us', as well as the 'generic' dataset class that can be used for any kind of dataset as long as it is
    in the correct format.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) is(are) read from the appropriate
        basin file, corresponding to the `period`.
    Returns
    -------
    BaseDataset
        A new data set instance, depending on the run configuration.

    Raises
    ------
    NotImplementedError
        If no data set class is implemented for the 'dataset' argument in the config.
    """
    if cfg.datazoo.dataset.lower() == "phillipsburg":
        Dataset = Phillipsburg
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.datazoo.dataset}")

    ds = Dataset(cfg=cfg,
                 is_train=is_train,
                 period=period,
                 basin=basin)
    return ds
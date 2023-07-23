"""
The Base Agent class, where all other training inherit from, that contains definitions for all the necessary functions
"""
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from dpLGAR.datazoo import get_dataset
from dpLGAR.datazoo import BaseDataset
from dpLGAR.training import get_optimizer, get_loss_obj


class BaseTrainer:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None

        self.basins = cfg.basin_id
        self.current_epoch = self._get_start_epoch_number()

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self, pred, obs):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the flat_files loader
        :return:
        """
        raise NotImplementedError

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, is_train=True, period="train", basin=self.basins)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        return DataLoader(ds,
                          batch_size=self.cfg.batch_size,
                          shuffle=False,)

    def _get_start_epoch_number(self):
        # TODO support loading epochs
        return 0

    def _set_device(self):
        self.device = torch.device("cpu")

    def _set_random_seeds(self):
        torch.manual_seed(0)

    def initialize_training(self):
        """Initialize the training class.

        This method will load the model, initialize loss, optimizer, dataset and dataloader,
        tensorboard logging, and Tester class.
        If called in a ``continue_training`` context, this model will also restore the model and optimizer state.
        """
        # Initialize dataset before the model is loaded.
        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)




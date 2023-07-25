"""
The Base Agent class, where all other training inherit from, that contains definitions for all the necessary functions
"""
import logging
import sys

import pandas as pd
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpLGAR.datazoo import get_dataset
from dpLGAR.datazoo import BaseDataset
from dpLGAR.modelzoo import get_model
from dpLGAR.training import get_optimizer, get_loss_obj


log = logging.getLogger(__name__)


class BaseTrainer:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.loader = None
        self.validator = None

        self.basins = cfg.basin_id
        self._epoch = self._get_start_epoch_number()

        self.device = torch.device(cfg.device)

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, is_train=True, period="train", basin=self.basins)

    def _get_model(self, c: pd.DataFrame) -> torch.nn.Module:
        return get_model(cfg=self.cfg, c=c)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self):
        return get_loss_obj(cfg=self.cfg)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        return DataLoader(ds,
                          batch_size=self.cfg.datazoo.batch_size,
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
        self.model = self._get_model(ds.attributes).to(self.device)
        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

    def train_and_validate(self):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs.
        """
        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            if epoch in self.cfg.learning_rate.keys():
                log.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)

    def _train_epoch(self, epoch: int):
        self.model.train()

        # process bar handle
        pbar = tqdm(self.loader, file=sys.stdout)
        pbar.set_description(f'# Epoch {epoch}')

        for data in pbar:
            predictions = self.model(data)
        self.model.global_mb.print()
        loss = self.loss_obj(predictions, data)

        # delete old gradients
        self.optimizer.zero_grad()

        # get gradients
        loss.backward()

        # update weights
        self.optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

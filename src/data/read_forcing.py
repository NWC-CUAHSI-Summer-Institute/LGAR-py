"""A file to store the function where we read the input data"""
import logging

import numpy as np
import pandas as pd
from pathlib import Path
import torch

from src.tests.sanity_checks import DataError

log = logging.getLogger("data.read_forcing")


def read_forcing_data(file_path: str) -> (np.ndarray, torch.Tensor, torch.Tensor):
    """
    a function to read the forcing input dataset
    :param file_path: the file we want to read
    :return:
    - time
    - precipitation
    - PET
    """
    forcing_file_path = Path(file_path)

    # Check if forcing file exists
    if not forcing_file_path.is_file():
        log.error(f"File {forcing_file_path} doesn't exist")
        raise DataError
    df = pd.read_csv(file_path)

    # Convert pandas dataframe to PyTorch tensors
    time = df["Time"].values
    precip = torch.tensor(df["P(mm/h)"].values, dtype=torch.float32)
    pet = torch.tensor(df["PET(mm/h)"].values, dtype=torch.float32)

    return time, precip, pet

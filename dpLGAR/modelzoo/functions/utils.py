import logging
from sklearn import preprocessing
import torch

log = logging.getLogger("modelzoo.functions.utils")

def to_physical(x, parameter_space):
    """
    The reverse scaling function to find the physical param from the scaled param (range [0,1))
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """
    x_ = x * (parameter_space[1] - parameter_space[0])
    output = x_ + parameter_space[0]
    return output


def from_physical(x, param):
    """
    The scaling function to convert a physical param to a value within [0,1)
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """
    raise NotImplementedError


def min_max_normalization(x):
    """
    A min/max Scaler for each feature to be fed into the MLP
    :param x:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_trans = x.transpose(1, 0)
    x_tensor = torch.zeros(x_trans.shape)
    for i in range(0, x_trans.shape[0]):
        x_tensor[i, :] = torch.tensor(
            min_max_scaler.fit_transform(x_trans[i, :].reshape(-1, 1)).transpose(1, 0)
        )
    """Transposing to do correct normalization"""
    return torch.transpose(x_tensor, 1, 0)

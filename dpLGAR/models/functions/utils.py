import logging
from sklearn import preprocessing
import torch

log = logging.getLogger("models.functions.utils")

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

def _setup_normalization():
    # default center and scale values are feature mean and std
    self.scaler["xarray_feature_scale"] = xr.std(skipna=True)
    self.scaler["xarray_feature_center"] = xr.mean(skipna=True)

    # check for feature-wise custom normalization
    for feature, feature_specs in self.cfg.custom_normalization.items():
        for key, val in feature_specs.items():
            # check for custom treatment of the feature center
            if key == "centering":
                if (val is None) or (val.lower() == "none"):
                    self.scaler["xarray_feature_center"][feature] = np.float32(0.0)
                elif val.lower() == "median":
                    self.scaler["xarray_feature_center"][feature] = xr[feature].median(skipna=True)
                elif val.lower() == "min":
                    self.scaler["xarray_feature_center"][feature] = xr[feature].min(skipna=True)
                elif val.lower() == "mean":
                    # Do nothing, since this is the default
                    pass
                else:
                    raise ValueError(f"Unknown centering method {val}")

            # check for custom treatment of the feature scale
            elif key == "scaling":
                if (val is None) or (val.lower() == "none"):
                    self.scaler["xarray_feature_scale"][feature] = np.float32(1.0)
                elif val == "minmax":
                    self.scaler["xarray_feature_scale"][feature] = xr[feature].max(skipna=True) - \
                                                                   xr[feature].min(skipna=True)
                elif val == "std":
                    # Do nothing, since this is the default
                    pass
                else:
                    raise ValueError(f"Unknown scaling method {val}")
            else:
                # raise ValueError to point to the correct argument names
                raise ValueError("Unknown dict key. Use 'centering' and/or 'scaling' for each feature.")

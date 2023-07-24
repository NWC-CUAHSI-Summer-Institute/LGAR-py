import torch


def read_test_params() -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Sets the values of alpha, n, and ksat to the values defined nby the LGAR-C soils profiles
    """
    alpha_test_params = torch.tensor(
        [
            0.01,
            0.02,
            0.01,
            0.03,
            0.04,
            0.03,
            0.02,
            0.03,
            0.01,
            0.02,
            0.01,
            0.01,
            0.0031297,
            0.0083272,
            0.0037454,
            0.009567,
            0.005288,
            0.004467,
        ],
    )
    n_test_params = torch.tensor(
        [
            1.25,
            1.42,
            1.47,
            1.75,
            3.18,
            1.21,
            1.33,
            1.45,
            1.68,
            1.32,
            1.52,
            1.6599999999,
            1.6858,
            1.299,
            1.6151,
            1.3579,
            1.5276,
            1.4585,
        ],
    )

    k_sat_test_params = torch.tensor(
        [
            0.612,
            0.3348,
            0.504,
            4.32,
            26.64,
            0.468,
            0.54,
            1.584,
            1.836,
            0.432,
            0.468,
            0.756,
            0.45,
            0.07,
            0.45,
            0.07,
            0.02,
            0.2,
        ],
    )

    return alpha_test_params, n_test_params, k_sat_test_params
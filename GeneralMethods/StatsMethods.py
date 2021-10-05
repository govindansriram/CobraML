import numpy as np
import torch

"""
This method follow the formula of multivariate normal distribution which can be easily googled.
"""


def multivariate_normal_distribution(feature: torch.Tensor,
                                     mu: torch.Tensor,
                                     sigma: torch.Tensor) -> torch.tensor:
    feature = torch.unsqueeze(feature, dim=0).T

    val = 1 / (((2 * np.pi) ** (feature.size()[1] / 2)) * (torch.det(sigma) ** (1 / 2)))

    val_2 = torch.mm(((-1 / 2) * (feature - mu).T), torch.inverse(sigma))  # regular inverse

    val_2 = torch.exp(torch.mm(val_2, (feature - mu)))

    return val * val_2

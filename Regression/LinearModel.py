import torch
from torch import nn
from GeneralMethods.ThetaMethods import add_extra_input


class LinearRegression(nn.Module):

    def __init__(self,
                 theta_params: int,
                 device="cpu",
                 randomize=False):
        super(LinearRegression, self).__init__()
        self.__device = device
        self.__parameters = torch.rand((1, theta_params + 1),
                                       dtype=torch.float64,
                                       requires_grad=True,
                                       device=device) if randomize else torch.zeros((1, theta_params + 1),
                                                                                    dtype=torch.float64,
                                                                                    requires_grad=True,
                                                                                    device=device)

    def get_parameters(self):
        return self.__parameters

    def forward(self,
                x_input: torch.FloatTensor) -> torch.Tensor:
        return torch.matmul(add_extra_input(x_input, self.__device),
                            self.__parameters.T)

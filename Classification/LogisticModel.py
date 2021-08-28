from torch import nn
from GeneralMethods.ThetaMethods import add_extra_input
import torch


class LogisticRegression(nn.Module):

    def __init__(self,
                 theta_params: int,
                 device="cpu",
                 randomize=False):
        super(LogisticRegression, self).__init__()
        self.__device = torch.device(device)
        self.__parameters = torch.rand((1, theta_params + 1),
                                       dtype=torch.float64,
                                       requires_grad=True,
                                       device=self.__device) if randomize else torch.zeros((1, theta_params + 1),
                                                                                           dtype=torch.float64,
                                                                                           requires_grad=True,
                                                                                           device=self.__device)

        self.__sigmoid_layer = nn.Sigmoid()

    def get_parameters(self):
        return self.__parameters

    def forward(self,
                x_input: torch.FloatTensor) -> torch.Tensor:
        return self.__sigmoid_layer(torch.matmul(add_extra_input(x_input, self.__device),
                                                 self.__parameters.T))

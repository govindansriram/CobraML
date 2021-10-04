import torch
from GeneralMethods.StatsMethods import multivariate_normal_distribution


class GaussianDiscriminantAnalysis:

    def __init__(self,
                 class_count: int,
                 feature_count: int):
        self.__class_count = class_count
        self.__feature_count = feature_count

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = torch.device(device=device)

        self.__mu_matrix = torch.zeros(size=(feature_count, class_count),
                                       dtype=torch.float64,
                                       device=self.__device)

        self.__covariance_matrix = torch.zeros(size=(feature_count, feature_count),
                                               dtype=torch.float64,
                                               device=self.__device)

        self.__fi_vector = torch.zeros(size=(1, class_count),
                                       dtype=torch.float64,
                                       device=self.__device)

        self.__total_count = 0

    def forward(self,
                x_input: torch.Tensor,
                y_target: torch.Tensor):

        x_input = x_input.to(self.__device)
        y_target = y_target.to(self.__device)

        full_tensor = torch.cat((x_input, y_target), dim=1)

        for i in range(self.__class_count):
            mask_tensor = full_tensor[full_tensor[:, self.__feature_count] == i][:, :self.__feature_count]
            self.__fi_vector[0][i] += mask_tensor.size()[0]
            sum_tensor = torch.unsqueeze(torch.sum(mask_tensor, dim=0), dim=1)
            self.__mu_matrix[:, i:i + 1] += sum_tensor

    def fit_variables(self):
        for i in range(self.__class_count):
            self.__mu_matrix[:, i] /= self.__fi_vector[1, i]

        fi_sum = torch.sum(self.__fi_vector)
        self.__total_count += fi_sum
        self.__fi_vector /= fi_sum

    def make_covariance_matrix(self,
                               x_input,
                               y_input):

        joint_tensor = torch.cat((x_input.to(self.__device),
                                  y_input.to(self.__device)),
                                 dim=1)

        for vector in joint_tensor:
            i = vector[self.__feature_count, self.__feature_count+1]
            sigma = (vector[:self.__feature_count] - self.__mu_matrix[:, i]).T
            self.__covariance_matrix += torch.mm(sigma, sigma.T)




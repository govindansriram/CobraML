import torch
from torch.utils.data import DataLoader


class BinaryGDA:

    def __init__(self,
                 feature_count: int,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 device="cpu"):
        self.__train_loader = train_data_loader
        self.__test_loader = test_data_loader

        self.__feature_count = feature_count

        self.__device = torch.device(device=device)
        self.__mu_one_vector = torch.zeros(size=(feature_count, 1),
                                           dtype=torch.float64,
                                           device=self.__device)

        self.__mu_zero_vector = torch.zeros(size=(feature_count, 1),
                                            dtype=torch.float64,
                                            device=self.__device)

        self.__co_variance_matrix = torch.zeros(size=(feature_count, feature_count),
                                                dtype=torch.float64,
                                                device=self.__device)
        self.__fi = 0

        self.__total_count = 0

    def train(self):

        one_count = 0
        zero_count = 0

        for batch in self.__train_loader:
            feature_tensor, target_tensor = batch

            feature_tensor = feature_tensor.to(self.__device)
            target_tensor = target_tensor.to(self.__device)

            full_tensor = torch.cat((feature_tensor, target_tensor), dim=1)

            mask = (full_tensor[:, self.__feature_count + 1] == 0)

            zero_matrix = full_tensor[mask][:, :self.__feature_count]
            one_matrix = full_tensor[~mask][:, :self.__feature_count]

            zero_count += zero_matrix.shape()[0]
            one_count += one_matrix.shape()[0]

            zero_sum = torch.unsqueeze(torch.sum(zero_matrix, dim=0), dim=0).T
            one_sum = torch.unsqueeze(torch.sum(one_matrix, dim=0), dim=0).T

            self.__mu_one_vector += one_sum
            self.__mu_zero_vector += zero_sum

        self.__fi = one_count / (one_count + zero_count)

        self.__mu_one_vector / one_count
        self.__mu_zero_vector / zero_count

        self.__total_count = one_count + zero_count

        self.__make_covariance_matrix()

    def __make_covariance_matrix(self):

        for batch in self.__train_loader:

            for feature_vector, target_tensor in batch:

                if target_tensor[0] is 0:
                    sigma = feature_vector - self.__mu_zero_vector
                else:
                    sigma = feature_vector - self.__mu_one_vector

                self.__co_variance_matrix += torch.mm(sigma, sigma.T)

        self.__co_variance_matrix /= self.__total_count


if __name__ == '__main__':
    mean_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    mean_tensor_2 = torch.tensor([[5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]])

    m_t_1 = torch.unsqueeze(torch.sum(mean_tensor, dim=0), dim=0).T
    m_t_2 = torch.unsqueeze(torch.sum(mean_tensor_2, dim=0), dim=0).T

    print(torch.cat((m_t_1, m_t_2), dim=1))

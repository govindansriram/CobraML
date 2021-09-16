import torch
from torch.utils.data import DataLoader
from GeneralMethods.GeneralDataset import GenericDataSet
from GeneralMethods.StatsMethods import multivariate_normal_distribution


class BinaryGDA:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[int],
                 batch_size: int,
                 device="cpu"):

        self.__feature_count = len(feature_data[0])

        data_set = GenericDataSet(torch.tensor(feature_data, dtype=torch.float),
                                  torch.tensor(target_data, dtype=torch.float))

        self.__data_loader = DataLoader(data_set,
                                        batch_size=batch_size,
                                        shuffle=True)

        self.__device = torch.device(device=device)
        self.__mu_one_vector = torch.zeros(size=(self.__feature_count, 1),
                                           dtype=torch.float64,
                                           device=self.__device)

        self.__mu_zero_vector = torch.zeros(size=(self.__feature_count, 1),
                                            dtype=torch.float64,
                                            device=self.__device)

        self.__co_variance_matrix = torch.zeros(size=(self.__feature_count, self.__feature_count),
                                                dtype=torch.float64,
                                                device=self.__device)

        self.__fi = 0
        self.__total_count = 0

    def train(self):

        one_count = 0
        zero_count = 0

        for batch in self.__data_loader:
            feature_tensor, target_tensor = batch

            feature_tensor = feature_tensor.to(self.__device)
            target_tensor = target_tensor.to(self.__device)

            full_tensor = torch.cat((feature_tensor, target_tensor), dim=1)

            mask = (full_tensor[:, self.__feature_count] == 1)

            zero_matrix = full_tensor[~mask][:, :self.__feature_count]
            one_matrix = full_tensor[mask][:, :self.__feature_count]

            zero_count += zero_matrix.size()[0]
            one_count += one_matrix.size()[0]

            zero_sum = torch.unsqueeze(torch.sum(zero_matrix, dim=0), dim=0).T
            one_sum = torch.unsqueeze(torch.sum(one_matrix, dim=0), dim=0).T

            self.__mu_one_vector += one_sum
            self.__mu_zero_vector += zero_sum

        self.__fi = one_count / (one_count + zero_count)

        self.__mu_one_vector /= one_count
        self.__mu_zero_vector /= zero_count

        self.__total_count = one_count + zero_count

        self.__make_covariance_matrix()

    def __make_covariance_matrix(self):

        for batch in self.__data_loader:

            joint_tensor = torch.cat((batch[0].to(self.__device),
                                      batch[1].to(self.__device)),
                                     dim=1)

            for joint_vector in joint_tensor:

                joint_vector = torch.unsqueeze(joint_vector, dim=0)

                if joint_vector[0][joint_vector.size()[1] - 1] == 0:
                    sigma = (joint_vector[:, :joint_vector.size()[1] - 1] - self.__mu_zero_vector.T).T
                else:
                    sigma = (joint_vector[:, :joint_vector.size()[1] - 1] - self.__mu_one_vector.T).T

                self.__co_variance_matrix += torch.mm(sigma, sigma.T)

        self.__co_variance_matrix /= self.__total_count

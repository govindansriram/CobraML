import torch
from torch.utils.data import DataLoader
from GeneralMethods.GeneralDataset import GenericDataSet
from GeneralMethods.StatsMethods import multivariate_normal_distribution


class BinaryGDA:
    """
    Initializes Mu of 0 as a 0 vector. Mu of 0 is the mean value of all the features which have a target of 0.
    Initializes Mu of 1 as a 1 vector. Mu of 1 is the mean value of all the features which have a target of 1.
    Initializes the covariance matrix(a.k.a sigma) as a 0 matrix of size n x n, n is the amount of features present
    which is the same as the length of either mu vector. Sets fi to be currently 0, fi is simply the percent of data
    samples that are 1. Sets the total count to zero since that will be computed later on.
    """

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

    """
    Here we iterate through the dataloader. From there we concatenate the target tensor to the feature tensor,
    and we generate a mask. We use the mask to split the matrix into two parts, one where all the targets are one, 
    and one where all the targets are 0. We add the lengths of each Matrix to their respective counts, and then sum 
    down the two matrices to a single vector. That vector is added to their relative mu value. Finally the averages 
    for each mu value is computed, along with the fi percentage, and total count.
    """

    def train(self):

        one_count = 0
        zero_count = 0

        for batch in self.__data_loader:
            # feature_tensor, target_tensor = batch

            # feature_tensor = feature_tensor.to(self.__device)
            # target_tensor = target_tensor.to(self.__device)

            full_tensor = torch.cat((batch[0].to(self.__device),
                                     batch[1].to(self.__device)),
                                    dim=1)

            mask = (full_tensor[:, self.__feature_count] == 1)

            zero_matrix = full_tensor[~mask][:, :self.__feature_count]
            one_matrix = full_tensor[mask][:, :self.__feature_count]

            zero_count += zero_matrix.size()[0]
            one_count += one_matrix.size()[0]

            zero_sum = torch.unsqueeze(torch.sum(zero_matrix, dim=0), dim=0).T
            one_sum = torch.unsqueeze(torch.sum(one_matrix, dim=0), dim=0).T

            self.__mu_one_vector += one_sum
            self.__mu_zero_vector += zero_sum

        self.__total_count = one_count + zero_count

        self.__fi = one_count / self.__total_count

        self.__mu_one_vector /= one_count
        self.__mu_zero_vector /= zero_count

        self.__make_covariance_matrix()

    """
    We iterate through the dataloader, and combine the feature tensor to the target tensor. We iterate through every
    row in the joint tensor and check if the target is either 0 or one. Depending on which it is we subtract it from
    it's corresponding mu tensor, multiply it by the transpose of itself, and then add that to the covariance 
    matrix. Finally we divide the covariance matrix by the total data count.
    """

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

    """
    We get the prediction of the class of each vector in the feature matrix and return the ratio of the correct number
    of classes predicted to total amount of data present in the sample.
    """

    def test_model(self,
                   feature_matrix: list[list[float]],
                   target_list: list[int]) -> torch.Tensor:

        target_tensor = torch.tensor(data=target_list,
                                     dtype=torch.int64,
                                     device=self.__device)

        pred_tensor = torch.zeros(size=target_tensor.size(),
                                  dtype=torch.int64,
                                  device=self.__device)

        for idx, feature_vector in enumerate(feature_matrix):
            pred_tensor[idx] += self.make_prediction(feature_vector)

        return torch.sum(torch.eq(pred_tensor, target_tensor)) / target_tensor.size()[0]

    """
    We get the multivariate normal distribution, which is the probability x is the feature if y is one or 0. We then
    multiply that by the respective fi which is the probability y is either one or zero. From there we return the class
    of the higher value.
    """

    def make_prediction(self, feature_vector: list[float]) -> int:

        feature_tensor = torch.tensor(data=feature_vector,
                                      dtype=torch.float64,
                                      device=self.__device)

        one_mnd = multivariate_normal_distribution(feature_tensor,
                                                   self.__mu_one_vector,
                                                   self.__co_variance_matrix)

        zero_mnd = multivariate_normal_distribution(feature_tensor,
                                                    self.__mu_zero_vector,
                                                    self.__co_variance_matrix)

        one_pred = self.__fi * one_mnd
        zero_pred = (1 - self.__fi) * zero_mnd

        return 1 if one_pred > zero_pred else 0

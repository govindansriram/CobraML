from Classification.GaussianDiscriminantAnalysis import GaussianDiscriminantAnalysis
from GeneralMethods.StatsMethods import multivariate_normal_distribution
from GeneralMethods.GeneralDataset import GenericDataSet
from torch.utils.data import DataLoader
import torch


class GDA:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[list[int]],
                 num_of_classes: int,
                 batch_size: int):

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_count = len(feature_data[0])

        feature_tensor = torch.tensor(feature_data)
        target_tensor = torch.tensor(target_data)

        dataset = GenericDataSet(feature_tensor, target_tensor)

        self.___class_count = num_of_classes

        self.__dataloader = DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True)

        self.__model = GaussianDiscriminantAnalysis(num_of_classes,
                                                    feature_count)

    def train_model(self):
        for data, label in self.__dataloader:
            self.__model.forward(data,
                                 label)

        self.__model.fit_variables()

        for data, label in self.__dataloader:
            self.__model.make_covariance_matrix(data, label)

        self.__model.fit_covariance_matrix()

    def get_prediction(self, feature_data: list[float]) -> torch.Tensor:

        pred_tensor = torch.zeros(size=(1, self.___class_count),
                                  dtype=torch.float64,
                                  device=self.__device)

        feature_tensor = torch.tensor(feature_data,
                                      device=self.__device)

        for i in range(self.___class_count):
            mnd_prob = multivariate_normal_distribution(feature_tensor,
                                                        torch.unsqueeze(self.__model.get_mu_matrix()[:, i], dim=1),
                                                        self.__model.get_covariance_matrix())

            pred_tensor[0][i] += mnd_prob[0][0] * self.__model.get_fi_vector()[0][i]

        return pred_tensor

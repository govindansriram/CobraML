import torch
from torch.utils.data import DataLoader
from Classification.NaiveBayes import NaiveBayes
from GeneralMethods.GeneralDataset import GenericDataSet


class MultinomialNaiveBayes(NaiveBayes):

    def __init__(self,
                 batch_size: int,
                 x_features: list[list[float]],
                 y_classes: list[list[int]],
                 class_count):

        super().__init__(class_count, len(x_features[0]))

        feature_tensor = torch.tensor(data=x_features,
                                      dtype=torch.float64,
                                      device=self.get_device())

        target_tensor = torch.tensor(data=y_classes,
                                     dtype=torch.int64,
                                     device=self.get_device())

        dataset = GenericDataSet(feature_tensor,
                                 target_tensor)

        self.__data_loader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

        self.__log_class_priors = torch.zeros(size=self.get_class_sum_tensor().size(),
                                              device=self.get_device())

        self.__log_prob_matrix = torch.zeros(size=self.get_count_matrix().size(),
                                             device=self.get_device())

    def train(self,
              laplace_smoothing=0.0):

        for batch in self.__data_loader:
            feature_tensor, label_tensor = batch

            feature_tensor = feature_tensor.to(self.get_device())
            label_tensor = label_tensor.to(self.get_device())

            self.forward(feature_tensor,
                         label_tensor)

        self.__log_class_priors += torch.log(self.get_class_sum_tensor() / torch.sum(self.get_class_sum_tensor()))

        self.set_count_matrix(self.get_count_matrix() + laplace_smoothing)

        self.__log_prob_matrix += torch.log(
            self.get_count_matrix() / torch.sum(self.get_count_matrix(), dim=0)
        )

    def predict(self,
                feature_list: list[int]) -> torch.Tensor:
        feature_vector = torch.unsqueeze(torch.tensor(data=feature_list,
                                                      dtype=torch.float64,
                                                      device=self.get_device()), dim=1)

        return torch.sum(feature_vector * self.__log_prob_matrix, dim=0) + self.__log_class_priors

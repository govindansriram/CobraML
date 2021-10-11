import torch
from torch.utils.data import DataLoader
from Classification.NaiveBayes import NaiveBayes
from GeneralMethods.GeneralDataset import GenericDataSet


class BernoulliNaiveBayes(NaiveBayes):

    def __init__(self,
                 batch_size: int,
                 x_features: list[list[int]],
                 y_classes: list[list[int]],
                 class_count):

        super().__init__(class_count, len(x_features[0]))

        feature_tensor = torch.tensor(data=x_features,
                                      dtype=torch.int64,
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

        self.__prob_matrix = torch.zeros(size=self.get_count_matrix().size(),
                                         device=self.get_device())

    def train(self, laplace_smoothing=0.0):
        for batch in self.__data_loader:
            feature_tensor, label_tensor = batch

            feature_tensor = feature_tensor.to(self.__device)
            label_tensor = label_tensor.to(self.__device)

            self.forward(feature_tensor,
                         label_tensor)

        self.__log_class_priors += torch.log(self.get_class_sum_tensor() / torch.sum(self.get_class_sum_tensor()))

        self.set_count_matrix(self.get_count_matrix() + laplace_smoothing)

        self.__prob_matrix += self.get_count_matrix() / (self.get_class_sum_tensor() + (2 * laplace_smoothing))

    def predict(self,
                feature_list: list[int]):

        log_tensor = torch.zeros(size=self.get_class_sum_tensor().size(),
                                 device=self.__device)

        feature_vector = torch.tensor(data=feature_list,
                                      dtype=torch.float64,
                                      device=self.__device)

        feature_tensor = torch.unsqueeze(feature_vector, dim=1)

        full_mat = torch.cat((self.get_count_matrix(), feature_tensor), dim=1)

        tensor_mask = (full_mat[:, self.get_class_sum_tensor().size()[0]] == 1)

        one_mat = full_mat[tensor_mask]
        zero_mat = full_mat[~tensor_mask]

        log_tensor += torch.sum(torch.log(one_mat), dim=0)

        zero_mat = 1 - zero_mat

        log_tensor += torch.sum(torch.log(zero_mat), dim=0)

        return log_tensor + self.__log_class_priors



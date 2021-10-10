import torch
from torch.utils.data import DataLoader
from Classification.NaiveBayes import NaiveBayes
from GeneralMethods.GeneralDataset import GenericDataSet


class MultinomialNaiveBayes:

    def __init__(self,
                 num_of_classes: int,
                 batch_size: int,
                 x_features: list[list[float]],
                 y_classes: list[list[int]]):

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_tensor = torch.tensor(data=x_features,
                                      dtype=torch.float64,
                                      device=self.__device)

        target_tensor = torch.tensor(data=y_classes,
                                     dtype=torch.int64,
                                     device=self.__device)

        dataset = GenericDataSet(feature_tensor,
                                 target_tensor)

        self.__data_loader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

        self.__model = NaiveBayes(num_of_classes,
                                  len(x_features[0]))

        self.__class_count = num_of_classes

    def train(self):

        for batch in self.__data_loader:
            feature_tensor, label_tensor = batch

            feature_tensor = feature_tensor.to(self.__device)
            label_tensor = label_tensor.to(self.__device)

            self.__model.forward(feature_tensor,
                                 label_tensor)

    def predict(self,
                feature_list: list[int],
                laplace_smoothing=0.0) -> torch.Tensor:

        prob_tensor = torch.zeros(size=self.__model.get_class_sum_tensor().size(),
                                  device=self.__device)

        feature_vector = torch.tensor(data=feature_list,
                                      dtype=torch.float64,
                                      device=self.__device)

        prob_tensor += self.__model.get_class_sum_tensor() / torch.sum(self.__model.get_class_sum_tensor())

        for idx, value in enumerate(feature_vector):

            if value > 0:
                value_tensor = self.__model.get_count_matrix()[idx] + laplace_smoothing
                value_tensor /= (torch.sum(self.__model.get_count_matrix(),
                                           dim=0) + (laplace_smoothing * self.__model.get_feature_count()))
                prob_tensor *= torch.pow(value_tensor, value)

        prediction_tensor = torch.zeros(size=(1, self.__class_count),
                                        device=self.__device)

        prob_sum = torch.sum(prob_tensor)

        for i in range(self.__class_count):
            prediction_tensor[0][i] += prob_tensor[0][i] / prob_sum

        return prediction_tensor

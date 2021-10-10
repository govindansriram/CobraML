import torch


class NaiveBayes:

    def __init__(self,
                 class_count,
                 feature_count):

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__class_count = class_count
        self.__class_sum_tensor = torch.zeros(size=(1, self.__class_count),
                                              device=self.__device)
        self.__feature_count = feature_count
        self.__count_matrix = torch.zeros(size=(feature_count, class_count),
                                          device=self.__device)

    def get_class_sum_tensor(self):
        return self.__class_sum_tensor

    def get_count_matrix(self):
        return self.__count_matrix

    def get_feature_count(self):
        return self.__feature_count

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor):

        full_features = torch.cat((features.to(self.__device),
                                   labels.to(self.__device)), dim=1)

        for i in range(self.__class_count):
            class_features = full_features[full_features[:, features.size()[1]] == i][:, :self.__feature_count]
            self.__class_sum_tensor[:, i] += class_features.size()[0]
            self.__count_matrix[:, i] += torch.sum(class_features, dim=0)

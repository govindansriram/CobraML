import torch


class MultinomialNaiveBayes:

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

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor):

        full_features = torch.cat((features.to(self.__device),
                                   labels.to(self.__device)), dim=1)

        for i in range(self.__class_count):
            class_features = full_features[full_features[:, features.size()[1]] == i][:, :self.__feature_count]
            self.__class_sum_tensor[:, i] += class_features.size()[0]
            self.__count_matrix[:, i] += torch.sum(class_features, dim=0)

    def predict(self,
                feature_vector: torch.Tensor,
                laplace_smoothing=0.0) -> torch.Tensor:

        prob_tensor = torch.zeros(size=self.__class_sum_tensor.size(),
                                  device=self.__device)

        prob_tensor += self.__class_sum_tensor / torch.sum(self.__class_sum_tensor)

        for idx, value in enumerate(feature_vector):

            if value > 0:
                value_tensor = self.__count_matrix[idx] + laplace_smoothing
                value_tensor /= (torch.sum(self.__count_matrix, dim=1) / (laplace_smoothing * self.__feature_count))
                prob_tensor = torch.mm(prob_tensor,
                                       torch.unsqueeze(torch.pow(value_tensor, value), dim=1))

        prediction_tensor = torch.zeros(size=(1, self.__class_count))

        prob_sum = torch.sum(prob_tensor)

        for i in range(self.__class_count):
            prediction_tensor[i] += prob_tensor[i] / prob_sum

        return prediction_tensor


if __name__ == '__main__':
    feature_data = torch.randint(0, 2, (10, 10))

    label_data = torch.randint(0, 3, (10, 1))

    full_data = torch.cat((feature_data, label_data), dim=1)

    print(full_data)

    model = MultinomialNaiveBayes(3, 10)

    model.forward(feature_data, label_data)

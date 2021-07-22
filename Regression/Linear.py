import torch
from CostFunctions.LossFunctions import squared_error
from Optimizers.GradientDescent import batch_grad


class LinearRegression:

    def __init__(self,
                 feature_list,
                 target_list,
                 epochs,
                 learning_rate):
        self.__feature_list = torch.cat((torch.ones(len(feature_list), 1), torch.FloatTensor(feature_list)), 1)
        self.__target_list = torch.FloatTensor(target_list)
        self.__parameters = torch.rand(1, self.__feature_list.size()[1])
        self.__epochs = epochs
        self.__learning_rate = learning_rate

        self.__error_list = []

    def __get_predictions(self):
        prediction = torch.matmul(self.__feature_list, torch.transpose(self.__parameters, 0, 1))
        return prediction

    def train_model(self):
        for epoch in range(self.__epochs):
            predictions = torch.squeeze(torch.transpose(self.__get_predictions(), 0, 1), dim=0)

            error = squared_error(predictions, self.__target_list)
            self.__error_list.append(error)

            self.__parameters = batch_grad(self.__learning_rate,
                                           self.__feature_list,
                                           self.__target_list,
                                           predictions,
                                           self.__parameters)

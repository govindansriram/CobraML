import torch


def make_feature_list(x_feature_list):
    return torch.cat((torch.ones(len(x_feature_list), 1), torch.FloatTensor(x_feature_list)), 1)


class LinearRegression:

    def __init__(self,
                 feature_list,
                 target_list):
        self.__feature_list = make_feature_list(feature_list)
        self.__target_list = torch.FloatTensor(target_list)
        self.__parameters = torch.rand(1, self.__feature_list.size()[1])

    def get_predictions(self):
        prediction = torch.matmul(self.__feature_list, torch.transpose(self.__parameters, 0, 1))
        return torch.squeeze(torch.transpose(prediction, 0, 1), dim=0)

    def normal_eqt(self):
        mat_1 = torch.matmul(torch.transpose(self.__feature_list, 0, 1), self.__feature_list)
        target = torch.unsqueeze(self.__target_list, dim=0)
        mat_2 = torch.matmul(torch.transpose(self.__feature_list, 0, 1), torch.transpose(target, 0, 1))
        return torch.unsqueeze(torch.squeeze(torch.matmul(torch.inverse(mat_1), mat_2), dim=1), dim=0)

    def get_parameters(self):
        return self.__parameters

    def get_target_list(self):
        return self.__target_list

    def get_feature_list(self):
        return self.__feature_list

    def update_parameters(self, theta_param_list):
        self.__parameters = theta_param_list

    def set_target_list(self, y_target_list):
        self.__target_list = y_target_list

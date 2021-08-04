import numpy as np


def make_feature_list(x_array: np.ndarray):
    ones_arr = np.ones((x_array.shape[0], 1), dtype=np.float64)
    x_array = x_array.astype('float64')

    return np.concatenate((ones_arr, x_array), axis=1)


class LinearRegression:

    def __init__(self,
                 feature_arr: np.ndarray,
                 target_arr: np.ndarray,
                 param_arr_rand=True):

        self.__feature_list = make_feature_list(feature_arr)
        self.__target_list = target_arr

        if param_arr_rand:
            self.__parameters = np.random.rand(1,
                                               self.__feature_list.shape[1])
        else:
            self.__parameters = np.zeros((1, self.__feature_list.shape[1]),
                                         dtype=np.float64)

    def get_predictions(self,
                        new_feat=False,
                        feat_arr=None):

        curr_arr = self.__feature_list if not new_feat else make_feature_list(feat_arr)

        return np.squeeze(
            np.matmul(curr_arr, self.__parameters.T),
            axis=1)

    def normal_eqt(self):
        mat_1 = np.matmul(self.__feature_list.T, self.__feature_list)
        target = np.expand_dims(self.__target_list, axis=0)
        mat_2 = np.matmul(self.__feature_list.T, target.T)
        mat_3 = np.matmul(np.linalg.inv(mat_1), mat_2)

        return np.expand_dims(np.squeeze(mat_3, axis=1), axis=0)

    def get_parameters(self):
        return self.__parameters

    def get_target_list(self):
        return self.__target_list

    def get_feature_list(self):
        return self.__feature_list

    def update_parameters(self, theta_param_list: np.ndarray):
        self.__parameters = theta_param_list

    def set_target_list(self, y_target_list: np.ndarray):
        self.__target_list = y_target_list

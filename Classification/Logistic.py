import numpy as np
from Regression.Linear import LinearRegression


def sigmoid(pred: np.ndarray) -> np.ndarray:
    ones = np.ones(pred.shape, dtype=np.float64)
    sig_arr = np.exp(pred * -1) + ones
    return ones / sig_arr


class LogisticRegression(LinearRegression):

    def __init__(self,
                 feature_arr: np.ndarray,
                 target_arr: np.ndarray,
                 param_arr_rand=True):
        super().__init__(feature_arr, target_arr, param_arr_rand)

    def get_predictions(self,
                        new_feat=False,
                        feat_arr=None) -> np.ndarray:
        return sigmoid(super(LogisticRegression, self).get_predictions(new_feat=new_feat, feat_arr=feat_arr))

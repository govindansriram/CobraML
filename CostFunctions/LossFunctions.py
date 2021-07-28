import numpy as np


def squared_error(pred_tensor: np.ndarray,
                  output_tensor: np.ndarray):

    diff_arr = np.subtract(pred_tensor, output_tensor)
    sum_scalar = np.sum(np.square(diff_arr))
    return sum_scalar * (1 / (2 * pred_tensor.shape[0]))

import numpy as np


def squared_error(pred_tensor: np.ndarray,
                  output_tensor: np.ndarray):

    diff_arr = np.subtract(pred_tensor, output_tensor)
    sum_scalar = np.sum(np.square(diff_arr))
    return sum_scalar * (1 / (2 * pred_tensor.shape[0]))


def binary_cross_entropy(pred_array: np.ndarray,
                         output_array: np.ndarray) -> np.float64:

    hyp_one_mat = np.matmul(output_array, np.log10(pred_array).T)
    hyp_zed_mat = np.ones(shape=output_array.shape, dtype=np.float64) - output_array
    log_mat = np.ones(shape=pred_array.shape, dtype=np.float64) - pred_array
    hyp_zed_mat = np.matmul(hyp_zed_mat, np.log10(log_mat).T)

    return np.sum(hyp_one_mat + hyp_zed_mat) / (pred_array.shape[0] * -1)

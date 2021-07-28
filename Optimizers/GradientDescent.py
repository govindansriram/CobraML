import numpy as np


def batch_grad(learning_rate: float,
               inputs: np.ndarray,
               targets: np.ndarray,
               predictions: np.ndarray,
               parameters: np.ndarray):

    updated_param = np.zeros(parameters.shape, dtype=np.float64)

    diff_arr = np.subtract(predictions, targets)

    for idx, theta in enumerate(parameters):
        mul_arr = np.matmul(diff_arr, inputs[:, idx:idx + 1])
        updated_param[idx] += theta - learning_rate / predictions.shape[0] * np.sum(mul_arr, dtype=np.float64)

    return updated_param


def stochastic_grad(learning_rate: float,
                    x_input: np.ndarray,
                    y_output: float,
                    prediction: float,
                    parameters: np.ndarray):

    updated_param = np.zeros(parameters.shape, dtype=np.float64)

    for idx in range(parameters.shape[0]):
        updated_param[idx] += parameters[idx] - (learning_rate * (prediction - y_output) * x_input[idx])

    return updated_param

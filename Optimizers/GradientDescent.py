import torch


def batch_grad(learning_rate, inputs, targets, predictions, parameters):
    updated_param = []

    diff_tensor = torch.subtract(predictions, targets)
    for idx, theta in enumerate(parameters):
        mul_tensor = torch.matmul(diff_tensor, inputs[:, idx:idx + 1])
        updated_param.append(theta - (learning_rate * (torch.sum(mul_tensor).item() / len(predictions))))

    return torch.FloatTensor(updated_param)


def stochastic_grad(learning_rate, x_input, y_output, predictions, theta_params):

    updated_param = [
        theta - (learning_rate * (predictions - y_output) * x_input[idx]) for idx, theta in enumerate(theta_params)
    ]

    return torch.FloatTensor(updated_param)

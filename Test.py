import torch
from Regression.Linear import LinearRegression

feature_vector = [
        [1, 2, 5, 4, 5],
        [1, 2, 3, 4, 1],
        [1, 2, 8, 4, 5]
    ]
target_vector = [1, 2, 3, 4, 5, ]

p_tensor = torch.FloatTensor([213, 123, 123, 5423])
t_tensor = torch.FloatTensor([123, 12, 453, 563])

# lin_reg = LinearRegression(feature_vector, target_vector, 3, 0.01)

print(torch.FloatTensor(feature_vector)[:, 0:1])

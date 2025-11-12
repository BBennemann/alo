import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features_entrada):
        super(Model, self).__init__()

        self.linear = nn.Linear(num_features_entrada, 1)

    def forward(self, x):
        linear_output = self.linear(x)
        probability = torch.sigmoid(linear_output)
        return probability
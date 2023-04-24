from util import parse_datafile
import torch
import numpy as np
import torch.nn as nn


class SPOplusFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        pass

    @staticmethod
    def backward(ctx, *args):
        pass


class SPOplus(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.spoPlusFunction = SPOplusFunction()

    def forward(self, *args):
        return self.spoPlusFunction.apply(*args)


class CostPredictor(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)



if __name__ == "__main__":
    vrps_train = [parse_datafile(f'data/generated_1000_25/instance_{i}') for i in range(900)]
    vrps_test = [parse_datafile(f'data/generated_1000_25/instance_{i}') for i in range(900, 1000)]


    cost_model = CostPredictor(len(vrps_train[0].edges[0].features), 32, 1)
    optimizer = torch.optim.Adam(cost_model.parameters(), lr=0.001)
    spo_plus = SPOplus()




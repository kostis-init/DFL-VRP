import torch
import numpy as np
import torch.nn as nn


class SPOplusFunction(torch.autograd.Function):
    """
    The SPO+ function as described in the paper Smart “Predict, then Optimize" https://arxiv.org/abs/1710.08005
    """

    @staticmethod
    def forward(ctx, pred_costs, true_decision, true_obj, spo_decision, spo_obj):

        loss = spo_obj + 2 * np.dot(pred_costs, true_decision) - true_obj
        loss = torch.Tensor(loss).to(pred_costs.device)

        # print('FORWARD')
        # print(f'pred_costs: {pred_costs}')
        # print(f'true_decision: {true_decision}')
        # print(f'true_obj: {true_obj}')
        # print(f'spo_decision: {spo_decision}')
        # print(f'spo_obj: {spo_obj}')
        # print(f'loss: {loss}')

        ctx.save_for_backward(torch.FloatTensor(true_decision).to(pred_costs.device),
                              torch.FloatTensor(spo_decision).to(pred_costs.device))
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        # grad_loss is a scalar tensor containing the gradient of the loss with respect to the output of the forward
        # function (i.e. the loss) (dL/dloss) = 1 in this case (see https://pytorch.org/docs/stable/notes/extending.html#gradients)

        true_decision, spo_decision = ctx.saved_tensors
        # print('BACKWARD')
        # print(f'grad_loss: {grad_loss}')
        # print(f'true_decision: {true_decision}')
        # print(f'spo_decision: {spo_decision}')
        # print(f'result: {grad_loss * 2 * (true_decision - spo_decision)}')
        return grad_loss * 2 * (true_decision - spo_decision), None, None, None, None


class SPOplus(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.spoPlusFunction = SPOplusFunction()

    def forward(self, *args):
        return self.spoPlusFunction.apply(*args)



class CostPredictor(torch.nn.Module):
    """
    A simple neural network that predicts the edge costs of a VRP.

    The input is a vector of edge features, the output is a vector of edge costs.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        return x

import torch
import numpy as np
import torch.nn as nn


class NCE(torch.nn.Module):

    def __init__(self, true_sols):
        super().__init__()
        self.pool = true_sols
        # print(f'pool: {self.pool}')

    def forward(self, pred_cost, true_cost, true_sol, vrp):
        # init loss to zero tensor
        loss = 0
        for non_optimal_sol in self.pool[vrp]:
            # print(f'non_optimal_sol: {non_optimal_sol}')
            # print(f'true_sol: {true_sol}')
            # print(f'pred_cost: {pred_cost}')
            # print(f'true_cost: {true_cost}')
            loss += torch.dot((pred_cost - true_cost), (torch.DoubleTensor(np.array(true_sol)) - torch.DoubleTensor(non_optimal_sol)))
            # loss += torch.dot(pred_cost, (torch.FloatTensor(true_sol) - torch.FloatTensor(non_optimal_sol)))

        return loss


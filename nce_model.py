import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from enums import SolverMode
from util import test
import torch.nn.functional as F


class NCETrueCostLoss(torch.nn.Module):

    def __init__(self, true_sols):
        super().__init__()
        self.pool = true_sols

    def forward(self, pred_cost, true_cost, true_sol, vrp):
        loss = 0
        for sol in self.pool[vrp]:
            loss += torch.dot((pred_cost - true_cost),
                              (torch.DoubleTensor(np.array(true_sol)) - torch.DoubleTensor(sol)))

        return loss


class NCELoss(torch.nn.Module):

    def __init__(self, true_sols):
        super().__init__()
        self.pool = true_sols

    def forward(self, pred_cost, true_cost, true_sol, vrp):
        loss = 0
        for sol in self.pool[vrp]:
            loss += torch.dot(pred_cost, (torch.FloatTensor(true_sol) - torch.FloatTensor(sol)))

        return loss


class CostPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc1(x)
        return x


class NCEModel:

    def __init__(self, vrps_train, vrps_val, vrps_test, lr=1e-4, solver_class=None):
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = CostPredictor(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr)

        self.criterion = NCELoss({vrp: [vrp.actual_solution] for vrp in vrps_train})

        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test

        if solver_class is None:
            raise Exception('Solver class must be specified')
        self.solver_class = solver_class

    def train(self, epochs=20, verbose=False, test_every=5):
        self.cost_model.train()
        for epoch in range(epochs):
            mean_loss = 0
            if verbose:
                loop = enumerate(self.vrps_train)
            else:
                loop = tqdm(enumerate(self.vrps_train))
            for idx, vrp in loop:
                # reset the gradients
                self.optimizer.zero_grad()
                # get the edge features
                edge_features = torch.tensor([edge.features for edge in vrp.edges])
                # predict the edge costs
                predicted_edge_costs = self.cost_model(edge_features)
                # set the predicted edge costs
                for i, edge in enumerate(vrp.edges):
                    edge.predicted_cost = predicted_edge_costs[i].detach().item()

                # add to pool with probability 0.05
                if np.random.rand() < 0.1:
                    solver = self.solver_class(vrp, mode=SolverMode.PRED_COST)
                    solver.solve()
                    self.criterion.pool[vrp].append(solver.get_decision_variables())

                true_costs = torch.tensor([edge.cost for edge in vrp.edges])
                loss = self.criterion(predicted_edge_costs, true_costs, vrp.actual_solution, vrp)
                # backpropagation
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item()
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, instance {idx + 1}/{len(self.vrps_train)}, loss: {loss.item()}')
            mean_loss /= len(self.vrps_train)
            print(
                f'Epoch {epoch + 1} / {epochs} done, mean loss: {mean_loss}')
            if (epoch + 1) % test_every == 0:
                test(self.cost_model, self.vrps_test, self.solver_class, is_two_stage=False)

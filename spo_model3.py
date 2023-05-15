import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from enums import SolverMode
from util import test


class SPOplusFunction(torch.autograd.Function):
    """
    The SPO+ function as described in the paper Smart “Predict, then Optimize" https://arxiv.org/abs/1710.08005
    """

    @staticmethod
    def forward(ctx, pred_costs, true_decision, true_obj, spo_decision, spo_obj):
        pred_obj_of_true_decision = torch.dot(pred_costs, torch.Tensor(true_decision))
        loss = 2 * pred_obj_of_true_decision - true_obj - spo_obj

        # print('FORWARD')
        # print(f'pred_costs: {pred_costs}')
        # print(f'true_decision: {true_decision}')
        # print(f'true_obj: {true_obj}')
        # print(f'spo_decision: {spo_decision}')
        # print(f'spo_obj: {spo_obj}')
        # print(f'loss: {loss}')

        ctx.save_for_backward(torch.Tensor(true_decision), torch.Tensor(spo_decision))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        true_decision, spo_decision = ctx.saved_tensors
        # print('BACKWARD')
        # print(f'grad_loss: {grad_loss}')
        # print(f'true_decision: {true_decision}')
        # print(f'spo_decision: {spo_decision}')
        # print(f'result: {grad_loss * 2 * (true_decision - spo_decision)}')
        return grad_output * 2 * (true_decision - spo_decision), None, None, None, None


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
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.view(-1)
        # x = self.fc1(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.relu(x)
        return x


class SPOModel3:

    def __init__(self, vrps_train, vrps_val, vrps_test, lr=1e-4, solver_class=None):
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = CostPredictor(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr)
        self.criterion = SPOplus()
        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test

        if solver_class is None:
            raise Exception('Solver class must be specified')
        self.solver_class = solver_class

    def train(self, epochs=20, verbose=False, test_every=5):

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
                # calculate the loss
                solver = self.solver_class(vrp, mode=SolverMode.SPO)
                solver.solve()
                loss = self.criterion(predicted_edge_costs, vrp.actual_solution, vrp.actual_obj,
                                      solver.get_decision_variables(), solver.get_spo_objective())
                # backpropagation
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item()
                if verbose:
                    print(f'Epoch {epoch + 1} / {epochs}, instance {idx + 1} / {len(self.vrps_train)}, loss: {loss.item()}')
            mean_loss /= len(self.vrps_train)
            print(
                f'Epoch {epoch + 1} / {epochs} done, mean loss: {mean_loss}, validation loss: {self.validation_loss()}')
            if (epoch + 1) % test_every == 0:
                test(self.cost_model, self.vrps_test, self.solver_class, is_two_stage=False)

    def validation_loss(self):
        with torch.no_grad():
            loss = 0.0
            for vrp in self.vrps_val:
                edge_features = torch.tensor([edge.features for edge in vrp.edges])
                predicted_edge_costs = self.cost_model(edge_features)
                for i, edge in enumerate(vrp.edges):
                    edge.predicted_cost = predicted_edge_costs[i].detach().item()
                solver = self.solver_class(vrp, mode=SolverMode.SPO)
                solver.solve()
                loss += self.criterion(predicted_edge_costs, vrp.actual_solution, vrp.actual_obj,
                                       solver.get_decision_variables(), solver.get_spo_objective())
            return loss / len(self.vrps_val)

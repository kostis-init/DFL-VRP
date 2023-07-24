import torch
import torch.nn as nn
from tqdm import tqdm

from dfl_vrp.enums import SolverMode
from dfl_vrp.util import test
import torch.nn.functional as F


# TODO: try that without custom backward

class SimpleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_costs, true_sol, pred_sol):
        loss = torch.dot(pred_costs, torch.Tensor(true_sol) - torch.FloatTensor(pred_sol))
        ctx.save_for_backward(torch.Tensor(true_sol), torch.Tensor(pred_sol))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        true_sol, pred_sol = ctx.saved_tensors
        return grad_output * (true_sol - pred_sol), None, None


class Simple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = SimpleFunction()

    def forward(self, *args):
        return self.fn.apply(*args)


class CostPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.Softplus()

    def forward(self, x):
        out = x.view(-1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.activation(out)
        return out

class MyModel:

    def __init__(self, vrps_train, vrps_val, vrps_test, lr=1e-4, solver_class=None, solve_prob=0.5):
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = CostPredictor(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=1e-4)

        self.criterion = Simple()

        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test
        self.solve_prob = solve_prob
        self.pool = {vrp: [vrp.actual_solution] for vrp in vrps_train}

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
                edge_features = torch.tensor([edge.features for edge in vrp.edges], dtype=torch.float32)
                # predict the edge costs
                predicted_edge_costs = self.cost_model(edge_features)
                # set the predicted edge costs
                for i, edge in enumerate(vrp.edges):
                    edge.predicted_cost = predicted_edge_costs[i].detach().item()

                # TODO: add probability of solving (caching)
                solver = self.solver_class(vrp, mode=SolverMode.PRED_COST)
                solver.solve()

                loss = self.criterion(predicted_edge_costs, vrp.actual_solution, solver.get_decision_variables())
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
                test(self.cost_model, self.vrps_test, is_two_stage=False)

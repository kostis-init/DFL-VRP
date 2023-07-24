import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from dfl_vrp.cost_models import CostPredictorLinear, EncoderDecoder
from dfl_vrp.enums import SolverMode
from dfl_vrp.util import test
from dfl_vrp.util import set_predicted_costs, get_edge_features


class SPOplusFunction(torch.autograd.Function):
    """
    The SPO+ function as described in the paper Smart “Predict, then Optimize" https://arxiv.org/abs/1710.08005
    """
    @staticmethod
    def forward(ctx, pred_costs, true_decision, true_obj, spo_decision, spo_obj):
        loss = 2 * torch.dot(pred_costs, torch.Tensor(true_decision)) - true_obj - spo_obj
        ctx.save_for_backward(torch.Tensor(true_decision), torch.Tensor(spo_decision))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        true_decision, spo_decision = ctx.saved_tensors
        return grad_output * 2 * (true_decision - spo_decision), None, None, None, None


class SPOplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spoPlusFunction = SPOplusFunction()

    def forward(self, *args):
        return self.spoPlusFunction.apply(*args)


class SPOModel:

    def __init__(self, vrps_train, vrps_val, vrps_test, lr=1e-4, solver_class=None, solve_prob=1, weight_decay=1e-4):
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)

        # self.cost_model = CostPredictorLinear(num_edges * num_features, num_edges)
        self.cost_model = EncoderDecoder(num_edges * num_features, 32, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = SPOplus()
        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test
        self.solve_prob = solve_prob
        self.pool = {vrp: [vrp.actual_routes] for vrp in vrps_train}

        if solver_class is None:
            raise Exception('Solver class must be specified')
        self.solver_class = solver_class

    def train(self, epochs=20, verbose=False, test_every=5):
        self.cost_model.train()
        for epoch in range(epochs):
            total_loss = 0
            if verbose:
                loop = enumerate(self.vrps_train)
            else:
                loop = tqdm(enumerate(self.vrps_train))
            for idx, vrp in loop:
                self.optimizer.zero_grad()
                edge_features = get_edge_features(vrp.edges)
                predicted_edge_costs = self.cost_model(edge_features)
                set_predicted_costs(vrp.edges, predicted_edge_costs)
                obj, sol = self.get_solution(vrp)
                loss = self.criterion(predicted_edge_costs.squeeze(), vrp.actual_solution, vrp.actual_obj, sol, obj)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, instance {idx + 1}/{len(self.vrps_train)}, loss: {loss.item()}')
            print(f'Epoch {epoch + 1} / {epochs} done, mean loss: {total_loss / len(self.vrps_train)},'
                  f' validation loss: {self.validation_loss()}')
            if (epoch + 1) % test_every == 0:
                test(self.cost_model, self.vrps_test, is_two_stage=False)

    def get_solution(self, vrp):
        sol = None
        obj = float('inf')
        if np.random.rand() < self.solve_prob:
            solver = self.solver_class(vrp, mode=SolverMode.SPO)
            solver.solve()
            sol = solver.get_decision_variables()
            obj = solver.get_spo_objective()
            self.pool[vrp].append(solver.get_routes())
        else:
            for routes in self.pool[vrp]:
                route_cost = sum(vrp.route_spo_cost(route) for route in routes)
                if route_cost < obj:
                    sol = vrp.get_decision_variables(routes)
                    obj = route_cost
        return obj, sol

    def validation_loss(self):
        self.cost_model.eval()
        with torch.no_grad():
            loss = 0.0
            for vrp in self.vrps_val:
                predicted_edge_costs = self.cost_model(get_edge_features(vrp.edges))
                set_predicted_costs(vrp.edges, predicted_edge_costs)
                solver = self.solver_class(vrp, mode=SolverMode.SPO)
                solver.solve()
                loss += self.criterion(predicted_edge_costs.squeeze(), vrp.actual_solution, vrp.actual_obj,
                                       solver.get_decision_variables(), solver.get_spo_objective()).item()
            return loss / len(self.vrps_val)

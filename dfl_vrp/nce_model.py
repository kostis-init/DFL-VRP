import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from dfl_vrp.cost_models import CostPredictorLinear
from dfl_vrp.enums import SolverMode
from dfl_vrp.util import test
from dfl_vrp.util import get_edge_features, set_predicted_costs


class NCETrueCostLoss(torch.nn.Module):

    def __init__(self, true_sols):
        super().__init__()
        self.pool = true_sols

    def forward(self, pred_cost, true_cost, true_sol, vrp):
        loss = 0
        for sol in self.pool[vrp]:
            loss += torch.dot((pred_cost - true_cost),
                              (torch.FloatTensor(np.array(true_sol)) - torch.FloatTensor(sol)))

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


class NCEModel:

    def __init__(self, vrps_train, vrps_val, vrps_test, lr=1e-4, solver_class=None, solve_prob=0.5,
                 include_true_costs=True, weight_decay=0.0):
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = CostPredictorLinear(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=weight_decay)

        if include_true_costs:
            self.criterion = NCETrueCostLoss({vrp: [vrp.actual_solution] for vrp in vrps_train})
        else:
            self.criterion = NCELoss({vrp: [vrp.actual_solution] for vrp in vrps_train})

        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test
        self.solve_prob = solve_prob

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
                self.optimizer.zero_grad()
                edge_features = get_edge_features(vrp.edges)
                predicted_edge_costs = self.cost_model(edge_features)
                set_predicted_costs(vrp.edges, predicted_edge_costs)

                # add to pool with probability
                if np.random.rand() < self.solve_prob:
                    solver = self.solver_class(vrp, mode=SolverMode.PRED_COST)
                    solver.solve()
                    self.criterion.pool[vrp].append(solver.get_decision_variables())

                true_costs = torch.tensor([edge.cost for edge in vrp.edges], dtype=torch.float32)
                loss = self.criterion(predicted_edge_costs, true_costs, vrp.actual_solution, vrp)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, instance {idx + 1}/{len(self.vrps_train)}, loss: {loss.item()}')
            mean_loss /= len(self.vrps_train)
            print(
                f'Epoch {epoch + 1} / {epochs} done, mean loss: {mean_loss}')
            if (epoch + 1) % test_every == 0:
                test(self.cost_model, self.vrps_test, is_two_stage=False)

    def validation_loss(self):
        self.cost_model.eval()
        with torch.no_grad():
            loss = 0.0
            for vrp in self.vrps_val:
                predicted_edge_costs = self.cost_model(get_edge_features(vrp.edges))
                set_predicted_costs(vrp.edges, predicted_edge_costs)

                # add to pool with probability
                if np.random.rand() < self.solve_prob:
                    solver = self.solver_class(vrp, mode=SolverMode.PRED_COST)
                    solver.solve()
                    self.criterion.pool[vrp].append(solver.get_decision_variables())

                true_costs = torch.tensor([edge.cost for edge in vrp.edges], dtype=torch.float32)
                loss += self.criterion(predicted_edge_costs, true_costs, vrp.actual_solution, vrp).item()
            return loss / len(self.vrps_val)

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from dfl_vrp.cost_models import CostPredictorLinear, EncoderDecoder
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

    def forward(self, pred_cost, true_sol, vrp):
        loss = 0
        for sol in self.pool[vrp]:
            loss += torch.dot(pred_cost, (torch.FloatTensor(true_sol) - torch.FloatTensor(sol)))

        return loss


class NCEModel:

    def __init__(self, vrps_train, vrps_val, vrps_test, solver_class, solve_prob=0.5, patience=2):
        self.cost_model = None
        self.optimizer = None
        all_vrps = vrps_train + vrps_val + vrps_test
        self.criterion = NCELoss({vrp: [vrp.actual_solution] for vrp in all_vrps})
        self.vrps_train = vrps_train
        self.vrps_val = vrps_val
        self.vrps_test = vrps_test
        self.solve_prob = solve_prob
        self.solver_class = solver_class
        self.patience = patience

    def train(self, epochs=20, verbose=False, test_every=5):
        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(epochs):
            self.cost_model.train()
            mean_loss = 0
            if verbose:
                loop = enumerate(self.vrps_train)
            else:
                loop = tqdm(enumerate(self.vrps_train))
            for idx, vrp in loop:
                self.optimizer.zero_grad()

                predicted_edge_costs = self.cost_model(get_edge_features(vrp.edges))
                set_predicted_costs(vrp.edges, predicted_edge_costs)

                # add to pool with probability
                if np.random.rand() < self.solve_prob:
                    solver = self.solver_class(vrp, mode=SolverMode.PRED_COST)
                    solver.solve()
                    self.criterion.pool[vrp].append(solver.get_decision_variables())

                loss = self.criterion(predicted_edge_costs, vrp.actual_solution, vrp)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, instance {idx + 1}/{len(self.vrps_train)}, loss: {loss.item()}')
            val_loss = self.validation_loss()
            print(f'Epoch {epoch + 1} / {epochs} done, mean loss: {mean_loss / len(self.vrps_train)},'
                  f' validation loss: {val_loss}')

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience and epoch > 3:
                    print(f"Early stopping at epoch {epoch}")
                    break

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
                loss += self.criterion(predicted_edge_costs, vrp.actual_solution, vrp).item()
            return loss / len(self.vrps_val)


class NCEModelLinear(NCEModel):

    def __init__(self, vrps_train, vrps_val, vrps_test, solver_class, lr=1e-4, solve_prob=0.5, weight_decay=0.0):
        super().__init__(vrps_train, vrps_val, vrps_test, solver_class, solve_prob)
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = CostPredictorLinear(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=weight_decay)


class NCEModelEncoderDecoder(NCEModel):

    def __init__(self, vrps_train, vrps_val, vrps_test, solver_class, lr=1e-4, solve_prob=0.5, weight_decay=0.0, hidden_size=256):
        super().__init__(vrps_train, vrps_val, vrps_test, solver_class, solve_prob)
        num_edges = len(vrps_train[0].edges)
        num_features = len(vrps_train[0].edges[0].features)
        self.cost_model = EncoderDecoder(num_edges * num_features, hidden_size, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=weight_decay)

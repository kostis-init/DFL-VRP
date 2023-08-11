import torch
import torch.nn as nn
from tqdm import tqdm

from dfl_vrp.util import get_edge_features, set_predicted_costs


class EdgeCostPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1)
        x = self.linear(x)
        x = self.activation(x)
        return x


class TwoStageModelNew:
    def __init__(self, train_set, val_set, test_set, lr=1e-3, patience=3, weight_decay=0.0):

        num_edges = len(train_set[0].edges)
        num_features = len(train_set[0].edges[0].features)
        self.cost_model = EdgeCostPredictor(num_edges * num_features, num_edges)
        self.optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = patience
        self.vrps_train = train_set
        self.vrps_val = val_set
        self.vrps_test = test_set

    def train(self, num_epochs=50):
        self.cost_model.train()
        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(num_epochs):
            total_loss = 0
            for idx, vrp in tqdm(enumerate(self.vrps_train)):
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                edge_features = get_edge_features(vrp.edges)
                predicted_edge_costs = self.cost_model(edge_features)
                # Compute loss
                actual_edge_costs = torch.tensor([edge.cost for edge in vrp.edges], dtype=torch.float32)
                loss = self.criterion(predicted_edge_costs, actual_edge_costs)
                # Backward pass
                loss.backward()
                # Update model parameters
                self.optimizer.step()
                # Add to the training loss
                total_loss += loss.item()
            # Compute average training loss for the epoch
            total_loss /= len(self.vrps_train)

            # Calculate val set loss for the epoch
            self.cost_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vrp in self.vrps_val:
                    edge_features = get_edge_features(vrp.edges)
                    predicted_edge_costs = self.cost_model(edge_features)
                    actual_edge_costs = torch.tensor([edge.cost for edge in vrp.edges], dtype=torch.float32)
                    val_loss += self.criterion(predicted_edge_costs, actual_edge_costs).item()
            val_loss /= len(self.vrps_val)

            print(f"Epoch {epoch}: Train Loss: {total_loss} | Validation Loss: {val_loss}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        self.test()

    def test(self):
        self.cost_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for vrp in self.vrps_test:
                edge_features = get_edge_features(vrp.edges)
                predicted_edge_costs = self.cost_model(edge_features)
                actual_edge_costs = torch.tensor([edge.cost for edge in vrp.edges], dtype=torch.float32)
                test_loss += self.criterion(predicted_edge_costs, actual_edge_costs).item()
        test_loss /= len(self.vrps_test)
        print(f"Test Loss: {test_loss}")

    def predict(self, edges):
        self.cost_model.eval()
        with torch.no_grad():
            edge_features = get_edge_features(edges)
            predicted_edge_costs = self.cost_model(edge_features)
            set_predicted_costs(edges, predicted_edge_costs)

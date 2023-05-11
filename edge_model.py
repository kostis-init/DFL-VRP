import bisect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import numpy as np


class VRPDataset(Dataset):
    def __init__(self, vrp_instances):
        self.vrp_instances = vrp_instances
        self.cumulative_sum = [0] + list(np.cumsum([len(instance.edges) for instance in self.vrp_instances]))

    def __len__(self):
        return self.cumulative_sum[-1]

    def __getitem__(self, idx):
        instance_idx = bisect.bisect_right(self.cumulative_sum, idx) - 1
        instance_offset = idx - self.cumulative_sum[instance_idx]
        edge = self.vrp_instances[instance_idx].edges[instance_offset]

        features = torch.tensor(edge.features, dtype=torch.float32)
        target = torch.tensor([edge.cost], dtype=torch.float32)
        return features, target


class EdgeCostPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, output_size)
        # )

    def forward(self, x):
        return self.fc(x)


class EdgeTrainer:
    def __init__(self, train_set, test_set, lr=0.001, patience=5):
        self.train_dataloader = DataLoader(VRPDataset(train_set), batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(VRPDataset(test_set), batch_size=32, shuffle=True)
        self.model = EdgeCostPredictor(len(train_set[0].edges[0].features), 32, 1)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.patience = patience

    def train(self, num_epochs=50):
        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(num_epochs):
            # Set model to training mode
            self.model.train()
            train_loss = 0.0
            for features, targets in self.train_dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(features)
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                # Backward pass
                loss.backward()
                # Update model parameters
                self.optimizer.step()
                # Add to the training loss
                train_loss += loss.item()
            # Compute average training loss for the epoch
            train_loss /= len(self.train_dataloader)

            # Calculate test set loss for the epoch
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_idx, (features, targets) in enumerate(self.test_dataloader):
                    test_loss += self.loss_fn(self.model(features), targets).item()
            test_loss /= len(self.test_dataloader)

            print(f"Epoch {epoch}: Train Loss: {train_loss} | Test Loss: {test_loss}")

            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def predict(self, features):
        """
        Predict the cost of an edge given its features
        :param features: list of features
        :return: predicted cost
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(features, dtype=torch.float32)).item()


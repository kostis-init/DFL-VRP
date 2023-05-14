import torch
import torch.nn as nn
import torch.optim as optim

from heuristic.heuristic_solver import HeuristicSolver
from util import parse_datafile


# Define the Gumbel-Softmax relaxation function
def gumbel_softmax(logits, temperature):
    """
    Converts logits to a relaxed one-hot vector using the Gumbel-Softmax trick. The output is a vector of probabilities
    that sum to 1.
    :param logits:
    :param temperature:
    :return:
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    softmax_output = nn.functional.softmax((logits + gumbel_noise) / temperature, dim=1)
    return softmax_output


# Define your model architecture
class VRPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VRPModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, temperature):
        logits = self.fc(x)
        relaxed_decision_vars = gumbel_softmax(logits, temperature)
        return relaxed_decision_vars


# Define your training loop
def train_model(model, optimizer, train_data, num_epochs, temperature):
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for vrp in train_data:
            optimizer.zero_grad()
            edge_features = torch.tensor([edge.features for edge in vrp.edges])

            predicted_costs = model(edge_features, temperature)

            loss = criterion(predicted_costs, vrp.actual_solution)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}")


# Define your training data (input_data, true_objective)
data = [parse_datafile(f'data/cvrp_10000_25_4_8_0.1/instance_{i}') for i in range(10)]
from tqdm import tqdm

# solve the VRPs with the actual edge costs and save the solutions
for vrp in tqdm(data):
    solver = HeuristicSolver(vrp, max_runtime=1)
    # solver = GurobiSolver(vrp)
    solver.solve()
    vrp.actual_solution = solver.get_decision_variables()
    vrp.actual_obj = solver.get_actual_objective()

num_feat = len(data[0].edges[0].features)

# Define model parameters
num_epochs = 10
temperature = 1.0

# Create the model and optimizer
model = VRPModel(num_feat, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, optimizer, data, num_epochs, temperature)

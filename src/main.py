import numpy as np
from util import *
import random
import tensorflow as tf
from solver import GurobiSolver

# todo: consider time
# todo: consider more than one vehicle
# todo: consider vehicle capacity/speed/cost

vrp = parse_datafile('../data/random/200/small.txt')

# construct features
features = []
for i, j in vrp.get_arcs():
    features.append([
        np.hypot(i.x - j.x, i.y - j.y),  # distance between nodes
        np.hypot(i.x - vrp.depot.x, i.y - vrp.depot.y),  # distance from node i to depot
        np.hypot(j.x - vrp.depot.x, j.y - vrp.depot.y),  # distance from node j to depot
        i.demand,  # demand of node i
        j.demand,  # demand of node j
        int(i != vrp.depot and j != vrp.depot),  # whether both nodes are customers
        # i.service_time,  # service time of node i
        # j.service_time,  # service time of node j
        # i.ready_time,  # ready time of node i
        # j.ready_time,  # ready time of node j
        # i.due_date,  # due date of node i
        # j.due_date,  # due date of node j
    ])


# Define a function to generate random costs based on the features
def generate_costs(features):
    # Randomly generate a scaling factor
    scale = random.uniform(0.5, 1.5)
    # Compute the costs based on the features, with some added randomness
    costs = []
    for f in features:
        # The cost is the sum of the features, scaled by the scaling factor, plus some random noise
        costs.append(sum(f) * scale + random.uniform(-0.5, 0.5))
    return costs


# # Create cost matrix (Euclidean distance for now)
# c = {(i, j): np.hypot(i.x - j.x, i.y - j.y) for i in vrp.get_all_nodes() for j in vrp.get_all_nodes()}
# Generate costs for each edge
actual_costs = {(i, j): generate_costs(features)[k] for k, (i, j) in enumerate(vrp.get_arcs())}


solver = GurobiSolver(vrp, actual_costs, mip_gap=0.2, time_limit=10)
solver.optimize()
draw_solution(solver.get_active_arcs(), vrp)

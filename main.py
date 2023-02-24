import numpy as np
from util import *
import random
# import tensorflow as tf
from solver import GurobiSolverMultiVehicle

# todo: consider vehicle capacity/speed/cost

vrp = parse_datafile('data/random/200/no_time.txt')
# vrp = parse_datafile('data/random/200/r101.txt')
# vrp = parse_datafile('data/random/200/r201.txt')
# vrp = parse_datafile('data/random/200/r301.txt')


# construct features
features = []
for i, j in vrp.arcs:
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


actual_costs = {(i, j): np.hypot(i.x - j.x, i.y - j.y) for i in vrp.nodes for j in vrp.nodes}
# actual_costs = {(i, j): generate_costs(features)[k] for k, (i, j) in enumerate(vrp.get_arcs())}
speed = 10.0
travel_times = {(i, j): vrp.service_times[i] + np.hypot(i.x - j.x, i.y - j.y)/speed for i in vrp.nodes for j in vrp.nodes}

solver = GurobiSolverMultiVehicle(vrp, actual_costs, travel_times, mip_gap=0.02, time_limit=20, verbose=True)

solver.optimize()
draw_solution(solver.get_active_arcs(), vrp)

import numpy as np
from util import *
import random
# import tensorflow as tf
from solver import GurobiSolver


# vrp = parse_datafile('data/no_time.txt')
# vrp = parse_datafile('data/r101.txt')
vrp = parse_datafile('data/c101.txt')
# vrp = parse_datafile('data/r201.txt')
# vrp = parse_datafile('data/r301.txt')

# TODO: try both per edge and per VRP instance training

# construct features
features = []
for i, j in vrp.arcs:
    features.append([
        np.hypot(i.x - j.x, i.y - j.y),  # distance between nodes
        np.hypot(i.x - vrp.depot.x, i.y - vrp.depot.y),  # distance from node i to depot
        np.hypot(j.x - vrp.depot.x, j.y - vrp.depot.y),  # distance from node j to depot
        i.demand + j.demand,  # total demand of both nodes
        int(i != vrp.depot and j != vrp.depot),  # whether both nodes are customers
        i.service_time + j.service_time,  # total service time of both nodes
        i.ready_time + j.ready_time,  # total ready time of both nodes
        i.due_date + j.due_date,  # total due date of both nodes
    ])

solver = GurobiSolver(vrp, mip_gap=0, time_limit=20, verbose=True)

solver.optimize()
draw_solution(solver)

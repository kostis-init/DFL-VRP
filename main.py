import numpy as np
from util import *
import random
# import tensorflow as tf
from solver import GurobiSolver


# vrp = parse_datafile('data/no_time.txt')
# vrp = parse_datafile('data/r101.txt')
# vrp = parse_datafile('data/c101.txt')
# vrp = parse_datafile('data/r201.txt')
# vrp = parse_datafile('data/r301.txt')
vrp = parse_datafile('./data/generated/instance_0')

# TODO: try both per edge and per VRP instance training

solver = GurobiSolver(vrp, mip_gap=0, time_limit=20, verbose=True)

solver.optimize()
draw_solution(solver)

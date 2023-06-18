import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from enums import SolverMode
from heuristic.heuristic_solver import HeuristicSolver
from solver import GurobiSolver
from util import euclidean_distance, parse_datafile
from sklearn.preprocessing import MinMaxScaler

NUM_INSTANCES = 10000
NUM_NODES = 1000
NUM_EDGES = 10
NUM_FEATURES = 4
DEGREE = 6
NOISE_WIDTH = 0.1
MIN_COORD, MAX_COORD = -2, 2
MIN_DEMAND, MAX_DEMAND = 0, 10
MIN_CAPACITY, MAX_CAPACITY = MAX_DEMAND, MAX_DEMAND * NUM_NODES / 2
FEAT_COEFF = np.random.binomial(1, 0.5, (NUM_NODES * NUM_NODES, NUM_FEATURES)) \
             * np.random.uniform(MIN_COORD, MAX_COORD, (NUM_NODES * NUM_NODES, NUM_FEATURES))
OUTPUT_PATH = f'./data/scaled/cvrp_{NUM_INSTANCES}_{NUM_NODES}_{NUM_EDGES}/'
scaler = MinMaxScaler()

def generate_nodes(file):
    nodes = pd.DataFrame({
        "id": np.arange(NUM_NODES),
        "xcord": np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES),
        "ycord": np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES),
        "demand": np.round(np.random.uniform(MIN_DEMAND, MAX_DEMAND, NUM_NODES), 2)
    })
    nodes.loc[0, "demand"] = 0
    nodes.to_csv(file, index=False)
    return nodes


def generate_edges(nodes, file):
    edges = []
    for j in range(NUM_NODES):
        node_edges = []
        if j == 0:
            loop = range(1, NUM_NODES)
        else:
            # random sample of NUM_EDGES nodes but remove the depot (node 0, so search from 1 to NUM_NODES)
            # also remove the current node
            possible_nodes = np.arange(1, NUM_NODES)
            possible_nodes = np.delete(possible_nodes, j - 1)
            loop = np.random.choice(possible_nodes, NUM_EDGES, replace=False)
        for k in loop:
            n1, n2 = nodes.iloc[j], nodes.iloc[k]
            distance = euclidean_distance(n1["xcord"], n1["ycord"], n2["xcord"], n2["ycord"])
            # the features are sampled from a multivariate Gaussian distribution with mean 0 and standard deviation 1
            features = np.random.normal(0, 1, NUM_FEATURES)
            # scale the features to be between 0 and 1
            features = scaler.fit_transform(features.reshape(-1, 1)).reshape(-1)
            cost = (np.dot(FEAT_COEFF[j * NUM_NODES + k], features)) / np.sqrt(NUM_FEATURES) + 3
            cost **= DEGREE
            # multiplicative noise that is sampled from a uniform distribution
            noise = np.random.uniform(1 - NOISE_WIDTH, 1 + NOISE_WIDTH)
            cost *= noise

            edge = [j, k, distance]
            edge.extend(features)
            edge.append(cost)
            node_edges.append(edge)

        if j == 0:
            edges.extend(node_edges)
            continue
        # add an edge to the depot
        n1, n2 = nodes.iloc[j], nodes.iloc[0]
        distance = euclidean_distance(n1["xcord"], n1["ycord"], n2["xcord"], n2["ycord"])
        features = np.random.normal(0, 1, NUM_FEATURES)
        cost = (np.dot(FEAT_COEFF[j * NUM_NODES], features)) / np.sqrt(NUM_FEATURES) + 3
        cost **= DEGREE
        noise = np.random.uniform(1 - NOISE_WIDTH, 1 + NOISE_WIDTH)
        cost *= noise
        edge = [j, 0, distance]
        edge.extend(features)
        edge.append(cost)
        node_edges.append(edge)
        edges.extend(node_edges)

    # scale the costs to be between 0 and 1
    costs = np.array([edge[-1] for edge in edges]).reshape(-1, 1)
    costs = scaler.fit_transform(costs).reshape(-1)
    for i, edge in enumerate(edges):
        edge[-1] = costs[i]

    pd.DataFrame(edges,
                 columns=["node1", "node2", "distance"] + [f"f{i}" for i in range(NUM_FEATURES)] + ["cost"]).to_csv(
        file, index=False)


def generate_metadata(file):
    pd.DataFrame({
        "capacity": [np.random.randint(MIN_CAPACITY, MAX_CAPACITY)],
        "num_nodes": [NUM_NODES],
        "num_features": [NUM_FEATURES],
        "degree": [DEGREE],
        "noise_width": [NOISE_WIDTH],
        "min_coord": [MIN_COORD],
        "max_coord": [MAX_COORD],
        "min_demand": [MIN_DEMAND],
        "max_demand": [MAX_DEMAND],
        "min_capacity": [MIN_CAPACITY],
        "max_capacity": [MAX_CAPACITY]
    }).to_csv(file, index=False)


def generate_solution(file, solver):
    solver.solve()
    with open(file, 'w') as f:
        f.write(f'{solver.get_routes()}\n')
        f.write(f'{solver.get_actual_objective()}\n')


def main():
    for i in tqdm(range(NUM_INSTANCES)):
        instance_dir = os.path.join(OUTPUT_PATH, f"instance_{i}")
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        nodes_file = os.path.join(instance_dir, "nodes.csv")
        edges_file = os.path.join(instance_dir, "edges.csv")
        metadata_file = os.path.join(instance_dir, "metadata.csv")
        solution_file = os.path.join(instance_dir, "solution.txt")
        heuristic_solution_file = os.path.join(instance_dir, "heuristic_solution.txt")

        nodes = generate_nodes(nodes_file)
        generate_edges(nodes, edges_file)
        generate_metadata(metadata_file)

        vrp = parse_datafile(instance_dir)
        generate_solution(solution_file, GurobiSolver(vrp, mip_gap=0.2, time_limit=30))
        # generate_solution(solution_file, HeuristicSolver(vrp, time_limit=5))


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    main()

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from heuristic.heuristic_solver import HeuristicSolver
from solver import GurobiSolver
from util import euclidean_distance, parse_datafile

NUM_INSTANCES, NUM_NODES, NUM_FEATURES, DEGREE, NOISE_WIDTH = 10000, 100, 4, 4, 0.1

MIN_COORD, MAX_COORD = -2, 2
MIN_DEMAND, MAX_DEMAND = 0, 10
MIN_CAPACITY, MAX_CAPACITY = MAX_DEMAND, MAX_DEMAND * NUM_NODES / 2
FEAT_COEFF = np.random.binomial(1, 0.5, (NUM_NODES * NUM_NODES, NUM_FEATURES)) \
             * np.random.uniform(MIN_COORD, MAX_COORD, (NUM_NODES * NUM_NODES, NUM_FEATURES))
OUTPUT_PATH = f'./data/cvrp_{NUM_INSTANCES}_{NUM_NODES}_{NUM_FEATURES}_{DEGREE}_{NOISE_WIDTH}/'


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
        return_edge = None
        for k in range(NUM_NODES):

            if j == k:
                continue

            n1, n2 = nodes.iloc[j], nodes.iloc[k]

            distance = euclidean_distance(n1["xcord"], n1["ycord"], n2["xcord"], n2["ycord"])

            # the features are sampled from a multivariate Gaussian distribution with mean 0 and standard deviation 1
            features = np.random.normal(0, 1, NUM_FEATURES)

            cost = (np.dot(FEAT_COEFF[j * NUM_NODES + k], features)) / np.sqrt(NUM_FEATURES) + 3
            cost **= DEGREE
            # multiplicative noise that is sampled from a uniform distribution
            noise = np.random.uniform(1 - NOISE_WIDTH, 1 + NOISE_WIDTH)
            cost *= noise

            edge = [j, k, distance]
            edge.extend(features)
            edge.append(cost)
            if k == 0:
                return_edge = edge
            else:
                node_edges.append(edge)

        if j != 0:
            # keep only the closest 20% of the edges
            node_edges = sorted(node_edges, key=lambda x: x[-1])[:NUM_NODES//5]
            node_edges.append(return_edge)
        edges.extend(node_edges)

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
        # generate_solution(solution_file, GurobiSolver(vrp, time_limit=1))
        generate_solution(solution_file, HeuristicSolver(vrp, time_limit=1))


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    main()

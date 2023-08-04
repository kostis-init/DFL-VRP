import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from dfl_vrp.heuristic.heuristic_solver import HeuristicSolver
from solver import GurobiSolver
from util import parse_datafile
from sklearn.preprocessing import MinMaxScaler

NUM_INSTANCES = 999
NUM_NODES = 10
NUM_EDGES = NUM_NODES - 2  # Complete graph
NUM_FEATURES = 4
DEGREE = 6  # Locked
NOISE_WIDTH = 0.1
MIN_DEMAND, MAX_DEMAND = 1, 50
# MIN_TIME, MAX_TIME = 0, 100
CAPACITY = 100
FEAT_COEFF = np.random.binomial(1, 0.5, (NUM_NODES ** 2, NUM_FEATURES)) * np.random.uniform(-2, 2, (
    NUM_NODES ** 2, NUM_FEATURES))

NEIGHBOUR_COST_FACTOR = 0.3  # The proportion of each neighbour's cost to add to the cost of an edge

OUTPUT_PATH = f'../data/capacity{CAPACITY}/instances{NUM_INSTANCES}/nodes{NUM_NODES}/noise{NOISE_WIDTH}/feat{NUM_FEATURES}/'


def generate_nodes(file):
    nodes = pd.DataFrame({
        "id": np.arange(NUM_NODES),
        "demand": np.round(np.random.uniform(MIN_DEMAND, MAX_DEMAND, NUM_NODES), 2)
    })
    nodes.loc[0, "demand"] = 0
    nodes.to_csv(file, index=False)


def generate_edges(file):
    scaler = MinMaxScaler()
    edges = []
    for j in range(NUM_NODES):
        node_edges = []
        if j == 0:
            loop = range(1, NUM_NODES)
        else:
            possible_nodes = np.arange(1, NUM_NODES)
            possible_nodes = np.delete(possible_nodes, j - 1)
            loop = np.random.choice(possible_nodes, NUM_EDGES, replace=False)
        for k in loop:
            # the features are sampled from a multivariate Gaussian distribution with mean 0 and standard deviation 1
            features = np.random.normal(0, 1, NUM_FEATURES)
            # scale the features to be between 0 and 1
            features = scaler.fit_transform(features.reshape(-1, 1)).reshape(-1)
            # round the features to 5 decimal places
            features = np.round(features, 5)

            cost = (np.dot(FEAT_COEFF[j * NUM_NODES + k], features)) / np.sqrt(NUM_FEATURES) + 3

            # make degree be a random number between 4, 6 and 8
            DEGREE = np.random.choice([4, 6, 8])
            cost **= DEGREE
            # multiplicative noise that is sampled from a uniform distribution
            noise = np.random.uniform(1 - NOISE_WIDTH, 1 + NOISE_WIDTH)
            cost *= noise
            edge = [j, k]
            edge.extend(features)
            edge.append(cost)
            node_edges.append(edge)

        if j == 0:
            edges.extend(node_edges)
            continue
        # add an edge to the depot
        features = np.random.normal(0, 1, NUM_FEATURES)
        features = scaler.fit_transform(features.reshape(-1, 1)).reshape(-1)
        features = np.round(features, 5)
        cost = (np.dot(FEAT_COEFF[j * NUM_NODES], features)) / np.sqrt(NUM_FEATURES) + 3
        cost **= DEGREE
        noise = np.random.uniform(1 - NOISE_WIDTH, 1 + NOISE_WIDTH)
        cost *= noise
        edge = [j, 0]
        edge.extend(features)
        edge.append(cost)
        node_edges.append(edge)
        edges.extend(node_edges)

    # after generating all edges, create a new list of edges with updated costs
    # updated_edges = []
    # for edge in edges:
    #     neighbour_costs = [e[-1] for e in edges if (e[0] == edge[0] or e[1] == edge[0]) and e != edge]
    #     # keep the largest 4 of neighbour costs
    #     neighbour_costs = sorted(neighbour_costs, reverse=True)[:4]
    #     updated_cost = edge[-1] + NEIGHBOUR_COST_FACTOR * sum(neighbour_costs)
    #     updated_edge = edge[:-1] + [updated_cost]
    #     updated_edges.append(updated_edge)
    # edges = updated_edges

    # scale the costs to be between 0 and 1
    costs = np.array([edge[-1] for edge in edges]).reshape(-1, 1)
    costs = scaler.fit_transform(costs).reshape(-1)
    # round the costs to 5 decimal places
    costs = np.round(costs, 5)
    for i, edge in enumerate(edges):
        edge[-1] = costs[i]

    pd.DataFrame(edges, columns=["node1", "node2"] + [f"f{i}" for i in range(NUM_FEATURES)] + ["cost"]) \
        .to_csv(file, index=False)


def generate_metadata(file):
    pd.DataFrame({
        "capacity": [CAPACITY],
        "degree": [DEGREE],
        "min_demand": [MIN_DEMAND],
        "max_demand": [MAX_DEMAND]
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

        generate_nodes(nodes_file)
        generate_edges(edges_file)
        generate_metadata(metadata_file)

        vrp = parse_datafile(instance_dir)
        h1_solution_file = os.path.join(instance_dir, "h1_solution.txt")
        h2_solution_file = os.path.join(instance_dir, "h2_solution.txt")
        h3_solution_file = os.path.join(instance_dir, "h3_solution.txt")
        h4_solution_file = os.path.join(instance_dir, "h4_solution.txt")
        h5_solution_file = os.path.join(instance_dir, "h5_solution.txt")
        h6_solution_file = os.path.join(instance_dir, "h6_solution.txt")
        generate_solution(solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=2))

        if False:
            g1_solution_file = os.path.join(instance_dir, "g1_solution.txt")
            g2_solution_file = os.path.join(instance_dir, "g2_solution.txt")
            g3_solution_file = os.path.join(instance_dir, "g3_solution.txt")
            g4_solution_file = os.path.join(instance_dir, "g4_solution.txt")
            g5_solution_file = os.path.join(instance_dir, "g5_solution.txt")
            g6_solution_file = os.path.join(instance_dir, "g6_solution.txt")
            g7_solution_file = os.path.join(instance_dir, "g7_solution.txt")
            g8_solution_file = os.path.join(instance_dir, "g8_solution.txt")
            g9_solution_file = os.path.join(instance_dir, "g9_solution.txt")
            h1_solution_file = os.path.join(instance_dir, "h1_solution.txt")
            h2_solution_file = os.path.join(instance_dir, "h2_solution.txt")
            h3_solution_file = os.path.join(instance_dir, "h3_solution.txt")
            h4_solution_file = os.path.join(instance_dir, "h4_solution.txt")
            h5_solution_file = os.path.join(instance_dir, "h5_solution.txt")
            h6_solution_file = os.path.join(instance_dir, "h6_solution.txt")
            h7_solution_file = os.path.join(instance_dir, "h7_solution.txt")
            h8_solution_file = os.path.join(instance_dir, "h8_solution.txt")
            h9_solution_file = os.path.join(instance_dir, "h9_solution.txt")
            generate_solution(g1_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=10))
            generate_solution(h1_solution_file, HeuristicSolver(vrp, time_limit=10))
            generate_solution(g2_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=20))
            generate_solution(h2_solution_file, HeuristicSolver(vrp, time_limit=20))
            generate_solution(g3_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=30))
            generate_solution(h3_solution_file, HeuristicSolver(vrp, time_limit=30))
            generate_solution(g4_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=4))
            generate_solution(h4_solution_file, HeuristicSolver(vrp, time_limit=4))
            generate_solution(g5_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=5))
            generate_solution(h5_solution_file, HeuristicSolver(vrp, time_limit=5))
            generate_solution(g6_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=6))
            generate_solution(h6_solution_file, HeuristicSolver(vrp, time_limit=6))
            generate_solution(g7_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=7))
            generate_solution(h7_solution_file, HeuristicSolver(vrp, time_limit=7))
            generate_solution(g8_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=8))
            generate_solution(h8_solution_file, HeuristicSolver(vrp, time_limit=8))
            generate_solution(g9_solution_file, GurobiSolver(vrp, mip_gap=0, time_limit=9))
            generate_solution(h9_solution_file, HeuristicSolver(vrp, time_limit=9))


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    main()

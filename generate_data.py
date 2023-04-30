import pandas as pd
import numpy as np
import os
from util import euclidean_distance

NUM_INSTANCES = 1000
NUM_NODES = 10
NUM_FEATURES = 5
# degree must be even
DEGREE = 4
NOISE_WIDTH = 0.05

MIN_COORD, MAX_COORD = -5, 5
MIN_DEMAND, MAX_DEMAND = 0, 10
MIN_READY_TIME, MAX_READY_TIME = 0, 10
MIN_DUE_TIME, MAX_DUE_TIME = 10000, 10000
MIN_SERVICE_TIME, MAX_SERVICE_TIME = 1, 5

OUTPUT_PATH = f'./data/v3_{NUM_INSTANCES}_{NUM_NODES}_{NUM_FEATURES}_{DEGREE}/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

for i in range(NUM_INSTANCES):
    instance_dir = os.path.join(OUTPUT_PATH, f"instance_{i}")
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)
    nodes_file = os.path.join(instance_dir, "nodes.csv")
    edges_file = os.path.join(instance_dir, "edges.csv")

    # generate nodes
    nodes = pd.DataFrame({
        "id": np.arange(NUM_NODES),
        "xcord": np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES),
        "ycord": np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES),
        "demand": np.round(np.random.uniform(MIN_DEMAND, MAX_DEMAND, NUM_NODES), 2),
        "ready_time": np.round(np.random.uniform(MIN_READY_TIME, MAX_READY_TIME, NUM_NODES), 2),
        "due_time": np.round(np.random.uniform(MIN_DUE_TIME, MAX_DUE_TIME, NUM_NODES), 2),
        "service_time": np.round(np.random.uniform(MIN_SERVICE_TIME, MAX_SERVICE_TIME, NUM_NODES), 2),
    })
    nodes.loc[0, "demand"] = 0
    nodes.loc[0, "ready_time"] = 0
    nodes.loc[0, "due_time"] = MAX_DUE_TIME
    nodes.loc[0, "service_time"] = 0
    nodes.to_csv(nodes_file, index=False)

    B = np.random.binomial(1, 0.5, (NUM_NODES * NUM_NODES, NUM_FEATURES))
    edges = []
    for j in range(NUM_NODES):
        for k in range(NUM_NODES):
            if j == k:
                continue
            n1, n2 = nodes.iloc[j], nodes.iloc[k]
            distance = euclidean_distance(n1["xcord"], n1["ycord"], n2["xcord"], n2["ycord"])
            features = np.random.normal(0, 1, NUM_FEATURES)
            noise = np.random.normal(1, NOISE_WIDTH)
            cost = (distance + np.dot(B[j * NUM_NODES + k], features)) / np.sqrt(NUM_FEATURES) + 3
            cost = cost ** DEGREE + 1
            cost *= noise

            edge = [j, k, distance]
            edge.extend(features)
            edge.append(cost)
            edges.append(edge)

    pd.DataFrame(edges, columns=["node1", "node2", "distance"] + [f"f{i}" for i in range(NUM_FEATURES)] + ["cost"])\
        .to_csv(edges_file, index=False)

    print(f"Instance {i} generated")

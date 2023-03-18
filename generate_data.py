import pandas as pd
import numpy as np
import os
from util import euclidean_distance

NUM_INSTANCES = 1000
NUM_NODES = 25

MIN_COORD = 0
MAX_COORD = 100

MIN_DEMAND = 1
MAX_DEMAND = 10

MIN_READY_TIME = 0
MAX_READY_TIME = 30

MIN_DUE_TIME = 70
MAX_DUE_TIME = 100

MIN_SERVICE_TIME = 1
MAX_SERVICE_TIME = 10

MIN_RAIN = 0
MAX_RAIN = 1
MIN_TRAFFIC = 0
MAX_TRAFFIC = 1

OUTPUT_PATH = "./data/generated/"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

for i in range(NUM_INSTANCES):

    instance_dir = os.path.join(OUTPUT_PATH, f"instance_{i}")
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)
    nodes_file = os.path.join(instance_dir, "nodes.csv")
    edges_file = os.path.join(instance_dir, "edges.csv")

    nodes = pd.DataFrame({
        "id": np.arange(NUM_NODES),
        "xcord": np.round(np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES), 2),
        "ycord": np.round(np.random.uniform(MIN_COORD, MAX_COORD, NUM_NODES), 2),
        "demand": np.round(np.random.uniform(MIN_DEMAND, MAX_DEMAND, NUM_NODES), 2),
        "ready_time": np.round(np.random.uniform(MIN_READY_TIME, MAX_READY_TIME, NUM_NODES), 2),
        "due_time": np.round(np.random.uniform(MIN_DUE_TIME, MAX_DUE_TIME, NUM_NODES), 2),
        "service_time": np.round(np.random.uniform(MIN_SERVICE_TIME, MAX_SERVICE_TIME, NUM_NODES), 2),
    })

    nodes.loc[0, "xcord"] = MAX_COORD / 2
    nodes.loc[0, "ycord"] = MAX_COORD / 2
    nodes.loc[0, "demand"] = 0
    nodes.loc[0, "ready_time"] = MIN_READY_TIME
    nodes.loc[0, "due_time"] = MAX_DUE_TIME
    nodes.loc[0, "service_time"] = 0

    edges = []
    for j in range(NUM_NODES):
        for k in range(NUM_NODES):
            if j == k:
                continue
            n1, n2 = nodes.iloc[j], nodes.iloc[k]
            dist = round(euclidean_distance(n1["xcord"], n1["ycord"], n2["xcord"], n2["ycord"]), 2)
            dist_depot = round(
                euclidean_distance(n2["xcord"], n2["ycord"], nodes.iloc[0]["xcord"], nodes.iloc[0]["ycord"]), 2)
            demand = round(n1["demand"] + n2["demand"], 2)
            is_customer = int(j != 0 and k != 0)
            svc_time = round(n1["service_time"] + n2["service_time"], 2)
            rd_time = round(max(n1["ready_time"], n2["ready_time"]), 2)
            due_time = round(min(n1["due_time"], n2["due_time"]), 2)
            rain = round(np.random.uniform(MIN_RAIN, MAX_RAIN), 2)
            traffic = round(np.random.uniform(MIN_TRAFFIC, MAX_TRAFFIC), 2)
            cost = round(dist + dist_depot + 10 * demand + 100 * is_customer + 10 * svc_time
                         + rd_time + due_time + 50 * rain + 50 * traffic, 2)
            edges.append(
                [j, k, dist, dist_depot, demand, is_customer, svc_time, rd_time, due_time, rain, traffic, cost])

    edges = pd.DataFrame(edges,
                         columns=["node1", "node2", "distance", "distance_to_depot", "total_demand", "is_customer",
                                  "total_service_time", "total_ready_time", "total_due_time", "rain", "traffic",
                                  "cost"])

    nodes.to_csv(nodes_file, index=False)
    edges.to_csv(edges_file, index=False)

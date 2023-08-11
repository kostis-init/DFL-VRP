import time

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from dfl_vrp.domain.vrp_node import VRPNode
from dfl_vrp.domain.vrp_edge import VRPEdge
from dfl_vrp.domain.vrp import VRP
from dataclasses import fields

from dfl_vrp.enums import SolverMode
from dfl_vrp.heuristic.heuristic_solver import HeuristicSolver
from dfl_vrp.solver import GurobiSolver
import torch
import os


def parse_datafile(instance_dir: str) -> VRP:
    nodes_file_path = f'{instance_dir}/nodes.csv'
    edges_file_path = f'{instance_dir}/edges.csv'
    metadata_file_path = f'{instance_dir}/metadata.csv'
    solution_file_path = f'{instance_dir}/solution.txt'

    # Nodes
    nodes_columns = {field.name: field.type for field in fields(VRPNode)}
    nodes_df = pd.read_csv(nodes_file_path, skiprows=1, names=nodes_columns, sep=',')
    nodes = [VRPNode(**row) for row in nodes_df.to_dict('records')]

    # Edges
    edges_df = pd.read_csv(edges_file_path, skiprows=0, sep=',')
    edges = []
    for _, row in edges_df.iterrows():
        node1_id, node2_id = int(row[0]), int(row[1])
        node1 = next(node for node in nodes if node.id == node1_id)
        node2 = next(node for node in nodes if node.id == node2_id)
        edge = VRPEdge(node1=node1, node2=node2, features=row[2:-1].tolist(), cost=row[-1])
        edges.append(edge)

    # Metadata
    metadata_df = pd.read_csv(metadata_file_path)
    capacity = metadata_df['capacity'][0]

    # Solution
    routes, objective = None, None
    if os.path.exists(solution_file_path):
        # first line contains the routes, which is a list of lists of node ids
        # second line contains the objective value
        with open(solution_file_path, 'r') as f:
            routes = eval(f.readline())
            # convert node ids to node objects
            routes = [[next(node for node in nodes if node.id == node_id) for node_id in route] for route in routes]
            objective = float(f.readline())

    return VRP(instance_dir, nodes, edges, nodes[0], capacity, routes, None, objective)


# TODO: check
# def plot_solution(solution, name="CVRP solution"):
#     """
#     Plot the routes of the passed-in solution.
#     """
#     fig, ax = plt.subplots(figsize=(12, 10))
#     cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
#
#     for idx, route in enumerate(solution.routes):
#         ax.plot(
#             [data.coordinates[loc][0] for loc in [0] + route + [0]],
#             [data.coordinates[loc][1] for loc in [0] + route + [0]],
#             color=cmap[idx],
#             marker='.'
#         )
#
#     # Plot the depot
#     kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
#     ax.scatter(*data.coordinates[0], c="tab:red", **kwargs)
#
#     ax.set_title(f"{name}\n Total distance: {solution.cost}")
#     ax.set_xlabel("X-coordinate")
#     ax.set_ylabel("Y-coordinate")
#     ax.legend(frameon=False, ncol=3)

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import numpy as np

def draw_solution(vrp, routes, total_cost, name="CVRP solution"):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(routes)))

    graph = nx.DiGraph()
    graph.add_nodes_from(node.id for node in vrp.nodes)
    pos = nx.random_layout(graph, seed=7)

    for idx, route in enumerate(routes):
        ax.plot(
            [pos[loc.id][0] for loc in [vrp.depot] + [node for node in route] + [vrp.depot]],
            [pos[loc.id][1] for loc in [vrp.depot] + [node for node in route] + [vrp.depot]],
            color=cmap[idx],
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="D", s=300)
    ax.scatter(*pos[vrp.depot.id], c="tab:red", **kwargs)

    total_cost = round(total_cost, 2)
    ax.set_title(f"{name}\n Total cost: {total_cost}")
    ax.legend(frameon=False)
    plt.xticks([])  # removes x-axis tick values
    plt.yticks([])  # removes y-axis tick values
    plt.xlabel('')  # removes x-axis label
    plt.ylabel('')  # removes y-axis label
    plt.show()


def draw_solution1(vrp, routes, total_cost, name="CVRP solution") -> None:

    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)

    # Automatic node positioning using spring layout
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ['red' if i == vrp.depot else 'black' for i in vrp.nodes]

    fig, ax = plt.subplots()
    total_cost = round(total_cost, 2)
    ax.set_title(f"{name}\n Total cost: {total_cost}")
    # ax.legend(frameon=False)
    nx.draw_networkx_nodes(G=graph, ax=ax, node_color=node_colors, node_size=10, pos=pos)
    for i, route in enumerate(routes):
        route_edges = [(vrp.depot, route[0])] + [(route[i], route[i + 1]) for i in range(len(route) - 1)] + [
            (route[-1], vrp.depot)]
        nx.draw_networkx_edges(G=graph, ax=ax, edgelist=route_edges, edge_color=f'C{i}', width=1, arrowsize=10,
                               arrowstyle='-|>', pos=pos)
    # save figure
    plt.savefig(f'./{name}.png', dpi=600)
    plt.show()


TEST_SOLVER = GurobiSolver
TEST_SOLVING_LIMIT = 3


def test(model, instances, is_two_stage=True, verbose=False):
    with torch.no_grad():
        accuracy = 0.0
        actual_sols_cost = 0.0
        predicted_sols_cost = 0.0
        regret = 0.0
        if verbose:
            loop = instances
        else:
            loop = tqdm(instances)
        for inst in loop:
            actual_edges = [inst.edges[i] for i in range(len(inst.actual_solution)) if inst.actual_solution[i] > 0.5]
            actual_sols_cost += inst.actual_obj

            if is_two_stage:
                model.predict(inst.edges)
            else:
                model.eval()
                predicted_edge_costs = model(torch.tensor([edge.features for edge in inst.edges], dtype=torch.float32))
                set_predicted_costs(inst.edges, predicted_edge_costs)

            solver = TEST_SOLVER(inst, mode=SolverMode.PRED_COST, time_limit=TEST_SOLVING_LIMIT)
            # solver = GurobiSolver(inst, mode=SolverMode.PRED_COST, mip_gap=0, time_limit=5)
            solver.solve()
            predicted_edges = solver.get_active_arcs()
            predicted_obj = solver.get_actual_objective()
            predicted_sols_cost += predicted_obj

            regret += predicted_obj - inst.actual_obj
            correct_edges = set(actual_edges).intersection(predicted_edges)
            accuracy += float(len(correct_edges)) / len(predicted_edges)
            if verbose:
                print(
                    f'VRP {inst} -> Accuracy: {accuracy}, True cost: {inst.actual_obj}, predicted cost: {predicted_obj}')
        regret /= len(instances)
        accuracy /= len(instances)
        accuracy = round(accuracy * 100, 2)
        cost_comparison = predicted_sols_cost / actual_sols_cost
        print(f'Accuracy: {accuracy}%, cost comparison: {cost_comparison}, regret: {regret}')
        return accuracy, cost_comparison, regret


def test_single(model, vrp, is_two_stage=False):

    if is_two_stage:
        model.predict(vrp.edges)
    else:
        model.eval()
        predicted_edge_costs = model(torch.tensor([edge.features for edge in vrp.edges], dtype=torch.float32))
        set_predicted_costs(vrp.edges, predicted_edge_costs)

    solver = HeuristicSolver(vrp, mode=SolverMode.PRED_COST, time_limit=5)
    # solver = GurobiSolver(vrp, mode=SolverMode.PRED_COST, mip_gap=0, time_limit=10)
    solver.solve()
    print(f'Actual objective: {vrp.actual_obj}')
    draw_solution1(vrp, vrp.actual_routes, vrp.actual_obj, name="Actual solution")

    predicted_edges = solver.get_active_arcs()
    actual_edges = [vrp.edges[i] for i in range(len(vrp.actual_solution)) if vrp.actual_solution[i] > 0.5]
    correct_edges = set(actual_edges).intersection(predicted_edges)
    accuracy = float(len(correct_edges)) / len(predicted_edges)
    # round to 2 decimal places
    accuracy = round(accuracy * 100, 2)

    print(f'Predicted objective: {solver.get_actual_objective()}')
    draw_solution1(vrp, solver.get_routes(), solver.get_actual_objective(), name="Predicted solution, accuracy: " + str(accuracy) + "%")


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print(f'{method.__name__} took: {te - ts} sec')
        return te - ts, result

    return timed


def set_predicted_costs(edges, costs):
    for i, edge in enumerate(edges):
        edge.predicted_cost = costs[i].detach().item()


def get_edge_features(edges: [VRPEdge]):
    return torch.tensor([edge.features for edge in edges], dtype=torch.float32)

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

def draw_solution(solver) -> None:
    vrp = solver.vrp

    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)

    pos = {i: (i.x, i.y) for i in vrp.nodes}
    node_colors = ['red' if i == vrp.depot else 'black' for i in vrp.nodes]

    fig, ax = plt.subplots()
    attrs = {i: {
        'cumulative_demand': round(solver.get_loads()[i], 2),
        'demand': round(i.demand, 2),
    } for i in vrp.customers}
    nx.set_node_attributes(graph, attrs)

    nodes = nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, node_color=node_colors, node_size=10)

    routes = solver.get_routes()
    for i, route in enumerate(routes):
        route_edges = [(vrp.depot, route[0])] + [(route[i], route[i + 1]) for i in range(len(route) - 1)] + [
            (route[-1], vrp.depot)]
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=route_edges, edge_color=f'C{i}', width=1, arrowsize=10,
                               arrowstyle='-|>')

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def hover(event):
        if event.inaxes != ax:
            return
        cont, ind = nodes.contains(event)
        if cont:
            node_obj = list(graph.nodes)[ind["ind"][0]]
            annot.xy = pos[node_obj]
            node_attr = {'node': node_obj}
            node_attr.update(graph.nodes[node_obj])
            annot.set_text('\n'.join(f'{k}: {v}' for k, v in node_attr.items()))
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


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

            solver = HeuristicSolver(inst, mode=SolverMode.PRED_COST, time_limit=3)
            # solver = GurobiSolver(inst, mode=SolverMode.PRED_COST, mip_gap=0, time_limit=3)
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

    solver = HeuristicSolver(vrp, mode=SolverMode.PRED_COST, time_limit=10)
    # solver = GurobiSolver(vrp, mode=SolverMode.PRED_COST, mip_gap=0, time_limit=40)
    solver.solve()
    print(f'Actual objective: {vrp.actual_obj}')
    print(f'Predicted objective: {solver.get_actual_objective()}')


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

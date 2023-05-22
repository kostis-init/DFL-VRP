import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from domain.vrp_node import VRPNode
from domain.vrp_edge import VRPEdge
from domain.vrp import VRP
from dataclasses import fields

from enums import SolverMode
from heuristic.heuristic_solver import HeuristicSolver
from solver import GurobiSolver
import math
import numpy as np
import torch
import os


def parse_datafile(instance_dir: str) -> VRP:
    # print(f'Parsing datafile: {instance_dir}...')
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
        edge = VRPEdge(node1=node1, node2=node2, distance=row[2], features=row[3:-1].tolist(), cost=row[-1])
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


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def test(model, instances, solver_, is_two_stage=True, verbose=False):

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
            actual_obj = inst.actual_obj
            sol = inst.actual_solution
            actual_edges = [inst.edges[i] for i in range(len(sol)) if sol[i] == 1]
            actual_sols_cost += actual_obj

            if is_two_stage:
                for edge in inst.edges:
                    edge.predicted_cost = model.predict(edge.features)
            else:
                model.eval()
                predicted_edge_costs = model(torch.tensor([edge.features for edge in inst.edges]))
                for i, edge in enumerate(inst.edges):
                    edge.predicted_cost = predicted_edge_costs[i].detach().item()

            solver = solver_(inst, mode=SolverMode.PRED_COST)
            solver.solve()
            predicted_edges = solver.get_active_arcs()
            predicted_obj = solver.get_actual_objective()
            predicted_sols_cost += predicted_obj

            regret += predicted_obj - actual_obj
            correct_edges = set(actual_edges).intersection(predicted_edges)
            accuracy += float(len(correct_edges)) / len(predicted_edges)
            if verbose:
                print(f'Parsed instance {inst}, accuracy: {accuracy}, actual cost: {actual_sols_cost}, predicted cost: {predicted_sols_cost}')
        regret /= len(instances)
        accuracy /= len(instances)
        cost_comparison = predicted_sols_cost / actual_sols_cost
        print(f'Accuracy: {accuracy}, cost comparison: {cost_comparison}, regret: {regret}')


def test_and_draw(trainer, vrp):
    trainer.model.eval()
    print(f'Testing example instance {vrp}, '
          f'predicted cost: {trainer.predict([vrp.edges[0].features])}, '
          f'actual cost: {vrp.edges[0].cost}')
    solver = HeuristicSolver(vrp, time_limit=1)
    solver.solve()
    print('Drawing actual solution')
    draw_solution(solver)
    actual_edges = solver.get_active_arcs()
    for edge in vrp.edges:
        edge.predicted_cost = trainer.predict([edge.features])

    solver = HeuristicSolver(vrp, mode=SolverMode.PRED_COST, time_limit=1)
    solver.solve()
    print('Drawing predicted solution')
    draw_solution(solver)
    predicted_edges = solver.get_active_arcs()
    print(f'Actual edges ({len(actual_edges)}): {actual_edges}')
    print(f'Predicted edges ({len(predicted_edges)}): {predicted_edges}')
    correct_edges = set(actual_edges).intersection(predicted_edges)
    print(f'Correct edges ({len(correct_edges)}): {correct_edges}')



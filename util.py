import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from vrp.vrp_node import VRPNode
from vrp.vrp_edge import VRPEdge
from vrp.vrp import VRP
from dataclasses import fields
from solver import GurobiSolver
import math
import numpy as np


def parse_datafile(instance_dir: str) -> VRP:
    # print(f'Parsing datafile: {instance_dir}...')
    nodes_file_path = f'{instance_dir}/nodes.csv'
    edges_file_path = f'{instance_dir}/edges.csv'

    nodes_columns = {field.name: field.type for field in fields(VRPNode)}
    nodes_df = pd.read_csv(nodes_file_path, skiprows=1, names=nodes_columns, sep=',')
    nodes = [VRPNode(**row) for row in nodes_df.to_dict('records')]

    edges_df = pd.read_csv(edges_file_path, skiprows=0, sep=',')
    edges = []
    for _, row in edges_df.iterrows():
        node1_id, node2_id = int(row[0]), int(row[1])
        node1 = next(node for node in nodes if node.id == node1_id)
        node2 = next(node for node in nodes if node.id == node2_id)
        edge = VRPEdge(node1=node1, node2=node2, distance=row[2], features=row[3:-1].tolist(), cost=row[-1])
        edges.append(edge)

    return VRP(instance_dir, nodes, edges, nodes[0], 50)


def draw_solution(solver) -> None:
    vrp = solver.vrp

    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)

    pos = {i: (i.x, i.y) for i in vrp.nodes}
    node_colors = ['red' if i == vrp.depot else 'green' for i in vrp.nodes]

    fig, ax = plt.subplots()
    # round to 2 decimals
    attrs = {i: {
        # 'start_time': round(solver.get_start_times()[i], 2),
        # 'cumulative_demand': round(solver.get_loads()[i], 2),
        'demand': round(i.demand, 2),
        'service_time': round(i.service_time, 2),
        'ready_time': round(i.ready_time, 2),
        'due_date': round(i.due_time, 2)
    } for i in vrp.nodes}
    nx.set_node_attributes(graph, attrs)

    nodes = nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, node_color=node_colors, node_size=50)

    # draw each edge sequence with a different color
    # find how many different colors are needed by checking the number of routes
    routes = solver.get_routes()
    num_routes = len(routes)
    # generate a list of colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_routes))
    for i, route in enumerate(routes):
        # for each route, draw the edges with the same color
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=route, edge_color=colors[i], width=1, arrowsize=10,
                               arrowstyle='->')

    # nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edge_color='black', width=1, arrowsize=10, arrowstyle='->')

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

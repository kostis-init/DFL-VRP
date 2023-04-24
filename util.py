import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from vrp.vrp_node import VRPNode
from vrp.vrp_edge import VRPEdge
from vrp.vrp import VRP
from dataclasses import fields
from solver import GurobiSolver
import math


def parse_datafile(instance_dir: str) -> VRP:
    # print(f'Parsing datafile: {instance_dir}...')
    nodes_file_path = f'{instance_dir}/nodes.csv'
    edges_file_path = f'{instance_dir}/edges.csv'

    nodes_columns = {field.name: field.type for field in fields(VRPNode)}
    nodes_df = pd.read_csv(nodes_file_path, skiprows=1, names=nodes_columns, sep=',')
    nodes = [VRPNode(**row) for row in nodes_df.to_dict('records')]

    edges_columns = {field.name: field.type for field in fields(VRPEdge)}
    edges_df = pd.read_csv(edges_file_path, skiprows=1, names=edges_columns, sep=',')
    edges = []
    for _, row in edges_df.iterrows():
        node1_id = int(row['node1'])
        node2_id = int(row['node2'])
        node1 = next(node for node in nodes if node.id == node1_id)
        node2 = next(node for node in nodes if node.id == node2_id)
        edge = VRPEdge(
            node1=node1,
            node2=node2,
            distance=row['distance'],
            distance_to_depot=row['distance_to_depot'],
            total_demand=row['total_demand'],
            is_customer=row['is_customer'],
            total_service_time=row['total_service_time'],
            total_due_time=row['total_due_time'],
            total_ready_time=row['total_ready_time'],
            rain=row['rain'],
            traffic=row['traffic'],
            cost=row['cost']
        )
        edges.append(edge)
    return VRP(instance_dir, nodes, edges, nodes[0], 1_000_000)


def draw_solution(solver: GurobiSolver) -> None:
    vrp = solver.vrp
    active_arcs = [(e.node1, e.node2) for e in solver.get_active_arcs()]

    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)
    graph.add_edges_from(active_arcs)

    pos = {i: (i.x, i.y) for i in vrp.nodes}
    node_colors = ['red' if i == vrp.depot else 'green' for i in vrp.nodes]

    fig, ax = plt.subplots()
    # round to 2 decimals
    attrs = {i: {'start_time': round(solver.get_start_times()[i], 2),
                 'cumulative_demand': round(solver.get_loads()[i], 2),
                 'demand': round(i.demand, 2),
                 'service_time': round(i.service_time, 2),
                 'ready_time': round(i.ready_time, 2),
                 'due_date': round(i.due_time, 2)} for i in vrp.nodes}
    nx.set_node_attributes(graph, attrs)

    nodes = nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, node_color=node_colors, node_size=50)
    nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edge_color='black', width=1, arrowsize=10, arrowstyle='->')

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
    # plot on maximized window
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show()


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

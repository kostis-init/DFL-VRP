import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses.vrp_node import VRPNode
from dataclasses.vrp import VRP
from dataclasses import fields


def parse_datafile(file_path: str) -> VRP:
    """
    Parse data file and return VRP instance
    :param file_path: path to data file
    :return: VRP instance
    """
    columns = {field.name: field.type for field in fields(VRPNode)}
    df = pd.read_csv(file_path, sep='\s+', skiprows=1, names=columns)
    nodes = [VRPNode(**row) for row in df.to_dict('records')]
    return VRP(file_path, 1, 200, nodes[0], nodes[1:])


def draw_solution(active_arcs, vrp) -> None:
    """
    Draw solution graph using matplotlib and networkx libraries
    :param active_arcs: list of active arcs
    :param vrp: VRP instance
    :return: None
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.get_all_nodes())
    graph.add_edges_from(active_arcs)
    pos = {i: (i.x, i.y) for i in vrp.get_all_nodes()}
    node_colors = ['r' if i == vrp.depot else 'b' for i in vrp.get_all_nodes()]
    nx.draw_networkx(graph, pos, node_color=node_colors, with_labels=False, node_size=30)
    plt.show()

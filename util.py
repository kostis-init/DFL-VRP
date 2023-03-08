import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import networkx as nx
from models.vrp_node import VRPNode
from models.vrp import VRP
from dataclasses import fields
from dataclasses import replace

from models.vrp_vehicle import VRPVehicle


def parse_datafile(file_path: str) -> VRP:
    """
    Parse data file and return VRP instance
    :param file_path: path to data file
    :return: VRP instance
    """
    columns = {field.name: field.type for field in fields(VRPNode)}
    df = pd.read_csv(file_path, sep='\s+', skiprows=1, names=columns)
    nodes = [VRPNode(**row) for row in df.to_dict('records')]
    depot_dup = replace(nodes[0])
    depot_dup.id = 1000
    return VRP(file_path, nodes[0], nodes[1:] + [depot_dup])


def draw_solution(active_arcs: [tuple[VRPNode, VRPNode, VRPVehicle]], vrp: VRP) -> None:
    """
    Draw solution graph using matplotlib and networkx libraries
    :param active_arcs: list of active arcs
    :param vrp: VRP instance
    :return: None
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)
    graph.add_edges_from(active_arcs)
    nx.draw_networkx(G=graph, pos={i: (i.x, i.y) for i in vrp.nodes}, with_labels=False, node_size=20,
                     node_color=['red' if i == vrp.depot else 'green' for i in vrp.nodes])
    plt.show()

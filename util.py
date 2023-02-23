import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import networkx as nx
from models.vrp_node import VRPNode
from models.vrp import VRP
from dataclasses import fields

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
    vehicles = [VRPVehicle(i) for i in range(1, 6)]
    return VRP(file_path, vehicles, nodes[0], nodes[1:])


def draw_solution(active_arcs: [tuple[VRPNode, VRPNode, VRPVehicle]], vrp: VRP) -> None:
    """
    Draw solution graph using matplotlib and networkx libraries
    :param active_arcs: list of active arcs
    :param vrp: VRP instance
    :return: None
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)
    graph.add_edges_from([arc[:2] for arc in active_arcs])
    nx.draw_networkx(G=graph,
                     pos={i: (i.x, i.y) for i in vrp.nodes},
                     node_color=['r' if i == vrp.depot else 'y' for i in vrp.nodes],
                     with_labels=False,
                     node_size=1,
                     edge_color=[arc[2].id for arc in active_arcs])
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import networkx as nx
from models.vrp_node import VRPNode
from models.vrp import VRP
from dataclasses import fields
from dataclasses import replace

from models.vrp_vehicle import VRPVehicle
from solver import GurobiSolver


def parse_datafile(file_path: str) -> VRP:
    """
    Parse data file and return VRP instance
    :param file_path: path to data file
    :return: VRP instance
    """
    columns = {field.name: field.type for field in fields(VRPNode)}
    df = pd.read_csv(file_path, sep='\s+', skiprows=1, names=columns)
    nodes = [VRPNode(**row) for row in df.to_dict('records')]
    return VRP(file_path, nodes[0], nodes[1:])


def draw_solution(solver: GurobiSolver) -> None:
    vrp = solver.vrp
    active_arcs = solver.get_active_arcs()
    start_times = solver.get_start_times()

    graph = nx.DiGraph()
    graph.add_nodes_from(vrp.nodes)
    graph.add_edges_from(active_arcs)

    # add as labels the start times of the nodes
    # labels = {i: f'{int(start_times[i])}' for i in vrp.nodes}
    # add labels to graph
    # nx.draw_networkx_labels(G=graph, pos={i: (i.x, i.y) for i in vrp.nodes}, labels=labels)
    # nx.draw(G=graph, labels=labels, with_labels=True)
    nx.draw_networkx(G=graph, pos={i: (i.x, i.y) for i in vrp.nodes}, with_labels=True,
                     # labels=labels,
                     node_size=30,
                     node_color=['red' if i == vrp.depot else 'green' for i in vrp.nodes])
    plt.show()

from dataclasses import dataclass

from vrp.vrp_edge import VRPEdge
from vrp.vrp_node import VRPNode


@dataclass
class VRP:
    name: str  # name of the problem instance
    nodes: list[VRPNode]  # list of nodes
    edges: list[VRPEdge]  # list of edges
    depot: VRPNode  # depot node
    capacity: float  # capacity of vehicles

    def __post_init__(self):
        self.customers = [node for node in self.nodes if node != self.depot]
        self.demands = {i: i.demand for i in self.nodes}
        self.service_times = {i: i.service_time for i in self.nodes}
        self.ready_times = {i: i.ready_time for i in self.nodes}
        self.due_times = {i: i.due_time for i in self.nodes}
        self.incoming_edges = {i: [edge for edge in self.edges if edge.node2 == i] for i in self.nodes}
        self.outgoing_edges = {i: [edge for edge in self.edges if edge.node1 == i] for i in self.nodes}
        self.find_edge_from_nodes = {(edge.node1, edge.node2): edge for edge in self.edges}
        self.actual_solution = None
        self.actual_obj = None

    def __str__(self):
        return f"VRP instance: {self.name}"

    def __repr__(self):
        return f"VRP instance: {self.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, item: VRPNode):
        return item in self.nodes

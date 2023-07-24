from dataclasses import dataclass

from dfl_vrp.domain.vrp_node import VRPNode


@dataclass
class VRPEdge:
    node1: VRPNode  # start node
    node2: VRPNode  # end node
    # distance: float  # distance between start and end node
    features: [float]  # list of features
    cost: float  # cost of edge

    def __post_init__(self):
        # self.features = [self.distance] + self.features
        self.predicted_cost = None

    def __str__(self):
        return f"{self.node1} -> {self.node2}"

    def __repr__(self):
        return f"{self.node1} -> {self.node2}"

    def __eq__(self, other):
        return self.node1 == other.node1 and self.node2 == other.node2

    def __hash__(self):
        return hash((self.node1, self.node2))

from dataclasses import dataclass

from models.vrp_node import VRPNode


@dataclass
class VRPEdge:
    node1: VRPNode
    node2: VRPNode
    distance: float
    distance_to_depot: float
    total_demand: float
    is_customer: bool
    total_service_time: float
    total_due_time: float
    total_ready_time: float
    rain: float
    traffic: float
    cost: float

    def __post_init__(self):
        self.travel_time = self.distance / 50

    def __str__(self):
        return f"{self.node1} -> {self.node2}"

    def __repr__(self):
        return f"{self.node1} -> {self.node2}"

    def __eq__(self, other):
        return self.node1 == other.node1 and self.node2 == other.node2

    def __hash__(self):
        return hash((self.node1, self.node2))

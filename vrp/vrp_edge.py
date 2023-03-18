from dataclasses import dataclass

from vrp.vrp_node import VRPNode


@dataclass
class VRPEdge:
    node1: VRPNode  # start node
    node2: VRPNode  # end node
    distance: float  # distance between start and end node
    distance_to_depot: float  # distance between end node and depot
    total_demand: float  # total demand of both nodes
    is_customer: bool  # is start node or end node a customer
    total_service_time: float  # total service time of both nodes
    total_due_time: float  # total due time of both nodes
    total_ready_time: float  # total ready time of both nodes
    rain: float  # rain
    traffic: float  # traffic
    cost: float  # cost of edge, used for the objective function

    def __post_init__(self):
        self.travel_time = self.distance / 10
        self.features = [self.distance, self.distance_to_depot, self.total_demand, self.is_customer,
                         self.total_service_time, self.total_due_time, self.total_ready_time, self.rain, self.traffic]

    def __str__(self):
        return f"{self.node1} -> {self.node2}"

    def __repr__(self):
        return f"{self.node1} -> {self.node2}"

    def __eq__(self, other):
        return self.node1 == other.node1 and self.node2 == other.node2

    def __hash__(self):
        return hash((self.node1, self.node2))

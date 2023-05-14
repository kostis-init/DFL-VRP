from dataclasses import dataclass

from domain.vrp_edge import VRPEdge
from domain.vrp_node import VRPNode


@dataclass
class VRP:
    name: str  # name of the problem instance
    nodes: [VRPNode]  # list of nodes
    edges: [VRPEdge]  # list of edges
    depot: VRPNode  # depot node
    capacity: float  # capacity of vehicles
    actual_routes: [[VRPNode]]  # the actual routes of the optimal solution
    actual_solution: [int]  # the optimal solution, i.e. the decision variables of the optimal solution
    actual_obj: float  # actual objective value of the optimal solution

    def __post_init__(self):
        self.customers = [node for node in self.nodes if node != self.depot]
        self.incoming_edges = {i: [edge for edge in self.edges if edge.node2 == i] for i in self.nodes}
        self.outgoing_edges = {i: [edge for edge in self.edges if edge.node1 == i] for i in self.nodes}
        self.find_edge_from_nodes = {(edge.node1, edge.node2): edge for edge in self.edges}
        if self.actual_routes is not None:
            self.actual_solution = self.get_decision_variables(self.actual_routes)


    def route_cost(self, route: [VRPNode]):
        return self.cost(self.depot, route[0]) + sum(
            self.cost(node1, node2) for node1, node2 in zip(route[:-1], route[1:])) + self.cost(route[-1], self.depot)

    def route_spo_cost(self, route: [VRPNode]):
        return self.spo_cost(self.depot, route[0]) + sum(
            self.spo_cost(node1, node2) for node1, node2 in zip(route[:-1], route[1:])) + self.spo_cost(route[-1],
                                                                                                        self.depot)

    def route_pred_cost(self, route: [VRPNode]):
        return self.pred_cost(self.depot, route[0]) + sum(
            self.pred_cost(node1, node2) for node1, node2 in zip(route[:-1], route[1:])) + self.pred_cost(route[-1],
                                                                                                          self.depot)

    def route_distance(self, route: [VRPNode]):
        return self.distance(self.depot, route[0]) + sum(
            self.distance(node1, node2) for node1, node2 in zip(route[:-1], route[1:])) + self.distance(route[-1],
                                                                                                        self.depot)

    def distance(self, node1: VRPNode, node2: VRPNode):
        return self.find_edge_from_nodes[(node1, node2)].distance

    def cost(self, node1: VRPNode, node2: VRPNode):
        return self.find_edge_from_nodes[(node1, node2)].cost

    def pred_cost(self, node1: VRPNode, node2: VRPNode):
        return self.find_edge_from_nodes[(node1, node2)].predicted_cost

    def spo_cost(self, node1: VRPNode, node2: VRPNode):
        return - self.find_edge_from_nodes[(node1, node2)].cost + 2 * self.find_edge_from_nodes[
            (node1, node2)].predicted_cost

    def get_decision_variables(self, routes: [[VRPNode]]) -> [int]:
        """
        Returns a list of 0-1 variables indicating whether an edge is used.
        """
        decision_vars = [0] * len(self.edges)
        for route in routes:
            decision_vars[self.edges.index(self.find_edge_from_nodes[(self.depot, route[0])])] = 1
            decision_vars[self.edges.index(self.find_edge_from_nodes[(route[-1], self.depot)])] = 1
            for node1, node2 in zip(route[:-1], route[1:]):
                decision_vars[self.edges.index(self.find_edge_from_nodes[(node1, node2)])] = 1
        return decision_vars

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

    def get_loads(self, routes):
        """
        Returns a dictionary of cumulative loads for each node.
        """
        loads = dict()
        for route in routes:
            prev = 0
            for node in route:
                if node != self.depot:
                    loads[node] = prev + node.demand
                    prev = loads[node]
        return loads

import copy

from alns import ALNS, State
from vrp.vrp import VRP
from vrp.vrp_edge import VRPEdge
import random

from vrp.vrp_node import VRPNode


class CvrpState(State):
    """
    Solution state for CVRP.
    """

    def __init__(self, vrp, routes, unassigned=None):
        self.vrp = vrp
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        return sum(self.route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, node: VRPNode):
        """
        Return the route that contains the passed-in node.
        """
        for route in self.routes:
            if node in route:
                return route

        raise ValueError(f"Solution does not contain node {node}.")

    def route_cost(self, route: list[(VRPNode, VRPNode)]):
        """
        Computes the cost of a single route.
        """
        cost = 0
        for node1, node2 in route:
            edge = self.vrp.find_edge_from_nodes[(node1, node2)]
            cost += edge.cost
        return cost


def get_initial_solution(vrp: VRP):
    """
    Returns an initial solution for the passed-in VRP instance.
    """
    # initialize routes
    routes = []
    unvisited = vrp.customers.copy()
    # while there are unvisited nodes
    while unvisited:
        # create a new route
        route = []
        # select the nearest node to the depot
        node = random.choice(unvisited)
        # add the node to the route
        route.append((vrp.depot, node))
        # remove the node from the unvisited nodes
        unvisited.remove(node)
        # while the route is not full
        while sum(vrp.demands[node] for node in route) < vrp.capacity:
            # select a random node from the unvisited nodes
            node = random.choice(unvisited)
            # add the node to the route
            route.append((route[-1][1], node))
            # remove the node from the unvisited nodes
            unvisited.remove(node)
        # add the depot to the route
        route.append((route[-1][1], vrp.depot))
        # add the route to the routes
        routes.append(route)

    return CvrpState(vrp, routes)

class HeuristicSolver:

    def __init__(self, vrp: VRP):
        self.vrp = vrp
        self.alns = ALNS()
        self.initial_solution = get_initial_solution(vrp)



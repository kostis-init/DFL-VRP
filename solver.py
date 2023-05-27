import gurobipy as gp

from enums import SolverMode
from domain.vrp import VRP
from domain.vrp_edge import VRPEdge


class GurobiSolver:
    """
    Gurobi solver for the CVRP.
    """

    def __init__(self,
                 vrp: VRP = None,
                 mip_gap: float = 0.2,
                 verbose: int = 0,
                 time_limit: float = 60,
                 mode: SolverMode = SolverMode.TRUE_COST):
        """
        Initializes the Gurobi solver.
        :param vrp: The VRP instance.
        :param mip_gap: The MIP gap for the solver. This is the difference between the best integer solution found and
        the best possible integer solution.
        :param verbose: The verbosity of the solver. The default value is 0. If set to 1, the solver will print
        information about the progress of the optimization.
        :param time_limit: The time limit for the solver in seconds. The default value is 0, which means that the
        solver will run until it finds an optimal solution.
        :param mode: The mode of the solver. This determines the objective function of the model.
        """

        if vrp is None:
            raise Exception('VRP is None')
        self.vrp = vrp
        self.model = gp.Model('CVRPTW')
        self.model.Params.OutputFlag = verbose
        if mip_gap is not None:
            self.model.Params.MIPGap = mip_gap
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit

        self.x = None
        self.u = None

        self.add_decision_variables()
        self.set_objective(mode)
        self.add_constraints()

    def set_objective(self, mode: SolverMode) -> None:
        """
        Sets the objective function of the model, depending on the mode.
        :param mode: The mode that determines the objective function of the model.
        :return: None
        """
        if mode == SolverMode.TRUE_COST:
            self.model.setObjective(
                gp.quicksum(self.x[edge] * edge.cost for edge in self.vrp.edges),
                gp.GRB.MINIMIZE)
        elif mode == SolverMode.PRED_COST:
            self.model.setObjective(
                gp.quicksum(self.x[edge] * edge.predicted_cost for edge in self.vrp.edges),
                gp.GRB.MINIMIZE)
        elif mode == SolverMode.SPO:
            self.model.setObjective(
                gp.quicksum(self.x[edge] * (2 * edge.predicted_cost - edge.cost) for edge in self.vrp.edges),
                gp.GRB.MINIMIZE)
        elif mode == SolverMode.DISTANCE:
            self.model.setObjective(
                gp.quicksum(self.x[edge] * edge.distance for edge in self.vrp.edges),
                gp.GRB.MINIMIZE)
        else:
            raise Exception('Invalid solver mode')

    def solve(self) -> None:
        """
        Solves the model.
        :return: None
        """
        self.model.optimize()

    def debug(self) -> None:
        """
        Prints the model to a file and computes the Irreducible Inconsistent Subsystem (IIS) of the model.
        :return: None
        """
        self.model.computeIIS()
        self.model.write(f'{self.vrp.name}_model.lp')

    def get_active_arcs(self) -> [VRPEdge]:
        """
        Returns the list of the edges that are used in the solution, i.e. the edges with a value of 1 in the solution.
        :return: list of active edges
        """
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

    def get_decision_variables(self) -> [int]:
        """
        Returns the edge selection decision, i.e. a list of 0-1 variables indicating whether an edge is used in the
        solution or not.
        :return: list of 0-1 decision variables
        """
        return [self.x[edge].x for edge in self.x.keys()]

    def get_loads(self):
        return {node: self.u[node].x for node in self.u.keys()}

    def get_routes(self):
        """
        Returns the list of routes in the solution.
        :return: list of routes
        """
        routes = []
        # append the first edge of each route
        for edge in self.x.keys():
            if self.x[edge].x > 0.5 and edge.node1 == self.vrp.depot:
                routes.append([edge.node2])
        # append the remaining edges of each route
        for route in routes:
            while route[-1] != self.vrp.depot:
                for edge in self.x.keys():
                    if self.x[edge].x > 0.5 and edge.node1 == route[-1]:
                        route.append(edge.node2)
        # remove the depot (last node) from each route
        for route in routes:
            route.pop()
        return routes

    def get_actual_objective(self):
        return sum(self.x[edge].x * edge.cost for edge in self.vrp.edges)

    def get_spo_objective(self):
        return sum(self.x[edge].x * (2 * edge.predicted_cost - edge.cost) for edge in self.vrp.edges)

    def get_pred_objective(self):
        return sum(self.x[edge].x * edge.predicted_cost for edge in self.vrp.edges)

    def add_decision_variables(self) -> None:
        """
        Adds the decision variables to the model. The decision variables are:
        - x[e] = a binary variable indicating whether edge e is used in the solution
        - u[i] = the cumulative load at node i, i.e. the total load of the route up to node i (including i)

        :return: None
        """
        self.x = self.model.addVars(self.vrp.edges, vtype=gp.GRB.BINARY, name='x')
        self.u = self.model.addVars(self.vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='u', lb=0.0, ub=self.vrp.capacity)

    def add_constraints(self) -> None:
        """
        Adds the constraints to the model. The constraints are:
        1. The flow into each node must equal the flow out of each node.
        2. Each customer must be visited exactly once.
        3. The load at the depot must be 0.
        4. Sub-tour elimination (Miller-Tucker-Zemlin).

        :return: None
        """
        vrp = self.vrp
        # 1.
        self.model.addConstrs(gp.quicksum(self.x[inc] for inc in vrp.incoming_edges[i]) ==
                              gp.quicksum(self.x[out] for out in vrp.outgoing_edges[i])
                              for i in vrp.nodes)
        # 2.
        self.model.addConstrs(gp.quicksum(self.x[edge] for edge in vrp.outgoing_edges[i]) == 1
                              for i in vrp.customers)
        # 3.
        self.model.addConstr(self.u[vrp.depot] == 0, name='depot_load')
        # 4.
        self.model.addConstrs(
            self.u[j] - self.u[i] >= j.demand - vrp.capacity * (1 - self.x[vrp.find_edge_from_nodes[(i, j)]])
            for i in vrp.nodes for j in vrp.customers if i != j and (i, j) in vrp.find_edge_from_nodes)

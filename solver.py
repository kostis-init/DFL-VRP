import gurobipy as gp
from vrp.vrp import VRP
from vrp.vrp_edge import VRPEdge


class GurobiSolver:

    def __init__(self, vrp: VRP, mip_gap: float = 0.0, verbose: int = 0):
        self.vrp = vrp
        self.model = gp.Model('CVRPTW')
        self.model.Params.MIPGap = mip_gap
        self.model.Params.OutputFlag = verbose
        self.x = None
        self.s = None
        self.u = None

        self.add_decision_variables()
        self.set_actual_objective()
        self.add_constraints()

    def solve(self):
        self.model.optimize()

    def debug(self):
        self.model.computeIIS()
        self.model.write(f'{self.vrp.name}_model.lp')

    def get_active_arcs(self) -> [VRPEdge]:
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

    def get_decision_variables(self):
        return [self.x[edge].x for edge in self.x.keys()]

    def get_start_times(self):
        return {node: self.s[node].x for node in self.s.keys()}

    def get_loads(self):
        return {node: self.u[node].x for node in self.u.keys()}

    def get_obj_val(self):
        return self.model.objVal

    def get_routes(self):
        """
        Returns the list of routes in the solution.
        :return: list of routes
        """
        routes = []
        # append the first edge of each route
        for edge in self.x.keys():
            if self.x[edge].x > 0.5 and edge.node1 == self.vrp.depot:
                routes.append([(edge.node1, edge.node2)])
        # append the remaining edges of each route
        for route in routes:
            while route[-1][1] != self.vrp.depot:
                for edge in self.x.keys():
                    if self.x[edge].x > 0.5 and edge.node1 == route[-1][1]:
                        route.append((edge.node1, edge.node2))
        return routes

    def set_spo_objective(self):
        self.model.setObjective(
            gp.quicksum(self.x[edge] * (edge.cost - 2 * edge.predicted_cost) for edge in self.vrp.edges),
            gp.GRB.MAXIMIZE)

    def set_predicted_objective(self):
        self.model.setObjective(
            gp.quicksum(self.x[edge] * edge.predicted_cost for edge in self.vrp.edges),
            gp.GRB.MINIMIZE)

    def set_actual_objective(self):
        self.model.setObjective(
            gp.quicksum(self.x[edge] * edge.cost for edge in self.vrp.edges),
            gp.GRB.MINIMIZE)

    def add_decision_variables(self):
        vrp = self.vrp
        # x[e] = 1 if edge e is used, 0 otherwise
        self.x = self.model.addVars(vrp.edges, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which node i is visited
        self.s = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='s', lb=0.0)
        # u[i] = cumulative amount of load at node i (including the demand of customer i)
        self.u = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='u', lb=0.0, ub=vrp.capacity)

    def add_constraints(self):
        vrp = self.vrp
        # ARC CONSTRAINTS
        # 1. Balance of flow at each node.
        self.model.addConstrs(gp.quicksum(self.x[incoming_edge] for incoming_edge in vrp.incoming_edges[i]) ==
                              gp.quicksum(self.x[outgoing_edge] for outgoing_edge in vrp.outgoing_edges[i])
                              for i in vrp.nodes)
        # 2. Every customer is visited once.
        self.model.addConstrs(gp.quicksum(self.x[edge] for edge in vrp.outgoing_edges[i]) == 1 for i in vrp.customers)

        # TIME CONSTRAINTS
        # 3. Customers are visited after the previous customer is visited.
        self.model.addConstrs((self.x[vrp.find_edge_from_nodes[(i, j)]] == 1) >> (self.s[j] >= self.s[i] + 1)
                              for i in vrp.nodes for j in vrp.customers if i != j)
        # 4. Nodes are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= i.ready_time for i in vrp.nodes)
        # 5. Nodes are visited only before they are due.
        self.model.addConstrs(self.s[i] <= i.due_time for i in vrp.nodes)

        # CAPACITY CONSTRAINTS
        # 6. Depot is always empty.
        self.model.addConstr(self.u[vrp.depot] == 0.0, name='depot_empty')
        # 7. Load at a customer is the sum of the load at the previous customer plus the demand of the current customer.
        self.model.addConstrs((self.x[vrp.find_edge_from_nodes[(i, j)]] == 1) >> (self.u[j] == self.u[i] + j.demand)
                              for i in vrp.nodes for j in vrp.customers if i != j)

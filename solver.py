import gurobipy as gp
from vrp.vrp import VRP


class GurobiSolver:

    def __init__(self, vrp: VRP, mip_gap=0.2, time_limit=5, verbose=False):
        self.vrp = vrp
        self.model = gp.Model('CVRPTW')
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables

        # x[e] = 1 if edge e is used, 0 otherwise
        self.x = self.model.addVars(vrp.edges, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which node i is visited
        self.s = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='s', lb=0.0)
        # u[i] = cumulative amount of load at node i (including the demand of customer i)
        self.u = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='u', lb=0.0, ub=vrp.capacity)

        # Add objective function

        # For now, we only consider the travel time
        self.model.setObjective(gp.quicksum(self.x[edge] * edge.cost for edge in vrp.edges), gp.GRB.MINIMIZE)

        # Add constraints

        # ARC CONSTRAINTS
        # 1. Balance of flow at each node.
        self.model.addConstrs(gp.quicksum(self.x[incoming_edge] for incoming_edge in vrp.incoming_edges[i]) ==
                              gp.quicksum(self.x[outgoing_edge] for outgoing_edge in vrp.outgoing_edges[i])
                              for i in vrp.nodes)
        # 2. Every customer is visited once.
        self.model.addConstrs(gp.quicksum(self.x[edge] for edge in vrp.outgoing_edges[i]) == 1 for i in vrp.customers)

        # TIME CONSTRAINTS
        # 3. Customers are visited after the previous customer is visited and serviced plus the travel time.
        # maximum_time = max(i.due_time + i.service_time + vrp.get_edge(i, j).travel_time - i.ready_time
        #                    for i in vrp.nodes for j in vrp.nodes if i != j)
        maximum_time = 1_000_000
        self.model.addConstrs(self.s[j] >= self.s[i] + i.service_time + vrp.get_edge(i, j).travel_time
                              - (1 - self.x[vrp.get_edge(i, j)]) * maximum_time
                              for i in vrp.nodes for j in vrp.customers if i != j)
        # 4. Nodes are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= i.ready_time for i in vrp.nodes)
        # 5. Nodes are visited only before they are due.
        self.model.addConstrs(self.s[i] <= i.due_time for i in vrp.nodes)

        # CAPACITY CONSTRAINTS
        # 6. Depot is always empty.
        self.model.addConstr(self.u[vrp.depot] == 0.0, name='depot_empty')
        # 7. Load at a customer is the sum of the load at the previous customer plus the demand of the current customer.
        self.model.addConstrs((self.x[vrp.get_edge(i, j)] == 1) >> (self.u[j] == self.u[i] + j.demand)
                              for i in vrp.nodes for j in vrp.customers if i != j)

    def solve(self):
        self.model.optimize()
        # self.model.computeIIS()
        # self.model.write('model.lp')

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

    def get_start_times(self):
        return {node: self.s[node].x for node in self.s.keys()}

    def get_loads(self):
        return {node: self.u[node].x for node in self.u.keys()}

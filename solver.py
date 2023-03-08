import gurobipy as gp
from ortools.constraint_solver import pywrapcp

from models.vrp import VRP


class GurobiSolver:

    def __init__(self, vrp: VRP, mip_gap=0.2, time_limit=5, verbose=False):
        self.vrp = vrp
        self.model = gp.Model('CVRPTW')
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables

        # x[i, j] = 1 if arc (i, j) is used on the route
        self.x = self.model.addVars(vrp.arcs, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which node i is visited
        self.s = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='s', lb=0.0)
        # u[i] = cumulative amount of load at node i (including the demand of customer i)
        self.u = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='u', lb=0.0, ub=vrp.capacity)

        # Add objective function

        # For now, we only consider the travel time
        self.model.setObjective(
            gp.quicksum(self.x[i, j] * vrp.travel_times[i, j] for i, j in vrp.arcs), gp.GRB.MINIMIZE)

        # Add constraints

        # ARC CONSTRAINTS
        # 1. No travel from a node to itself.
        self.model.addConstrs(self.x[i, i] == 0 for i in vrp.nodes)
        # 2. Balance
        self.model.addConstrs(
            gp.quicksum(self.x[i, j] for j in vrp.nodes) == gp.quicksum(self.x[j, i] for j in vrp.nodes)
            for i in vrp.nodes)
        # 3. Every customer is visited once.
        self.model.addConstrs(gp.quicksum(self.x[i, j] for i in vrp.nodes) == 1
                              for j in vrp.customers)

        # TIME CONSTRAINTS
        # 4. Customers are visited after the previous customer is visited plus the travel time.
        maximum_time = max(i.due_date + vrp.travel_times[i, j] - i.ready_time for i, j in vrp.arcs)
        self.model.addConstrs(self.s[j] >= self.s[i] + vrp.travel_times[i, j] - (1 - self.x[i, j]) * maximum_time
                              for i in vrp.nodes for j in vrp.customers)
        # 5. Customers are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= i.ready_time for i in vrp.nodes)
        # 6. Customers are visited only before they are due.
        self.model.addConstrs(self.s[i] <= i.due_date for i in vrp.nodes)

        # CAPACITY CONSTRAINTS
        # 7. Depot is always empty.
        self.model.addConstr(self.u[vrp.depot] == 0.0, name='depot_empty')
        # 8. Load at a customer is the sum of the load at the previous customer plus the demand of the current customer.
        self.model.addConstrs((self.x[i, j] == 1) >> (self.u[j] == self.u[i] + j.demand)
                              for i in vrp.nodes for j in vrp.customers)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

    def get_start_times(self):
        return {node: self.s[node].x for node in self.s.keys()}

# class ORSolver:
#
#     def __init__(self, vrp: VRP, costs, travel_times, mip_gap=0.2, time_limit=5, verbose=False):
#         manager = pywrapcp.RoutingIndexManager(len(vrp.nodes), len(vrp.vehicles), vrp.depot.id)
#         routing = pywrapcp.RoutingModel(manager)

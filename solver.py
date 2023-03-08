import gurobipy as gp
from ortools.constraint_solver import pywrapcp

from models.vrp import VRP


class GurobiSolverNoDemand:

    def __init__(self, vrp: VRP, mip_gap=0.2, time_limit=5, verbose=False):
        self.model = gp.Model('CVRPTW')
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables

        # x[i, j] = 1 if arc (i, j) is used on the route
        self.x = self.model.addVars(vrp.arcs, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which customer i is visited
        self.s = self.model.addVars(vrp.customers, vtype=gp.GRB.CONTINUOUS, name='s', lb=0.0)

        # Add objective function

        # For now, we only consider the travel time
        self.model.setObjective(
            gp.quicksum(self.x[i, j] * vrp.travel_times[i, j] for i, j in vrp.arcs), gp.GRB.MINIMIZE)

        # Add constraints
        # Note: Using the formulation from https://how-to.aimms.com/Articles/332/332-Formulation-CVRP.html

        # 1. No travel from a node to itself.
        self.model.addConstrs(self.x[i, i] == 0 for i in vrp.nodes)
        # 2. Balance
        self.model.addConstrs(
            gp.quicksum(self.x[i, j] for j in vrp.nodes) == gp.quicksum(self.x[j, i] for j in vrp.nodes)
            for i in vrp.nodes)
        # 3. Every customer is visited once.
        self.model.addConstrs(gp.quicksum(self.x[i, j] for i in vrp.nodes) == 1
                              for j in vrp.customers)
        # 4. Every vehicle leaves the depot.
        # self.model.addConstrs(self.x[vrp.depot, j] for j in vrp.customers >= 1)
        # 5. Customers are visited after the previous customer is visited plus the travel time.
        maximum_time = max(i.due_date + vrp.travel_times[i, j] - i.ready_time for i, j in vrp.arcs)
        self.model.addConstrs(self.s[j] >= self.s[i] + vrp.travel_times[i, j] - (1 - self.x[i, j]) * maximum_time
                              for i in vrp.customers for j in vrp.customers)
        # 6. Customers are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= i.ready_time for i in vrp.customers)
        # 7. Customers are visited only before they are due.
        self.model.addConstrs(self.s[i] <= i.due_date for i in vrp.customers)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

# class ORSolver:
#
#     def __init__(self, vrp: VRP, costs, travel_times, mip_gap=0.2, time_limit=5, verbose=False):
#         manager = pywrapcp.RoutingIndexManager(len(vrp.nodes), len(vrp.vehicles), vrp.depot.id)
#         routing = pywrapcp.RoutingModel(manager)

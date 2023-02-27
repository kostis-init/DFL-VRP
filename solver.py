import gurobipy as gp
from ortools.constraint_solver import pywrapcp

from models.vrp import VRP


class GurobiSolver:

    def __init__(self, vrp: VRP, mip_gap=0.2, time_limit=5, verbose=False):
        self.model = gp.Model('CVRPTW')
        self.model.modelSense = gp.GRB.MINIMIZE
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables

        # x[i, j, k] = 1 if arc (i, j) is used on the route of vehicle k
        self.x = self.model.addVars(vrp.arcs, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which customer i is visited
        self.s = self.model.addVars(vrp.customers, vtype=gp.GRB.CONTINUOUS, name='s')

        # Add objective function

        # For now, we only consider the travel time
        self.model.setObjective(gp.quicksum(self.x[i, j, k] * vrp.travel_times[i, j, k] for i, j, k in vrp.arcs))

        # Add constraints
        # Note: Using the formulation from https://how-to.aimms.com/Articles/332/332-Formulation-CVRP.html

        # 1. No travel from a node to itself.
        self.model.addConstrs(self.x[i, i, k] == 0 for i in vrp.nodes for k in vrp.vehicles)
        # 2. Every vehicle leaves every node that it enters.
        self.model.addConstrs(
            gp.quicksum(self.x[i, j, k] for j in vrp.nodes) == gp.quicksum(self.x[j, i, k] for j in vrp.nodes)
            for i in vrp.nodes for k in vrp.vehicles)
        # 3. Every customer is visited once.
        # Note: Combined with constraint 2, we ensure that every node is left by the same vehicle that entered it.
        self.model.addConstrs(gp.quicksum(self.x[i, j, k] for i in vrp.nodes for k in vrp.vehicles) == 1
                              for j in vrp.customers)
        # 4. Every vehicle leaves the depot.
        # Note: Combined with constraint 2, we ensure that every vehicle arrives again at the depot.
        # Note: If we use strict equality (== instead of >=), we ensure that every vehicle leaves depot exactly once.
        self.model.addConstrs(gp.quicksum(self.x[vrp.depot, j, k] for j in vrp.customers) == 1 for k in vrp.vehicles)
        # 5. Customers are visited after the previous customer is visited plus the travel time.
        maximum_time = max(i.due_date + vrp.travel_times[i, j, k] - i.ready_time
                           for i in vrp.nodes for j in vrp.nodes for k in vrp.vehicles)
        self.model.addConstrs(self.s[j] >= self.s[i] + vrp.travel_times[i, j, k] - (1 - self.x[i, j, k]) * maximum_time
                              for i in vrp.customers for j in vrp.customers for k in vrp.vehicles)
        # 6. Customers are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= i.ready_time for i in vrp.customers)
        # 7. Customers are visited only before they are due.
        self.model.addConstrs(self.s[i] <= i.due_date for i in vrp.customers)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]


class ORSolver:

    def __init__(self, vrp: VRP, costs, travel_times, mip_gap=0.2, time_limit=5, verbose=False):
        manager = pywrapcp.RoutingIndexManager(len(vrp.nodes), len(vrp.vehicles), vrp.depot.id)
        routing = pywrapcp.RoutingModel(manager)

import gurobipy as gp
from ortools.constraint_solver import pywrapcp

from models.vrp import VRP


class GurobiSolverNoTime:

    def __init__(self, vrp: VRP, mip_gap=0.2, time_limit=5, verbose=False):
        self.model = gp.Model('CVRP')
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables

        # x[i, j] = 1 if arc (i, j) is used on the route
        self.x = self.model.addVars(vrp.arcs, vtype=gp.GRB.BINARY, name='x')
        # u[i] = the cumulative demand of the route up to node i (including i)
        self.u = self.model.addVars(vrp.nodes, vtype=gp.GRB.CONTINUOUS, name='u', lb=0.0, ub=vrp.capacity)

        # Add objective function

        # For now, we only consider the travel time
        self.model.setObjective(
            gp.quicksum(self.x[i, j] * vrp.travel_times[i, j] for i, j in vrp.arcs), gp.GRB.MINIMIZE)

        # Add constraints

        # Flow in
        self.model.addConstrs(gp.quicksum(self.x[i, j] for i in vrp.nodes) == 1 for j in vrp.customers)
        # Flow out
        self.model.addConstrs(gp.quicksum(self.x[i, j] for j in vrp.nodes) == 1 for i in vrp.customers)
        # Depot balance
        self.model.addConstr(
            gp.quicksum(self.x[vrp.depot, j] for j in vrp.customers)
            == gp.quicksum(self.x[i, vrp.depot_dup] for i in vrp.customers))
        # Demand MTZ
        self.model.addConstrs(
            self.u[j] >= self.u[i] + j.demand * self.x[i, j] - vrp.capacity * (1 - self.x[i, j])
            for i, j in vrp.arcs)
        self.model.addConstrs(self.u[i] >= i.demand for i in vrp.nodes)
        self.model.addConstrs(self.x[vrp.depot_dup, j] == 0 for j in vrp.nodes)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

# class ORSolver:
#
#     def __init__(self, vrp: VRP, costs, travel_times, mip_gap=0.2, time_limit=5, verbose=False):
#         manager = pywrapcp.RoutingIndexManager(len(vrp.nodes), len(vrp.vehicles), vrp.depot.id)
#         routing = pywrapcp.RoutingModel(manager)

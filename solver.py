import gurobipy as gp
from models.vrp import VRP



class GurobiSolver:

    def __init__(self, vrp: VRP, costs, mip_gap=0.2, time_limit=5, verbose=False):
        self.model = gp.Model('VRP')
        self.model.modelSense = gp.GRB.MINIMIZE
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # x[i, j] = 1 if arc (i, j) is used on the route
        self.x = self.model.addVars(vrp.get_arcs(), vtype=gp.GRB.BINARY, name='x')
        # u[i] = cumulative demand up to node i on the route (including i)
        self.u = self.model.addVars(vrp.__construct_nodes__(), vtype=gp.GRB.INTEGER, name='u')

        # Add objective function
        self.model.setObjective(gp.quicksum(self.x[i, j] * costs[i, j] for i, j in vrp.get_arcs()))

        # Add constraints
        nodes = vrp.__construct_nodes__()
        customers = vrp.customers
        demands = vrp.get_demands()
        capacity = vrp.capacity
        depot = vrp.depot
        self.model.addConstrs(gp.quicksum(self.x[i, j] for j in nodes) == 1 for i in customers)  # flow out
        self.model.addConstrs(gp.quicksum(self.x[i, j] for i in nodes) == 1 for j in customers)  # flow in
        self.model.addConstrs(self.u[i] >= self.u[j] + demands[i] - capacity * (1 - self.x[j, i])
                              for i in customers for j in customers)  # cumulative demand
        self.model.addConstr(self.u[depot] == 0, name='depot')  # depot
        self.model.addConstrs(self.u[i] >= demands[i] for i in customers)  # demand
        self.model.addConstrs(self.u[i] <= capacity for i in customers)  # capacity

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]


class GurobiSolverMultiVehicle:

    def __init__(self, vrp: VRP, costs, travel_times, mip_gap=0.2, time_limit=5, verbose=False):
        self.model = gp.Model('CVRPTW')
        self.model.modelSense = gp.GRB.MINIMIZE
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = verbose

        # Add decision variables
        # x[i, j, k] = 1 if arc (i, j) is used on the route of vehicle k
        self.x = self.model.addVars(vrp.arcs_per_vehicle, vtype=gp.GRB.BINARY, name='x')
        # s[i] = time point at which customer i is visited
        self.s = self.model.addVars(vrp.customers, vtype=gp.GRB.CONTINUOUS, name='s')

        # Add objective function
        # As objective function, we want to minimize the longest single route.
        # Note: We could also minimize the total route length, but this would be TSP.
        # Start by finding the longest route.
        # self.model.setObjective(gp.max_(gp.quicksum(self.x[i, j, k] * costs[i, j] for i, j in vrp.arcs)
        #                             for k in vrp.vehicles))
        self.model.setObjective(gp.quicksum(self.x[i, j, k] * costs[i, j] for i, j, k in vrp.arcs_per_vehicle))

        # Add constraints
        nodes, customers, demands, depot, vehicles, ready_times, service_times, due_dates, capacities = \
            vrp.nodes, vrp.customers, vrp.demands, vrp.depot, vrp.vehicles, vrp.ready_times, vrp.service_times, \
            vrp.due_dates, vrp.vehicle_capacities
        maximum_amount_time = max(due_dates[i] + travel_times[i, j] - ready_times[i] for i in nodes for j in nodes)

        # 1. No travel from a node to itself.
        self.model.addConstrs(self.x[i, i, k] == 0 for i in nodes for k in vehicles)
        # 2. Every vehicle leaves every node that it enters.
        self.model.addConstrs(gp.quicksum(self.x[i, j, k] for j in nodes) == gp.quicksum(self.x[j, i, k] for j in nodes)
                              for i in nodes for k in vehicles)
        # 3. Every customer is visited once.
        # Note: Combined with constraint 2, we ensure that every node is left by the same vehicle that entered it.
        self.model.addConstrs(gp.quicksum(self.x[i, j, k] for i in nodes for k in vehicles) == 1 for j in customers)
        # 4. Every vehicle leaves the depot.
        # Note: Combined with constraint 2, we ensure that every vehicle arrives again at the depot.
        # Note: If we use strict equality (== instead of >=), we ensure that every vehicle leaves depot and arrives exactly once.
        self.model.addConstrs(gp.quicksum(self.x[depot, j, k] for j in customers) >= 1 for k in vehicles)

        # Sub-tour elimination. Using time windows, we can eliminate sub-tours by adding the following time constraints:
        # 5. Customers are visited after the previous customer is visited plus the travel time.
        self.model.addConstrs(self.s[j] >= self.s[i] + travel_times[i, j] - (1 - self.x[i, j, k]) * maximum_amount_time
                              for i in customers for j in customers for k in vehicles)
        # 6. Customers are visited only after they are ready.
        self.model.addConstrs(self.s[i] >= ready_times[i] for i in customers)
        # 7. Customers are visited only before they are due.
        self.model.addConstrs(self.s[i] <= due_dates[i] for i in customers)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]


# class ORSolver:


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
        self.u = self.model.addVars(vrp.get_all_nodes(), vtype=gp.GRB.INTEGER, name='u')

        # Add objective function
        self.model.setObjective(gp.quicksum(self.x[i, j] * costs[i, j] for i, j in vrp.get_arcs()))

        # Add constraints
        nodes = vrp.get_all_nodes()
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
        self.x = self.model.addVars(vrp.get_arcs_per_vehicle(), vtype=gp.GRB.BINARY, name='x')
        # s[i] = time at which node i is visited
        self.s = self.model.addVars(vrp.get_all_nodes(), vtype=gp.GRB.CONTINUOUS, name='s')

        # Add objective function
        self.model.setObjective(gp.quicksum(self.x[i, j, k] * costs[i, j] for i, j, k in vrp.get_arcs_per_vehicle()))

        # Add constraints
        nodes = vrp.get_all_nodes()
        customers = vrp.customers
        demands = vrp.get_demands()
        depot = vrp.depot
        vehicles = vrp.vehicles
        ready_times = vrp.get_ready_times()
        service_times = vrp.get_service_times()
        due_dates = vrp.get_due_dates()
        capacities = vrp.get_vehicle_capacities()
        maximum_amount_time = max(due_dates[i] + travel_times[i, j] - ready_times[i] for i in nodes for j in nodes)

        # No travel from a node to itself
        self.model.addConstrs(self.x[i, i, k] == 0 for i in nodes for k in vehicles)
        # Vehicle leaves node that it enters
        self.model.addConstrs(gp.quicksum(self.x[i, j, k] for j in nodes) == gp.quicksum(self.x[j, i, k] for j in nodes)
                              for i in nodes for k in vehicles)
        # Ensure that every node is entered once. Together with the first constraint, it ensures that every
        # node is entered only once, and it is left by the same vehicle.
        self.model.addConstrs(gp.quicksum(self.x[i, j, k] for i in nodes for k in vehicles) == 1 for j in customers)
        # Every vehicle leaves the depot. Together with constraint 1, we know that every vehicle arrives again at
        # the depot. If we use == instead of >=, we would know that every vehicle arrives exactly once at the depot.
        self.model.addConstrs(gp.quicksum(self.x[depot, j, k] for j in customers) >= 1 for k in vehicles)
        # Capacity constraint
        self.model.addConstrs(
            gp.quicksum(self.x[i, j, k] * demands[j] for j in customers for i in nodes) <= capacities[k]
            for k in vehicles)
        # Sub-tour elimination. Using time windows, we can eliminate sub-tours by adding the following constraints:
        # Time window constraints. Keep track of the duration of the routes.
        self.model.addConstrs(self.s[j] >= self.s[i] + travel_times[i, j] - (1 - self.x[i, j, k]) * maximum_amount_time
                              for i in nodes for j in customers for k in vehicles)
        # Time window constraints. Ensure that the vehicle arrives at the customer after the ready time.
        self.model.addConstrs(self.s[i] >= ready_times[i] for i in nodes)
        # Time window constraints. Ensure that the vehicle arrives at the customer before the due date.
        self.model.addConstrs(self.s[i] <= due_dates[i] for i in nodes)

    def optimize(self):
        self.model.optimize()

    def get_active_arcs(self):
        return [arc for arc in self.x.keys() if self.x[arc].x > 0.5]

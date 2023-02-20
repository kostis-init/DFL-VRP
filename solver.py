import gurobipy as gp
from models.vrp import VRP


class GurobiSolver:

    def __init__(self, vrp: VRP, costs, mip_gap=0.2, time_limit=5, verbose=True):
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

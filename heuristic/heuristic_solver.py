from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

from enums import SolverMode
from heuristic.destroy_ops import string_removal
from heuristic.init_ops import init_routes_nn
from heuristic.repair_ops import greedy_repair
import numpy.random as rnd

import copy

from alns import State

from domain.vrp import VRP
from domain.vrp_node import VRPNode


class HeuristicSolver:
    def __init__(self,
                 vrp: VRP,
                 mode: SolverMode = SolverMode.TRUE_COST,
                 seed: int = 1234,
                 time_limit: float = 0.1,
                 num_iterations: int = 10000):
        """
        Initialize the heuristic solver. The solver uses the ALNS framework to solve the VRP instance.
        :param vrp: The VRP instance.
        :param mode: The mode of the solver. This determines the objective function of the model.
        :param seed: The seed for the random number generator.
        :param time_limit: The maximum runtime of the solver in seconds.
        :param num_iterations: The maximum number of iterations of the solver.
        """
        self.vrp = vrp
        self.mode = mode

        self.alns = ALNS(rnd.RandomState(seed))
        self.alns.add_destroy_operator(string_removal)
        self.alns.add_repair_operator(greedy_repair)

        self.state = CvrpState(self, init_routes_nn(self), [])
        # The RouletteWheel selection operator is used to select the next operator to apply.
        self.select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
        # The RecordToRecordTravel acceptance operator is used to accept solutions.
        self.accept = RecordToRecordTravel.autofit(max(0.0, self.state.objective()), 0.02, 0, num_iterations)
        self.stop = MaxRuntime(time_limit)

    def solve(self):
        result = self.alns.iterate(self.state, self.select, self.accept, self.stop)
        self.state = result.best_state

    def get_routes(self):
        return self.state.routes

    def get_actual_objective(self):
        return self.state.true_cost()

    def get_pred_objective(self):
        return self.state.pred_cost()

    def get_spo_objective(self):
        return self.state.spo_objective()

    def get_decision_variables(self):
        return self.vrp.get_decision_variables(self.state.routes)

    def get_loads(self):
        return self.vrp.get_loads(self.state.routes)

    def get_active_arcs(self):
        """
        Returns a list of active arcs, i.e. a list of edges that are used in the current solution.
        """
        return [edge for edge, var in zip(self.vrp.edges, self.get_decision_variables()) if var > 0.5]


class CvrpState(State):
    """
    Solution state for CVRP. Contains the routes and the unassigned nodes.
    """

    def __init__(self, solver: HeuristicSolver, routes: [[VRPNode]], unassigned: [VRPNode]):
        self.solver = solver
        self.routes = routes
        self.unassigned = unassigned

    def copy(self):
        return CvrpState(self.solver, copy.deepcopy(self.routes), copy.deepcopy(self.unassigned))

    def spo_objective(self):
        return sum(self.solver.vrp.route_spo_cost(route) for route in self.routes)

    def true_cost(self):
        return sum(self.solver.vrp.route_cost(route) for route in self.routes)

    def pred_cost(self):
        return sum(self.solver.vrp.route_pred_cost(route) for route in self.routes)

    def objective(self) -> float:
        """
        Computes the total route costs.
        """
        mode = self.solver.mode
        if mode == SolverMode.TRUE_COST:
            return self.true_cost()
        elif mode == SolverMode.SPO:
            return self.spo_objective()
        elif mode == SolverMode.PRED_COST:
            return self.pred_cost()
        else:
            raise ValueError(f"Invalid solver mode {mode}.")

    def find_route(self, node: VRPNode):
        """
        Return the route that contains the passed-in node.
        """
        for route in self.routes:
            if node in route:
                return route
        raise ValueError(f"Solution does not contain node {node}.")

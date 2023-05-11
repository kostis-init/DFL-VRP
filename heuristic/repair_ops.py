from enums import SolverMode


def greedy_repair(state, rnd_state):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    rnd_state.shuffle(state.unassigned)
    while state.unassigned:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state)
        if route is not None:
            route.insert(idx, customer)
        else:
            state.routes.append([customer])
    state.routes = [route for route in state.routes if len(route) != 0]
    return state


def best_insert(candidate, state):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        if sum(cust.demand for cust in route) + candidate.demand > state.solver.vrp.capacity:
            continue
        for idx in range(len(route) + 1):
            route.insert(idx, candidate)
            if state.solver.mode == SolverMode.TRUE_COST:
                cost = state.solver.vrp.route_cost(route)
            elif state.solver.mode == SolverMode.SPO:
                cost = state.solver.vrp.route_spo_cost(route)
            elif state.solver.mode == SolverMode.PRED_COST:
                cost = state.solver.vrp.route_pred_cost(route)
            elif state.solver.mode == SolverMode.DISTANCE:
                cost = state.solver.vrp.route_distance(route)
            else:
                raise ValueError(f"Unknown solver mode {state.solver.mode}.")
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx
            route.remove(candidate)

    return best_route, best_idx

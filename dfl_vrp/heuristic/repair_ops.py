from dfl_vrp.enums import SolverMode


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


def regret_insertion(state, rnd_state):
    """
    Inserts the unassigned customers based on the difference between the best and the second-best insertion cost.
    """
    rnd_state.shuffle(state.unassigned)
    while state.unassigned:
        regrets = []
        for customer in state.unassigned:
            # Find the best and second best insertion points for the customer
            best_route1, best_idx1, best_cost1 = None, None, float('inf')
            best_route2, best_idx2, best_cost2 = None, None, float('inf')
            for route in state.routes:
                # Check capacity constraint
                if sum(cust.demand for cust in route) + customer.demand > state.solver.vrp.capacity:
                    continue
                for idx in range(len(route) + 1):
                    route.insert(idx, customer)
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
                    if cost < best_cost1:
                        best_route2, best_idx2, best_cost2 = best_route1, best_idx1, best_cost1
                        best_route1, best_idx1, best_cost1 = route, idx, cost
                    elif cost < best_cost2:
                        best_route2, best_idx2, best_cost2 = route, idx, cost
                    route.remove(customer)
            if best_route1 is not None and best_route2 is not None:
                regret = best_cost2 - best_cost1
                regrets.append((regret, customer, best_route1, best_idx1))

        # Sort customers by regret in descending order and insert the one with the highest regret
        if regrets:
            regrets.sort(key=lambda x: x[0], reverse=True)
            regret, customer, route, idx = regrets[0]
            state.unassigned.remove(customer)
            route.insert(idx, customer)
        else:
            # If no feasible routes, create a new route for the customer with the highest demand that does not exceed the vehicle capacity
            feasible_customers = [customer for customer in state.unassigned if customer.demand <= state.solver.vrp.capacity]
            if feasible_customers:
                customer = max(feasible_customers, key=lambda c: c.demand)
                state.unassigned.remove(customer)
                state.routes.append([customer])

    state.routes = [route for route in state.routes if len(route) != 0]
    return state

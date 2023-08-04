import numpy as np

from dfl_vrp.enums import SolverMode

MAX_STRING_REMOVALS = 3
MAX_STRING_SIZE = 2


DESTRUCTION_SIZE = 0.2


def string_removal(state, rnd_state):
    """
    Remove partial routes around a randomly chosen customer.
    """
    destroyed = state.copy()

    avg_route_size = int(np.mean([len(route) for route in state.routes]))
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

    destroyed_routes = []
    center = rnd_state.choice(destroyed.solver.vrp.customers)

    customers_rest = [c for c in destroyed.solver.vrp.customers if c != center]
    if destroyed.solver.mode == SolverMode.TRUE_COST:
        customers_rest.sort(key=lambda c: destroyed.solver.vrp.cost(center, c))
    elif destroyed.solver.mode == SolverMode.SPO:
        customers_rest.sort(
            key=lambda c: 2 * destroyed.solver.vrp.pred_cost(center, c) - destroyed.solver.vrp.cost(center, c))
    elif destroyed.solver.mode == SolverMode.PRED_COST:
        customers_rest.sort(key=lambda c: destroyed.solver.vrp.pred_cost(center, c))
    elif destroyed.solver.mode == SolverMode.DISTANCE:
        customers_rest.sort(key=lambda c: destroyed.solver.vrp.distance(center, c))
    else:
        raise ValueError(f"Unknown solver mode {destroyed.solver.mode}.")

    for customer in customers_rest:
        if len(destroyed_routes) >= max_string_removals:
            break
        if customer in destroyed.unassigned:
            continue
        route = destroyed.find_route(customer)
        if route in destroyed_routes:
            continue

        customers = remove_string(route, customer, max_string_size, rnd_state)
        destroyed.unassigned.extend(customers)
        destroyed_routes.append(route)

    destroyed.routes = [route for route in destroyed.routes if len(route) != 0]
    return destroyed


def remove_string(route, cust, max_string_size, rnd_state):
    """
    Remove a string that contains the passed-in customer.
    """
    # Find consecutive indices to remove that contain the customer
    size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
    start = route.index(cust) - rnd_state.randint(size)
    idcs = [idx % len(route) for idx in range(start, start + size)]

    # Remove indices in descending order
    removed_customers = []
    for idx in sorted(idcs, reverse=True):
        removed_customers.append(route.pop(idx))

    return removed_customers


def random_removal(state, rnd_state):
    """
    Removes a number of randomly selected customers from the passed-in solution.
    """

    customers_to_remove = int(DESTRUCTION_SIZE * len(state.solver.vrp.customers))
    destroyed = state.copy()

    for customer in rnd_state.choice(destroyed.solver.vrp.customers, customers_to_remove, replace=False):
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(customer)

    destroyed.routes = [route for route in destroyed.routes if len(route) != 0]
    return destroyed



def worst_removal(state, rnd_state):
    """
    Removes the customer whose removal results in the best cost reduction.
    """
    destroyed = state.copy()
    customers_to_remove = int(DESTRUCTION_SIZE * len(state.solver.vrp.customers))

    for _ in range(customers_to_remove):
        worst_cost, worst_customer, worst_route, worst_index = None, None, None, None

        for route in destroyed.routes:
            for customer in route:
                customer_index = route.index(customer)
                route.remove(customer)
                if len(route) > 0:  # Check if the route is not empty
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
                    if worst_cost is None or cost < worst_cost:
                        worst_cost, worst_customer, worst_route, worst_index = cost, customer, route, customer_index
                route.insert(customer_index, customer)

        if worst_customer is not None:
            worst_route.remove(worst_customer)
            destroyed.unassigned.append(worst_customer)

    destroyed.routes = [route for route in destroyed.routes if len(route) != 0]
    return destroyed


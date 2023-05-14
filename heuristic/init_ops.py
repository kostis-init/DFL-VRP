from enums import SolverMode


def init_routes_nn(solver):
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity limit.
    """
    routes = []
    vrp = solver.vrp
    unvisited = vrp.customers.copy()
    while unvisited:
        # create a new route
        route = []
        # set the current node to the depot
        current_node = vrp.depot
        # while the route has not met the vehicle capacity limit and there are unvisited nodes
        while sum(node.demand for node in route) < vrp.capacity and unvisited:
            # find the nearest unvisited node depending on the mode
            if solver.mode == SolverMode.TRUE_COST:
                nearest = min(unvisited, key=lambda node: vrp.cost(current_node, node))
            elif solver.mode == SolverMode.SPO:
                nearest = min(unvisited,
                              key=lambda node: -vrp.cost(current_node, node) + 2 * vrp.pred_cost(current_node, node))
            elif solver.mode == SolverMode.PRED_COST:
                nearest = min(unvisited, key=lambda node: vrp.pred_cost(current_node, node))
            elif solver.mode == SolverMode.DISTANCE:
                nearest = min(unvisited, key=lambda node: vrp.distance(current_node, node))
            else:
                raise ValueError(f"Unknown solver mode {solver.mode}.")
            if sum(node.demand for node in route) + nearest.demand > vrp.capacity:
                break
            # add the nearest node to the route
            route.append(nearest)
            # remove the nearest node from the unvisited nodes
            unvisited.remove(nearest)
            # update the current node
            current_node = nearest
        # add the route to the solution
        routes.append(route)
    return routes

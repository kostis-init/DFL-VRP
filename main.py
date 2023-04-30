from util import *
import numpy as np
from edge_model import EdgeTrainer


def test(trainer, vrp_instances_test):
    # Iterate over the test set and print the accuracy of the model
    accuracy = 0.0
    actual_sols_cost = 0.0
    predicted_sols_cost = 0.0
    for inst in vrp_instances_test:
        # solve gurobi with actual edge costs
        solver = GurobiSolver(inst)
        solver.solve()
        actual_edges = solver.get_active_arcs()
        # set edge costs to predicted values, but first backup the actual costs
        actual_costs = [edge.cost for edge in inst.edges]
        actual_sols_cost += sum([edge.cost for edge in actual_edges])
        for edge in inst.edges:
            edge.predicted_cost = trainer.predict([edge.features])
        # solve gurobi with predicted edge costs
        solver = GurobiSolver(inst)
        solver.set_predicted_objective()
        solver.solve()
        predicted_edges = solver.get_active_arcs()
        # for predicted sols cost, we have to use the actual edge costs of the predicted edges
        predicted_sols_cost += sum([actual_costs[inst.edges.index(edge)] for edge in predicted_edges])
        # compare actual and predicted edges
        correct_edges = set(actual_edges).intersection(predicted_edges)
        incorrect_edges = set(actual_edges).symmetric_difference(predicted_edges)
        accuracy += float(len(correct_edges)) / float((len(correct_edges) + len(incorrect_edges)))
        print(f'Parsed instance {inst}, accuracy: {accuracy}, actual cost: {actual_sols_cost}, '
              f'predicted cost: {predicted_sols_cost}')
    accuracy /= len(vrp_instances_test)
    cost_comparison = predicted_sols_cost / actual_sols_cost
    print(f'Accuracy: {accuracy}, cost comparison: {cost_comparison}')


def perform_single_test(trainer, vrp_instances_test):
    print(f'Testing example instance {vrp_instances_test[0]}, '
          f'predicted cost: {trainer.predict([vrp_instances_test[0].edges[0].features])}, '
          f'actual cost: {vrp_instances_test[0].edges[0].cost}')
    # solve gurobi with actual edge costs
    test_instance = vrp_instances_test[0]
    solver = GurobiSolver(test_instance)
    solver.solve()
    print('Drawing actual solution')
    draw_solution(solver)
    actual_edges = solver.get_active_arcs()
    # set edge costs to predicted values
    for edge in test_instance.edges:
        edge.cost = trainer.predict([edge.features])
    # solve gurobi with predicted edge costs
    solver = GurobiSolver(test_instance)
    solver.solve()
    print('Drawing predicted solution')
    draw_solution(solver)
    predicted_edges = solver.get_active_arcs()
    # compare actual and predicted edges
    print(f'Actual edges ({len(actual_edges)}): {actual_edges}')
    print(f'Predicted edges ({len(predicted_edges)}): {predicted_edges}')
    correct_edges = set(actual_edges).intersection(predicted_edges)
    incorrect_edges = set(actual_edges).symmetric_difference(predicted_edges)
    print(f'Correct edges ({len(correct_edges)}): {correct_edges}')
    print(f'Incorrect edges ({len(incorrect_edges)}): {incorrect_edges}')


def main():
    vrp_instances_train = [parse_datafile(f'data/non_linear_1000_25/instance_{i}') for i in range(900)]
    vrp_instances_test = [parse_datafile(f'data/non_linear_1000_25/instance_{i}') for i in range(900, 1000)]

    trainer = EdgeTrainer(vrp_instances_train, vrp_instances_test)
    trainer.train()
    perform_single_test(trainer, vrp_instances_test)
    test(trainer, vrp_instances_test)


if __name__ == '__main__':
    main()

from util import *
import numpy as np
# from keras_model import EdgeTrainer
from torch_model import EdgeTrainer


# TODO: Implement the model
# maybe consider graph neural networks
# class DFLModel(tf.keras.Model):
#     """
#     This class implements a Decision Focused Learning model for solving VRPs.
#     More specifically, the input will be VRP instances, and the output will be the
#     optimal routes for each vehicle using the class GurobiSolver. The costs of each
#     edge is unknown and will be learned by the model.
#     """


def main():
    vrp_instances_train = [parse_datafile(f'data/generated_backup/instance_{i}') for i in range(10)]
    vrp_instances_test = [parse_datafile(f'data/generated_backup/instance_{i}') for i in range(10, 20)]

    trainer = EdgeTrainer(vrp_instances_train, vrp_instances_test)
    trainer.train()
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


if __name__ == '__main__':
    main()

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
    vrp_instances_train = [parse_datafile(f'data/generated/instance_{i}') for i in range(10)]
    vrp_instances_test = [parse_datafile(f'data/generated/instance_{i}') for i in range(10, 20)]

    trainer = EdgeTrainer(vrp_instances_train, vrp_instances_test, patience=4)
    trainer.train()
    print('Test prediction: ', trainer.predict([26.83, 26.83, 8.07, 0, 2.52, 11.24, 74.11, 0.57, 0.55]))

    # solve gurobi
    test_instance = vrp_instances_test[0]
    solver = GurobiSolver(test_instance)
    solver.solve()
    print('Drawing actual solution')
    draw_solution(solver)

    # set edge costs to predicted values
    for edge in test_instance.edges:
        edge.cost = trainer.predict([edge.features])

    # solve gurobi
    solver = GurobiSolver(test_instance)
    solver.solve()
    print('Drawing predicted solution')
    draw_solution(solver)






if __name__ == '__main__':
    main()

from util import *
import numpy as np

# TODO: Implement the model
# maybe consider graph neural networks
# class DFLModel(tf.keras.Model):
#     """
#     This class implements a Decision Focused Learning model for solving VRPs.
#     More specifically, the input will be VRP instances, and the output will be the
#     optimal routes for each vehicle using the class GurobiSolver. The costs of each
#     edge is unknown and will be learned by the model.
#     """


from torch_model import EdgeTrainer
# from keras_model import EdgeTrainer


def main():
    vrp_instances_train = [parse_datafile(f'data/generated/instance_{i}') for i in range(90)]
    vrp_instances_test = [parse_datafile(f'data/generated/instance_{i}') for i in range(90, 100)]

    trainer = EdgeTrainer(vrp_instances_train, vrp_instances_test)
    trainer.train()
    test_feat = [26.83, 26.83, 8.07, 0, 2.52, 11.24, 74.11, 0.57, 0.55]
    print('Test prediction, for features: ', trainer.predict(test_feat))


if __name__ == '__main__':
    main()

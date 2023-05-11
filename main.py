from util import *
from edge_model import EdgeTrainer


def main():
    # vrp_instances_train = [parse_datafile(f'data/cvrp_1000_10_5_1/instance_{i}') for i in range(800)]
    # vrp_instances_test = [parse_datafile(f'data/cvrp_1000_10_5_1/instance_{i}') for i in range(800, 1000)]
    #
    # trainer = EdgeTrainer(vrp_instances_train, vrp_instances_test, lr=1e-4)
    # trainer.train()
    # test_and_draw(trainer, vrp_instances_test[0])
    # test(trainer, vrp_instances_test)
    vrp = parse_datafile('data/cvrp_10_25_4_4/instance_0')
    solver = GurobiSolver(vrp)
    solver.solve()
    draw_solution(solver)


if __name__ == '__main__':
    main()

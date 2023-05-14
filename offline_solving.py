from solver import GurobiSolver
from util import parse_datafile
from heuristic.heuristic_solver import HeuristicSolver
from util import draw_solution
from tqdm import tqdm


def main():
    vrp_folders = [f'data/cvrp_10000_200_4_6_0.1/instance_{i}' for i in range(10_000)]
    for vrp_folder in tqdm(vrp_folders):
        vrp = parse_datafile(vrp_folder)
        solver_h = HeuristicSolver(vrp, time_limit=15)
        solver_h.solve()
        # draw_solution(solver_h)
        # save solution to a separate file in the same folder
        with open(f'{vrp_folder}/solution.txt', 'w') as f:
            f.write(f'{solver_h.get_routes()}\n')
            f.write(f'{solver_h.get_actual_objective()}\n')


if __name__ == '__main__':
    main()

import random
from datetime import datetime

from dfl_vrp.heuristic.heuristic_solver import HeuristicSolver
from dfl_vrp.util import *
from dfl_vrp.nce_model import NCEModel
from tqdm import tqdm
from dfl_vrp.two_stage_model import TwoStageModel
from dfl_vrp.spo_model import SPOModel
from dfl_vrp.nce_model import NCETrueCostLoss
import gurobipy as gp

# Constants
TRAIN_PERC, VALIDATION_PERC, TEST_PERC = 0.8, 0.1, 0.1
NUM_DATA = 1000
DATA_PATH = '../data/capacity100/instances1000/nodes120/noise0.1/feat4'
SOLVER_CLASS = HeuristicSolver

NUM_EPOCHS_2_STAGE = 20
NUM_EPOCHS_SPO = 20
NUM_EPOCHS_NCE = 20
NUM_EPOCHS_NCE_NO_TRUE_COSTS = 20

LR_2_STAGE = 5e-4
LR_SPO = 1e-4
LR_NCE = 1e-4
LR_NCE_NO_TRUE_COSTS = 1e-4

WEIGHT_DECAY = 1e-4
SOLVE_PROB = 1

two_stage_enabled = True
# two_stage_enabled = False
spo_enabled = True
# spo_enabled = False
nce_enabled = True
# nce_enabled = False
nce_no_true_costs_enabled = True
# nce_no_true_costs_enabled = False

# Init gurobi
gp.Model()


def after_train(model, train_time):
    test_time, results = timeit(test)(model.cost_model, vrps_test, is_two_stage=False)

    print('Testing first TEST instance...')
    test_single(model.cost_model, vrps_test[0])
    print('Testing first TRAIN instance...')
    test_single(model.cost_model, vrps_train[0])

    return tuple(list(results) + [train_time, test_time])


# Load data
print('\nLoading data...')
data = [parse_datafile(f'{DATA_PATH}/instance_{i}') for i in tqdm(range(NUM_DATA))]
random.shuffle(data)
num_train, num_val, num_test = int(len(data) * TRAIN_PERC), int(len(data) * VALIDATION_PERC), int(len(data) * TEST_PERC)
vrps_train, vrps_val, vrps_test = data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]
print(f'Number of training data: {len(vrps_train)}, validation data: {len(vrps_val)}, test data: {len(vrps_test)}')

# 2-stage model
if two_stage_enabled:
    print('\nTraining 2-stage model...')
    model = TwoStageModel(vrps_train, vrps_val, vrps_test, lr=LR_2_STAGE, weight_decay=WEIGHT_DECAY)
    train_time, _ = timeit(model.train)(num_epochs=NUM_EPOCHS_2_STAGE)
    print('Testing 2-stage model...')
    model.test()
    test_time, results = timeit(test)(model, vrps_test, is_two_stage=True)
    two_stage_results = tuple(list(results) + [train_time, test_time])
    print('Testing first test instance...')
    test_single(model.model, vrps_test[0])
    print('Testing first train instance...')
    test_single(model.model, vrps_train[0])
else:
    two_stage_results = None
    print('2-stage model not enabled')

# SPO model
if spo_enabled:
    print('\nTraining SPO model...')
    model = SPOModel(vrps_train, vrps_val, vrps_test, solver_class=SOLVER_CLASS,
                     lr=LR_SPO, weight_decay=WEIGHT_DECAY, solve_prob=SOLVE_PROB)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_SPO, verbose=False, test_every=1)
    print('Testing SPO model...')
    spo_results = after_train(model, train_time)
else:
    spo_results = None
    print('SPO model not enabled')

# NCE model
if nce_enabled:
    print('\nTraining NCE model...')
    model = NCEModel(vrps_train, vrps_val, vrps_test, solver_class=SOLVER_CLASS, lr=LR_NCE,
                     include_true_costs=True, weight_decay=WEIGHT_DECAY, solve_prob=SOLVE_PROB)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE, verbose=False, test_every=1)
    print('Testing NCE model...')
    nce_results = after_train(model, train_time)
else:
    nce_results = None
    print('NCE model not enabled')

# NCE model without true costs
if nce_no_true_costs_enabled:
    print('\nTraining NCE model without true costs...')
    model = NCEModel(vrps_train, vrps_val, vrps_test, solver_class=SOLVER_CLASS, lr=LR_NCE_NO_TRUE_COSTS,
                     include_true_costs=False, weight_decay=WEIGHT_DECAY, solve_prob=SOLVE_PROB)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE_NO_TRUE_COSTS, verbose=False, test_every=1)
    print('Testing NCE model without true costs...')
    nce_no_true_costs_results = after_train(model, train_time)
else:
    nce_no_true_costs_results = None
    print('NCE model without true costs not enabled')

# Save results
results = {
    'two_stage': two_stage_results,
    'spo': spo_results,
    'nce': nce_results,
    'nce_no_true_costs': nce_no_true_costs_results
}
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_PATH = f'../results/{DATA_PATH}_{current_time}/'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
with open(f'{RESULTS_PATH}{NUM_DATA}.txt', 'wb') as f:
    for model_name, result in results.items():
        f.write(f'{model_name}: {result}\n'.encode('utf-8'))

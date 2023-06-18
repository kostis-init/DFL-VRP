import pickle
from datetime import datetime

from util import *
from nce_model import NCEModel
from tqdm import tqdm
from two_stage_model import TwoStageModel
from spo_model import SPOModel, SPOModelNoTrueCosts
from nce_model import NCETrueCostLoss
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np

# Constants
TRAIN_PERC, VALIDATION_PERC, TEST_PERC = 0.75, 0.05, 0.20
NUM_DATA = 10_000
DATA_PREFIX = 'data/scaled/'
DATA_PATH = 'cvrp_10000_100_5'
SOLVER_CLASS = GurobiSolver
NUM_EPOCHS_2_STAGE = 10
NUM_EPOCHS_SPO = 20
NUM_EPOCHS_NCE = 20
NUM_EPOCHS_SPO_NO_TRUE_COSTS = 20
NUM_EPOCHS_NCE_NO_TRUE_COSTS = 20

# Init gurobi
gp.Model()

# Load data
print('\nLoading data...')
data = [parse_datafile(f'{DATA_PREFIX}{DATA_PATH}/instance_{i}') for i in tqdm(range(NUM_DATA))]
num_train, num_val, num_test = int(len(data) * TRAIN_PERC), int(len(data) * VALIDATION_PERC), int(len(data) * TEST_PERC)
vrps_train, vrps_val, vrps_test = data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]
print(f'Number of training data: {len(vrps_train)}, validation data: {len(vrps_val)}, test data: {len(vrps_test)}')

# 2-stage model
print('\nTraining 2-stage model...')
model = TwoStageModel(vrps_train, vrps_val, vrps_test, lr=1e-2)
train_time, _ = timeit(model.train)(num_epochs=NUM_EPOCHS_2_STAGE)
print('Testing 2-stage model...')
model.test()
test_time, results = timeit(test)(model, vrps_test, SOLVER_CLASS)
two_stage_results = tuple(list(results) + [train_time, test_time])
print('Testing single instance...')
test_single(model.model, vrps_test[0], SOLVER_CLASS)

# SPO model
print('\nTraining SPO model...')
model = SPOModel(vrps_train, vrps_val, vrps_test, solver_class=SOLVER_CLASS, lr=1e-2)
train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_SPO, verbose=False, test_every=5)
print('Testing SPO model...')
test_time, results = timeit(test)(model.cost_model, vrps_test, SOLVER_CLASS, is_two_stage=False)
spo_results = tuple(list(results) + [train_time, test_time])
print('Testing single instance...')
test_single(model.cost_model, vrps_test[0], SOLVER_CLASS)

# SPO model without true costs
print('\nTraining SPO model without true costs...')
model = SPOModelNoTrueCosts(vrps_train, vrps_val, vrps_test, solver_class=GurobiSolver, lr=1e-2)
train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_SPO_NO_TRUE_COSTS, verbose=False, test_every=4)
print('Testing SPO model without true costs...')
test_time, results = timeit(test)(model.cost_model, vrps_test, SOLVER_CLASS, is_two_stage=False)
spo_no_true_costs_results = tuple(list(results) + [train_time, test_time])
print('Testing single instance...')
test_single(model.cost_model, vrps_test[0], SOLVER_CLASS)

# NCE model
print('\nTraining NCE model...')
model = NCEModel(vrps_train, vrps_val, vrps_test, solver_class=GurobiSolver, lr=1e-2, solve_prob=0.7)
model.criterion = NCETrueCostLoss({vrp: [vrp.actual_solution] for vrp in vrps_train})
train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE, verbose=False, test_every=5)
print('Testing NCE model...')
test_time, results = timeit(test)(model.cost_model, vrps_test, SOLVER_CLASS, is_two_stage=False)
nce_results = tuple(list(results) + [train_time, test_time])
print('Testing single instance...')
test_single(model.cost_model, vrps_test[0], SOLVER_CLASS)

# NCE model without true costs
print('\nTraining NCE model without true costs...')
model = NCEModel(vrps_train, vrps_val, vrps_test, solver_class=GurobiSolver, lr=1e-2, solve_prob=0.7)
train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE_NO_TRUE_COSTS, verbose=False, test_every=5)
print('Testing NCE model without true costs...')
test_time, results = timeit(test)(model.cost_model, vrps_test, SOLVER_CLASS, is_two_stage=False)
nce_no_true_costs_results = tuple(list(results) + [train_time, test_time])
print('Testing single instance...')
test_single(model.cost_model, vrps_test[0], SOLVER_CLASS)

# Save results
results = {
    'two_stage': two_stage_results,
    'spo': spo_results,
    'nce': nce_results,
    'spo_no_true_costs': spo_no_true_costs_results,
    'nce_no_true_costs': nce_no_true_costs_results
}
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_PATH = f'results/{DATA_PATH}_{current_time}/'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
with open(f'{RESULTS_PATH}{NUM_DATA}.txt', 'wb') as f:
    for model_name, result in results.items():
        f.write(f'{model_name}: {result}\n'.encode('utf-8'))



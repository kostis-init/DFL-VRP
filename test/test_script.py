import random
from datetime import datetime
from dfl_vrp.util import *
from tqdm import tqdm
from dfl_vrp.two_stage_model import TwoStageModel
from dfl_vrp.two_stage_non_linear_model import TwoStageNonLinearModel
from dfl_vrp.spo_model import SPOModelLinear, SPOModelEncoderDecoder
from dfl_vrp.nce_model import NCEModelLinear, NCEModelEncoderDecoder

# Constants
TRAIN_PERC, VALIDATION_PERC, TEST_PERC = 0.8, 0.1, 0.1
NUM_DATA = 1000
DATA_PATH = '../data/capacity100/instances1000/nodes40/noise0.1/feat4'
SOLVER_CLASS = HeuristicSolver

NUM_EPOCHS_2_STAGE, NUM_EPOCHS_SPO, NUM_EPOCHS_NCE = 50, 50, 50
LR_2_STAGE, LR_SPO, LR_NCE = 5e-4, 2e-4, 2e-4
WEIGHT_DECAY = 1e-4
SOLVE_PROB = 0.25

RESULTS_PATH = f'{DATA_PATH}_{SOLVER_CLASS.__name__}_0.2sec_solveprob{SOLVE_PROB}/'

two_stage_enabled, two_stage_nl_enabled, spo_enabled, spo_nl_enabled, nce_enabled, nce_nl_enabled =\
    True, False, True, False, True, False


def after_train(cost_model, time_elapsed, is_two_stage=False):
    _, res = timeit(test)(cost_model, vrps_test, is_two_stage=is_two_stage)
    print('Testing first TEST instance...')
    test_single(cost_model, vrps_test[0], is_two_stage=is_two_stage)
    print('Testing first TRAIN instance...')
    test_single(cost_model, vrps_train[0], is_two_stage=is_two_stage)
    return tuple(list(res) + [time_elapsed])


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
    two_stage_results = after_train(model, train_time, is_two_stage=True)
else:
    two_stage_results = None
    print('2-stage model not enabled')

# 2-stage DL model
if two_stage_nl_enabled:
    print('\nTraining 2-stage Non Linear model...')
    model = TwoStageNonLinearModel(vrps_train, vrps_val, vrps_test, lr=LR_2_STAGE, weight_decay=WEIGHT_DECAY)
    train_time, _ = timeit(model.train)(num_epochs=NUM_EPOCHS_2_STAGE)
    print('Testing 2-stage Non Linear model...')
    two_stage_non_linear_results = after_train(model, train_time, is_two_stage=True)
else:
    two_stage_non_linear_results = None
    print('2-stage Non Linear model not enabled')

# SPO model
if spo_enabled:
    print('\nTraining SPO model...')
    model = SPOModelLinear(vrps_train, vrps_val, vrps_test, SOLVER_CLASS,
                           lr=LR_SPO, weight_decay=WEIGHT_DECAY, solve_prob=SOLVE_PROB)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_SPO, verbose=False, test_every=1000)
    print('Testing SPO model...')
    spo_results = after_train(model.cost_model, train_time)
else:
    spo_results = None
    print('SPO model not enabled')

# SPO Non linear model
if spo_nl_enabled:
    print('\nTraining SPO Non linear model...')
    model = SPOModelEncoderDecoder(vrps_train, vrps_val, vrps_test, SOLVER_CLASS,
                                   lr=LR_SPO, weight_decay=WEIGHT_DECAY, solve_prob=SOLVE_PROB, hidden_size=256)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_SPO, verbose=False, test_every=1000)
    print('Testing SPO Non Linear model...')
    spo_nl_results = after_train(model.cost_model, train_time)
else:
    spo_nl_results = None
    print('SPO Non linear model not enabled')

# NCE model
if nce_enabled:
    print('\nTraining NCE model...')
    model = NCEModelLinear(vrps_train, vrps_val, vrps_test, SOLVER_CLASS, lr=LR_NCE, weight_decay=WEIGHT_DECAY,
                           solve_prob=SOLVE_PROB)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE, verbose=False, test_every=1000)
    print('Testing NCE model...')
    nce_results = after_train(model.cost_model, train_time)
else:
    nce_results = None
    print('NCE model not enabled')

# NCE Non linear model
if nce_nl_enabled:
    print('\nTraining NCE Non linear model...')
    model = NCEModelEncoderDecoder(vrps_train, vrps_val, vrps_test, SOLVER_CLASS, lr=LR_NCE, weight_decay=WEIGHT_DECAY,
                                   solve_prob=SOLVE_PROB, hidden_size=256)
    train_time, _ = timeit(model.train)(epochs=NUM_EPOCHS_NCE, verbose=False, test_every=1000)
    print('Testing NCE Non linear model...')
    nce_nl_results = after_train(model.cost_model, train_time)
else:
    nce_nl_results = None
    print('NCE Non linear model not enabled')

# Save results
results = {
    'two_stage': two_stage_results,
    'two_stage_non_linear': two_stage_non_linear_results,
    'spo': spo_results,
    'spo_nl': spo_nl_results,
    'nce': nce_results,
    'nce_nl': nce_nl_results
}

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
with open(f'{RESULTS_PATH}res.txt', 'wb') as f:
    for model_name, result in results.items():
        f.write(f'{model_name}: {result}\n'.encode('utf-8'))

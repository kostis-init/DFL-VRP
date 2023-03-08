import random
import numpy as np
import pandas as pd

num_instances = 100
num_nodes = 50

coord_range = (0, 100)
demand_range = (1, 10)
ready_time_range = (0, 20)
due_date_range = (30, 50)
service_time_range = (0, 5)

for i in range(num_instances):
    df = pd.DataFrame()
    df['customer'] = range(num_nodes)
    df['xcord'] = np.random.randint(*coord_range, num_nodes)
    df['ycord'] = np.random.randint(*coord_range, num_nodes)
    df['demand'] = np.random.randint(*demand_range, num_nodes)
    df['ready_time'] = np.random.randint(*ready_time_range, num_nodes)
    df['due_date'] = np.random.randint(*due_date_range, num_nodes)
    df['service_time'] = np.random.randint(*service_time_range, num_nodes)


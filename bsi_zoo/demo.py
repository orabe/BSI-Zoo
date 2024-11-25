from joblib import Memory
from pathlib import Path
import numpy as np
import pandas as pd
import time

from bsi_zoo.benchmark import Benchmark
from bsi_zoo.estimators import Solver, gamma_map
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, reconstructed_noise
from bsi_zoo.config import get_leadfield_path
from bsi_zoo.data_generator import get_data
from un_ca_functions import *

from sklearn.utils import check_random_state
from scipy import linalg
import yaml
import matplotlib.pyplot as plt

# Load configuration parameters from the config file
config_path = "bsi_zoo/config_un_ca.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

n_jobs = config['n_jobs']
nruns = config['nruns']
memory_path = config['memory_path']
random_state = config['random_state']

data_path = Path(config['data_path'])
spatial_cv = config['spatial_cv']
subject = config['subject']

orientation_type = config['data_args_II']['orientation_type']
cov_type = config['data_args_II']['cov_type']
alpha_SNR = config['data_args_II']['alpha_SNR']
nnzs = config['data_args_II']['nnz']
n_times = config['data_args_II']['n_times']


if config['estimator'] == 'gamma_map':
    estimator = gamma_map
    
estimator_args = config['estimator_args']
estimator_extra_params = config['estimator_extra_params']

# Load gamma initialization parameters
gamma_init = config['estimator_extra_params']['gammas']
path_to_leadfield = get_leadfield_path(subject, type=orientation_type)
lead_field = np.load(path_to_leadfield, allow_pickle=True)
L = lead_field["lead_field"]
        
# Check which gamma initialization method is enabled
if gamma_init['fixed']['enabled']:
    estimator_extra_params['gammas'] = np.full((L.shape[1],), gamma_init['fixed']['value'], dtype=np.float64)
elif gamma_init['random_uniform']['enabled']:
    rng = check_random_state(random_state)
    estimator_extra_params['gammas'] = rng.uniform(
        low=gamma_init['random_uniform']['low'],
        high=gamma_init['random_uniform']['high'],
        size=(L.shape[1],)
    )
elif gamma_init['ones']['enabled']:
    estimator_extra_params['gammas'] = None
    
estimator_extra_params['tol'] = float(estimator_extra_params['tol'])

# Memory and job settings
memory = Memory(memory_path)

# Setup subject and data path
# subject = "fsaverage"
# data_path = Path("bsi_zoo/tests/data/")


# Data arguments
data_args_II = {
    "n_times": n_times,
    "nnz": nnzs,
    "cov_type": cov_type, # or 'diag'
    "path_to_leadfield": path_to_leadfield,
    "orientation_type": orientation_type,
    "alpha": alpha_SNR, 
}

# estimator = gamma_map
# estimator_args = {"alpha": estimator_alphas}
# estimator_extra_params = {"update_mode": 2, "max_iter":1000, "tol":1e-15}

# Metrics for benchmarking
metrics = [
    # euclidean_distance,
    # mse,
    emd, # we will mainly use this
    # f1,
    # reconstructed_noise,
]

# Memory and job settings. Not really used as we will consider one experiment at once
# memory = Memory(".")
# n_jobs = 30
# nruns = 1

results = dict(estimator=estimator.__name__) 

x, x_hat, y, L, cov, active_set, posterior_cov, results = run(
    estimator=estimator,
    subject=subject,
    metrics=metrics,
    data_args=data_args_II,
    estimator_args=estimator_args,
    random_state=42,
    memory=memory,
    n_jobs=n_jobs,
    do_spatial_cv=spatial_cv, # False
    estimator_extra_params=estimator_extra_params,
    results=results,
    nruns=nruns
)

# results = eval(
#     x=x,
#     x_hat=x_hat,
#     y=y,
#     L=L,
#     cov=cov,
#     subject=subject,
#     data_args = data_args_II,
#     estimator_args=estimator_args,
#     estimator_extra_params=estimator_extra_params,
#     metrics=metrics,
#     results=results
# )

# save the results
results = {key: [value] if not isinstance(value, (list, pd.Series)) else value for key, value in results.items()}
results = pd.DataFrame(results)
# print(results)

# filename = f"benchmark_data_{subject}_{data_args_II['orientation_type'][0]}_{time.strftime('%b-%d-%Y_%H%M', time.localtime())}.pkl"
# results.to_pickle(data_path/filename)

# export data
np.savez_compressed('bsi_zoo/data/est_data.npz', x=x, x_hat=x_hat, y=y, L=L, cov=cov, active_set=active_set, posterior_cov=posterior_cov, data_args_II=data_args_II, estimator_args=estimator_args, estimator_extra_params=estimator_extra_params, allowpickle=True)

print("------------------ Done! ------------------")
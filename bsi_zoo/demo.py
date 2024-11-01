from joblib import Memory
from pathlib import Path
import numpy as np
import pandas as pd
import time

from bsi_zoo.benchmark import Benchmark
from bsi_zoo.estimators import gamma_map
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, reconstructed_noise
from bsi_zoo.config import get_leadfield_path

n_jobs = 30
# nruns = 10
nruns = 1
spatial_cv = [False]
subject = "testMoh"
metrics = [
    euclidean_distance,
    mse,
    emd,
    f1,
    reconstructed_noise,
]  # list of metric functions here
nnzs = [5] # number non-zero sources. Used to randomly select a subset of n_sources indices
# nnzs = [1]
alpha_SNR = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
# alpha_SNR = [0.5]
# estimator_alphas = [
#     0.01,
#     0.01544452,
#     0.02385332,
#     0.03684031,
#     0.0568981,
#     0.08787639,
#     0.13572088,
#     0.2096144,
# ]  # logspaced
estimator_alphas = np.logspace(0, -2, 20)[1:]
# estimator_alphas = [0.5]
memory = Memory(".")

path_to_leadfield = 'bsi_zoo/meta/leadfield_fixed.npz'
orientation_type = "fixed"
data_args_II = {
    # "n_sensors": [50],
    # "n_times": [10],
    "n_times": [2],
    # "n_sources": [200],
    "nnz": nnzs,
    "cov_type": ["full"],
    "path_to_leadfield": [path_to_leadfield],
    "orientation_type": [orientation_type],
    "alpha": alpha_SNR,  # this is actually SNR
}
df_results = []

benchmark = Benchmark(
    estimator=gamma_map,
    subject=subject,
    metrics=metrics,
    data_args=data_args_II,
    estimator_args={"alpha": estimator_alphas},
    random_state=42,
    memory=memory,
    n_jobs=n_jobs,
    do_spatial_cv=spatial_cv[0], # False
    estimator_extra_params={"update_mode": 2},
)
results = benchmark.run(nruns=nruns)

df_results.append(results)
df_results = pd.concat(df_results, axis=0)

data_path = Path("bsi_zoo/data/testMoh")
data_path.mkdir(exist_ok=True)
FILE_NAME = f"testMoh_benchmark_data_{subject}_{data_args_II['orientation_type'][0]}_{time.strftime('%b-%d-%Y_%H%M', time.localtime())}.pkl"
df_results.to_pickle(data_path / FILE_NAME)

print(df_results)

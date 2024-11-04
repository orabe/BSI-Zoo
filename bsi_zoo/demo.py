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

from sklearn.utils import check_random_state
from scipy import linalg


def run(
    estimator,
    subject,
    metrics,
    data_args,
    estimator_args,
    random_state,
    memory,
    n_jobs,
    do_spatial_cv,
    estimator_extra_params,
    results):
        
    rng = check_random_state(random_state)       
    seeds = rng.randint(low=0, high=2 ** 32, size=nruns)
    
    try:
        y, L, x, cov, _ = get_data(**data_args_II, seed=seeds)
        n_orient = 3 if data_args["orientation_type"] == "free" else 1
        
        if data_args["cov_type"] == "diag":
            whitener = linalg.inv(linalg.sqrtm(cov))
            L = whitener @ L
            y = whitener @ y
        
        estimator_ = Solver(
            estimator,
            alpha=estimator_args["alpha"],
            cov_type=data_args["cov_type"],
            cov=cov,
            n_orient=n_orient,
            extra_params=estimator_extra_params,
        ).fit(L=L, y=y)

        x_hat, posterior_cov = estimator_.predict(y)
            
        if data_args["orientation_type"] == "free":
            x_hat = x_hat.reshape(x.shape)
    
    except Exception as e:
        results["error"] = e
        return None, None, None, None, None, results
    
    return x, x_hat, y, L, cov, results
    
def eval(
    x,
    x_hat,
    y,
    L,
    cov,
    subject,
    data_args,
    estimator_args,
    estimator_extra_params,
    metrics,
    results):
    
    for metric in metrics:
        try:
            metric_score = metric(
                x,
                x_hat,
                subject=subject,
                orientation_type=data_args["orientation_type"],
                nnz=data_args["nnz"],
                y=y,
                L=L,
                cov=cov,
            )
        except Exception:
            # estimators that predict less vertices for certain parameter combinations; these cannot be evaluated by all current metrics
            metric_score = np.nan
        results[metric.__name__] = metric_score
        
    results.update(data_args)
    results.update({"extra_params": estimator_extra_params})
    results.update({f"estimator__{k}": v for k, v in estimator_args.items()})
    return results

# Configuration parameters
orientation_type = "fixed"
nnzs = 5 # number non-zero sources. Used to select a subset of n_sources indices
estimator_alphas = 0.02 # sigma^2 that is used in capital_sigma_y
alpha_SNR = 0.5
cov_type = "full" # or 'diag'
spatial_cv = [False] # or True. Should corresponnd to["temporal", "spatial"]?

# Setup subject and data path
subject = "fsaverage"
data_path = Path("bsi_zoo/tests/data/")
path_to_leadfield = get_leadfield_path(subject, type=orientation_type)


# Data arguments
data_args_II = {
    "n_times": 2,
    "nnz": nnzs,
    "cov_type": cov_type, # or 'diag'
    "path_to_leadfield": path_to_leadfield,
    "orientation_type": orientation_type,
    "alpha": alpha_SNR, 
}

estimator = gamma_map
estimator_args = {"alpha": estimator_alphas}
estimator_extra_params = {"update_mode": 2}

# Metrics for benchmarking
metrics = [
    euclidean_distance,
    mse,
    emd, # we will mainly use this
    f1,
    reconstructed_noise,
]

# Memory and job settings. Not really used as we will consider one experiment at once
memory = Memory(".")
n_jobs = 30
nruns = 1

results = dict(estimator=estimator.__name__) 

x, x_hat, y, L, cov, results = run(
    estimator=estimator,
    subject=subject,
    metrics=metrics,
    data_args=data_args_II,
    estimator_args=estimator_args,
    random_state=42,
    memory=memory,
    n_jobs=n_jobs,
    do_spatial_cv=spatial_cv[0], # False
    estimator_extra_params=estimator_extra_params,
    results=results
)

results = eval(
    x=x,
    x_hat=x_hat,
    y=y,
    L=L,
    cov=cov,
    subject=subject,
    data_args = data_args_II,
    estimator_args=estimator_args,
    estimator_extra_params=estimator_extra_params,
    metrics=metrics,
    results=results
)

# save the results
results = {key: [value] if not isinstance(value, (list, pd.Series)) else value for key, value in results.items()}
results = pd.DataFrame(results)
print(results)

filename = f"benchmark_data_{subject}_{data_args_II['orientation_type'][0]}_{time.strftime('%b-%d-%Y_%H%M', time.localtime())}.pkl"
results.to_pickle(data_path/filename)


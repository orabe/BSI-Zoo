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
import matplotlib.pyplot as plt

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

        x_hat, active_set, posterior_cov = estimator_.predict(y)
            
        if data_args["orientation_type"] == "free":
            x_hat = x_hat.reshape(x.shape)
    
    except Exception as e:
        results["error"] = e
        return None, None, None, None, None, results
    
    return x, x_hat, y, L, cov, active_set, posterior_cov, results
    
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
spatial_cv = [False] # or True. Cross valudation. Should corresponnd to["temporal", "spatial"]?

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

x, x_hat, y, L, cov, active_set, posterior_cov, results = run(
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



# -------
"""
# CL
Interpreting the Results
	1.	Understanding the Confidence Interval:
	•	For each source at each time point, the confidence interval provides a range within which we expect the true parameter (source activity) to lie with a certain probability (e.g., 95%).
	•	If the CIs are narrow, it indicates that the estimates are precise, whereas wide CIs suggest more uncertainty in the estimates.
	•	If the CI does not include zero (for mean estimates centered around zero), it may indicate statistical significance at that time point.
 
	2.	Comparing Sources:
	•	By looking at the CIs of different sources, you can assess which sources show significant activity compared to others.
	•	If two sources have overlapping CIs, their activities may not be significantly different.
 
"""
posterior_mean = x_hat[active_set]



posterior_mean = posterior_mean[0:10]
posterior_cov = posterior_cov[0:10, 0:10]

def check_symmetric(a, rtol=1e-03, atol=1e-03):
    print(np.allclose(a, a.T, rtol=rtol, atol=atol))
check_symmetric(posterior_cov)

# Ensure positive definiteness
def is_pos_def(x):
    if not np.all(np.linalg.eigvals(x) > 0):
        regularization_strength = 1e-6  # Adjust as necessary
        posterior_cov += np.eye(posterior_cov.shape[0]) * regularization_strength
        return False
    else:
        return True
        
print(is_pos_def(posterior_cov))

# Confidence level
confidence_level = 0.95
z = 1.96  # z-score for 95% confidence

# Calculate standard deviations from the covariance matrix
std_dev = np.sqrt(np.diag(posterior_cov))

# Compute confidence intervals
ci_lower = posterior_mean - z * std_dev[:, np.newaxis]  # Expand dims to align with mean shape
ci_upper = posterior_mean + z * std_dev[:, np.newaxis]

# ci_lower and ci_upper now have shape (n_sources, n_times)
print("Confidence Interval Lower Bounds:\n", ci_lower)
print("Confidence Interval Upper Bounds:\n", ci_upper)

# Visualization
n_times = posterior_mean.shape[1] 
n_sources_hat = posterior_mean.shape[0]


# Set up the figure for subplots
fig, axes = plt.subplots(1, n_times, figsize=(12, 6), sharey=True)

# Create subplots for each time point
for t in range(n_times):
    for i in range(n_sources_hat):  # Loop over each source
        # Plot the mean
        axes[t].plot(i, posterior_mean[i, t], marker='o', label=f'Source {i+1}')  
        # Fill between for confidence intervals
        axes[t].fill_betweenx(
            [ci_lower[i, t], ci_upper[i, t]],  # y range
            i - 0.1,  # x start (left side)
            i + 0.1,  # x end (right side)
            alpha=0.3
        )  # Fill between CI bounds
    
    # Customizing the subplot
    axes[t].set_title(f'Time Point {t + 1}')
    axes[t].set_xticks(np.arange(n_sources_hat))  # Set x ticks for each source index
    axes[t].set_xticklabels([f'Source {i+1}' for i in range(n_sources_hat)], rotation=45)  # Label each tick
    axes[t].axhline(0, color='grey', lw=0.8, ls='--')  # Reference line at y=0
    axes[t].grid()

# Overall figure customization
fig.suptitle('Estimated Source Activity with Confidence Intervals')
fig.text(0.5, 0.04, 'Sources', ha='center')
fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust layout
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Move legend outside
plt.show()





# --- CI per time


time_points = np.linspace(0, 1, n_times)  # Continuous time points from 0 to 1

# Determine the number of rows and columns for the grid layout
n_cols = 5  # Number of columns for the grid
n_rows = (n_sources_hat + n_cols - 1) // n_cols  # Calculate number of rows

# Set up the figure for subplots in a grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True, sharey=False)
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Create subplots for each source
for i in range(n_sources_hat):  # Loop over each source
    # Plot the mean and confidence intervals for each time point
    axes[i].plot(time_points, posterior_mean[i, :], marker='o', label='Posterior Mean')  
    axes[i].fill_between(
        time_points,
        ci_lower[i, :],
        ci_upper[i, :],
        alpha=0.3,
        label='Confidence Interval'
    )  # Fill between CI bounds

    # Customizing the subplot
    axes[i].set_title(f'Source {i + 1}')
    axes[i].axhline(0, color='grey', lw=0.8, ls='--')  # Reference line at y=0
    axes[i].grid()
    axes[i].set_xlabel('Time')  # X-axis label for each subplot
    axes[i].set_ylabel('Estimated Activity')  # Y-axis label for each subplot

# Hide any unused subplots
for j in range(n_sources_hat, n_rows * n_cols):
    fig.delaxes(axes[j])

# Overall figure customization
fig.suptitle('Estimated Source Activity with Confidence Intervals')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust layout
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Move legend outside
plt.show()

# print(posterior_mean)
# print(posterior_cov)

# 000===========

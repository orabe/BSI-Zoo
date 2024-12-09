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
from itertools import product
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage, sample
import multiprocessing


# load src space

data_path = sample.data_path()
subject = "fsaverage"
subjects_dir = data_path / "subjects"
sample_dir = data_path / "MEG" / subject

# rng = np.random.RandomState(0)
sfreq = 150  # Sampling frequency in Hz
duration = 2 # 0.5  # Duration in seconds
tstep = 1.0 / sfreq  # Time step between samples
times = np.arange(0, duration, tstep)
n_times = len(times) # = int(sfreq * duration)  # Total number of time points
ch_type = 'eeg'

montage = mne.channels.make_standard_montage("easycap-M43")
info = mne.create_info(montage.ch_names, sfreq=sfreq, ch_types=ch_type)
info.set_montage(montage);

src = mne.read_source_spaces('bsi_zoo/tests/data/my_ico4-src.fif')
print('Number of sources:', src[0]['nuse'], src[1]['nuse'])
print('Location of the first source point (left hemisphere):', src[0]['rr'][0])
print('Orientation of the first source point (left hemisphere):', src[0]['nn'][0])
type(src)

# Extract vertices from the source space
vertices = [s['vertno'] for s in src]  # Left and right hemisphere vertices

n_jobs = 30
nruns = 5
random_state = 42
memory_path = "."
memory = Memory(memory_path)

my_data_path = "bsi_zoo/tests/data/"
spatial_cvs = [False]
subject = "fsaverage"

# Load gamma initialization parameters
# gamma_init = config['estimator_extra_params']['gammas']
path_to_leadfield = get_leadfield_path(subject, type='fixed')
lead_field = np.load(path_to_leadfield, allow_pickle=True)
L = lead_field["lead_field"]
        

# Metrics for benchmarking
metrics = [
    emd, # we will mainly use this
    # euclidean_distance,
    # mse,
    # f1,
    # reconstructed_noise,
]

"""
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
# results = {key: [value] if not isinstance(value, (list, pd.Series)) else value for key, value in results.items()}
# results = pd.DataFrame(results)
# print(results)

# filename = f"benchmark_data_{subject}_{data_args_II['orientation_type'][0]}_{time.strftime('%b-%d-%Y_%H%M', time.localtime())}.pkl"
# results.to_pickle(data_path/filename)

# export data
# np.savez_compressed('bsi_zoo/data/est_data.npz', x=x, x_hat=x_hat, y=y, L=L, cov=cov, active_set=active_set, posterior_cov=posterior_cov, data_args_II=data_args_II, estimator_args=estimator_args, estimator_extra_params=estimator_extra_params, allowpickle=True)
"""

print("------------------ Done! ------------------")

# def check_symmetric(a, rtol=1e-03, atol=1e-03):
#     print(np.allclose(a, a.T, rtol=rtol, atol=atol))
# check_symmetric(posterior_cov)

# # Ensure positive definiteness
# def is_pos_def(x):
#     if not np.all(np.linalg.eigvals(x) > 0):
#         regularization_strength = 1e-6  # Adjust as necessary
#         x += np.eye(x.shape[0]) * regularization_strength
#         return False
#     else:
#         return True


# --------------------------------------------------------------------------------

# plot_active_sources(x_hat, active_set, 'Posterior Mean Over Time')
# plot_active_sources(x, np.where(x[:, 0] != 0)[0], 'Ground Truth Source Activity Over Time')


# --------------------------------------------------------------------------------
base_dir = "experiment_results"


estimators = [gamma_map]
orientation_types = ["fixed"]
cov_types = ["full"]
alpha_snrs = [0.01, 0.1, 0.4, 0.6, 0.8, 0.99]
nnzs = [1, 2, 5]
n_times = 2
estimator_alphas = [0.01, 0.1, 0.4, 0.6, 0.8, 0.99] # np.logspace(0, -2, 10)[1:]
estimator_args = {"alpha": estimator_alphas}
estimator_extra_params = {
    "update_mode": 2,
    "max_iter": 1000,
    "tol": 1e-15,
    "gammas": {
        "fixed": {
            "value": 0.001
        },
        "random_uniform": {
            "low": 0.001,
            "high": 0.1
        },
        "ones": {
            "value": 1.0
        }
    }
}

orientations = ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal']


estimator_extra_params_copy = estimator_extra_params.copy()

for estimator in estimators:
    for spatial_cv in spatial_cvs:
        for orientation_type in orientation_types:
            for nnz in nnzs:
                for cov_type in cov_types:
                    for sigma_squared in estimator_args["alpha"]:
                        for alpha_snr in alpha_snrs:
                            
                            # print(f"nnz: {nnz}, alpha: {sigma_squared}, alpha_snr: {alpha_snr}")
                            
                            for gamma_init_method, value in estimator_extra_params_copy['gammas'].items():
                                if gamma_init_method == "fixed":
                                    estimator_extra_params['gammas'] = np.full((L.shape[1],), value['value'], dtype=np.float64)
                                    
                                elif gamma_init_method == "random_uniform":
                                    rng = check_random_state(random_state)
                                    estimator_extra_params['gammas'] = rng.uniform(low=value['low'], high=value['high'], size=(L.shape[1],))
                                    
                                elif gamma_init_method == "ones":
                                    estimator_extra_params['gammas'] = np.ones((L.shape[1],), dtype=np.float64)
                                
                                else:
                                    raise ValueError(f"Unknown gamma initialization method: {gamma_init_method}")
                                
                                
                                # ------------------------------------------------------
                                
                                
                                # Define the hyperparameters
                                hyperparameters = {
                                    "estimator": estimator.__name__,
                                    "spatial_cv": spatial_cv,
                                    "orientation_type": orientation_type,
                                    "cov_type": cov_type,
                                    "n_times": 1,
                                    "nnz": nnz,
                                    "sigma_squared": round(sigma_squared, 2),
                                    "alpha_snr": alpha_snr,
                                    "gamma_init": gamma_init_method,
                                }

                                experiment_dir = create_directory_structure(base_dir, hyperparameters)
                                
                                if experiment_dir is None:
                                    print("Hyperparameters already exist. Skipping...")
                                    continue  
                                                            
                                data_args_II = {
                                    "n_times": n_times,
                                    "nnz": nnz,
                                    "cov_type": cov_type, # or 'diag'
                                    "path_to_leadfield": path_to_leadfield,
                                    "orientation_type": orientation_type,
                                    "alpha": alpha_snr, 
                                }
                                
                                
                                results = dict(estimator=estimator.__name__) 
                                x, x_hat, y, L, cov, active_set, posterior_cov, results = run(
                                    estimator=estimator,
                                    subject=subject,
                                    metrics=metrics,
                                    data_args=data_args_II,
                                    estimator_args={"alpha":sigma_squared},
                                    random_state=random_state,
                                    memory=memory,
                                    n_jobs=n_jobs,
                                    do_spatial_cv=spatial_cv, 
                                    estimator_extra_params=estimator_extra_params,
                                    results=results,
                                    nruns=nruns
                                )
                                
                                print("------------------ Done! ------------------")

                                # Create a full covariance matrix with zeros
                                full_posterior_cov = np.zeros((x_hat.shape[0], x_hat.shape[0]))

                                # Fill in the active set covariance values
                                for i, idx_i in enumerate(active_set):
                                    for j, idx_j in enumerate(active_set):
                                        full_posterior_cov[idx_i, idx_j] = posterior_cov[i, j]

                                # Test the shape of the full posterior covariance matrix
                                non_zero_values = full_posterior_cov[full_posterior_cov != 0]
                                assert non_zero_values.size == posterior_cov.size == active_set.size ** 2
                                
                                # ------------------------------------------------------
                                
                                plot_active_sources_single_time_step(x, x_hat, active_set, time_step=0, experiment_dir=experiment_dir)
                                
                                plot_posterior_covariance_matrix(posterior_cov, experiment_dir)
                                
                                # ------------------------------------------------------
                                                                
                                x_t0 = x[:, 0]
                                x_hat_t0 = x_hat[:, 0]

                                confidence_levels = np.linspace(0.1, 0.99, 10)
                                CI_count_per_confidence_level = []
                                if not os.path.exists(os.path.join(experiment_dir, 'CI')):
                                    os.makedirs(os.path.join(experiment_dir, 'CI'))
                                
                                    for confidence_level in confidence_levels:    
                                        ci_lower, ci_upper = compute_confidence_intervals(
                                            x_hat_t0, full_posterior_cov, confidence_level
                                        )
                                        count_within_ci = count_values_within_ci(x_t0, ci_lower, ci_upper)
                                        print(f"Confidence lvl: {confidence_level:.2f}, Count Within CI: {count_within_ci}")
                                        CI_count_per_confidence_level.append(count_within_ci)

                                        plot_ci_times(x_hat_t0[:, np.newaxis][active_set],
                                                ci_lower[:, np.newaxis][active_set],
                                                ci_upper[:, np.newaxis][active_set],
                                                x_t0[:, np.newaxis][active_set],
                                                active_set,
                                                figsize=(30,7),
                                                title=f'confidence level {(confidence_level*100):.0f}%',
                                                experiment_dir=os.path.join(experiment_dir, 'CI'),
                                                filename=f'CI_{round(confidence_level, 2)}.png',)

                                plot_ci_count_per_confidence_level(
                                    confidence_levels, 
                                    CI_count_per_confidence_level,
                                    active_set.shape[0],
                                    experiment_dir=experiment_dir)

                                plot_proportion_of_hits(
                                    confidence_levels,
                                    CI_count_per_confidence_level,
                                    x.shape[0],
                                    experiment_dir=experiment_dir)
                                
                                # ------------------------------------------------------

                                # Ensure the number of vertices matches the data
                                n_sources = sum(len(v) for v in vertices)
                                if x.shape[0] != n_sources:
                                    raise ValueError(f"Data has {x.shape[0]} sources, but source space has {n_sources} sources!")

                                # ------------------------------------------------------
                                
                                # Plot the source estimates
                                posteroir_var = np.diag(full_posterior_cov)
                                z_score = x_hat_t0 / (np.sqrt(posteroir_var) + 1e-10)  # mean / std

                                stc_x_t0 = mne.SourceEstimate(x_t0, vertices=vertices, tmin=0, tstep=0)
                                stc_x_hat_t0 = mne.SourceEstimate(x_hat_t0, vertices=vertices, tmin=0, tstep=0)
                                stc_variance = mne.SourceEstimate(posteroir_var, vertices=vertices, tmin=0, tstep=0)
                                stc_zscore = mne.SourceEstimate(z_score, vertices=vertices, tmin=0, tstep=0)

                                source_estimates = [
                                    (stc_x_t0, 'Ground Truth'),
                                    (stc_x_hat_t0, 'Posterior Mean'),
                                    (stc_variance, 'Posterior Variance'),
                                    (stc_zscore, 'Z-Score')
                                ]

                                for stc, title in source_estimates:
                                    brain = stc.plot(hemi="both", subject='fsaverage', subjects_dir=subjects_dir, spacing='ico4', title=title)
                                    
                                    for orientation in orientations:
                                        orientation_dir = os.path.join(experiment_dir, 'brain', orientation)
                                        os.makedirs(orientation_dir, exist_ok=True)
                                        brain.show_view(orientation)
                                        brain.save_image(os.path.join(orientation_dir, f'{title.replace(" ", "_").lower()}_{orientation}.png'))

                                    stc_fs = mne.compute_source_morph(
                                        stc, "sample", "fsaverage", subjects_dir, smooth=5, verbose="error"
                                    ).apply(stc)

                                    brain_fs = stc_fs.plot(
                                        subjects_dir=subjects_dir,
                                        surface="flat",
                                        hemi="both",
                                        size=(1000, 500),
                                        time_viewer=False,
                                        add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
                                        title=title,
                                    )
                                    
                                    brain_fs.save_image(os.path.join(experiment_dir, 'brain', f'{title.replace(" ", "_").lower()}_flat.png'))
                                    brain.close()
                                    brain_fs.close()
                                
print("------------------ Done! ------------------")
                                
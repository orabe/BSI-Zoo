import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, reconstructed_noise
from un_ca_functions import *
from sklearn.utils import check_random_state
import yaml
from joblib import Memory
from pathlib import Path

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

data_args_II = config['data_args_II']
orientation_type = data_args_II['orientation_type']
cov_type = data_args_II['cov_type']
alpha_SNR = data_args_II['alpha_SNR']
nnzs = data_args_II['nnz']
n_times = data_args_II['n_times']


if config['estimator'] == 'gamma_map':
    estimator = gamma_map
    
estimator_args = config['estimator_args']
estimator_extra_params = config['estimator_extra_params']

# Load gamma initialization parameters
gamma_init_method = config['estimator_extra_params']['gammas']
        
# Check which gamma initialization method is enabled
if gamma_init_method['fixed']['enabled']:
    gamma_init_method = 'small_positive_fixed'
elif gamma_init_method['random_uniform']['enabled']:
    gamma_init_method = 'random_uniform'
elif gamma_init_method['ones']['enabled']:
     gamma_init_method = 'ones'
    
estimator_extra_params['tol'] = float(estimator_extra_params['tol'])



data = np.load('bsi_zoo/data/est_data.npz', allow_pickle=True)
# data_args_II=data['data_args_II'].item()
# nnz = data_args_II['nnz']
# alpha_SNR = data_args_II['alpha']

# estimator_args=data['estimator_args'].item()
# estimator_alpha = estimator_args['alpha']

# estimator_extra_params=data['estimator_extra_params']

x = data['x']  # Ground truth sources
x_hat = data['x_hat']  # Estimated sources
y = data['y']  # Sensor data
L = data['L']  # Leadfield matrix
active_set = data['active_set']  # Active set of sources
cov = data['cov']  # Covariance matrix

posterior_cov = data['posterior_cov']  # Posterior covariance matrix

# Create a directory structure based on the hyperparameters
def create_directory_structure(base_dir, params):
    path = base_dir
    for key, value in params.items():
        path = os.path.join(path, f"{key}_{value}")
    os.makedirs(path, exist_ok=True)
    return path

# Define the hyperparameters
hyperparameters = {
    "estimator": estimator.__name__,
    "spatial_cv": spatial_cv,
    "ori_type": data_args_II['orientation_type'],
    "cov_type": data_args_II['cov_type'],
    "n_times": 1,
    "nnz": nnzs,
    "sigma_squared": estimator_args['alpha'],
    "alpha_snr": alpha_SNR,
    "gamma_init": gamma_init_method,
}

# Create the directory structure
base_dir = "experiment_results"
experiment_dir = create_directory_structure(base_dir, hyperparameters)

# # Save the plots in the structured directory
# brain_gt.save_image(os.path.join(experiment_dir, 'brain_gt.png'))
# brain_mean.save_image(os.path.join(experiment_dir, 'brain_mean.png'))
# brain_var.save_image(os.path.join(experiment_dir, 'brain_var.png'))
# brain_zscore.save_image(os.path.join(experiment_dir, 'brain_zscore.png'))

# Create a full covariance matrix with zeros
full_posterior_cov = np.zeros((x_hat.shape[0], x_hat.shape[0]))

# Fill in the active set covariance values
for i, idx_i in enumerate(active_set):
    for j, idx_j in enumerate(active_set):
        full_posterior_cov[idx_i, idx_j] = posterior_cov[i, j]

# Test the shape of the full posterior covariance matrix
non_zero_values = full_posterior_cov[full_posterior_cov != 0]
assert non_zero_values.size == posterior_cov.size == active_set.size ** 2
# posterior_mean = x_hat[active_set]  # Posterior mean
# posterior_mean = posterior_mean[0:10] # [0:10, :]
# posterior_cov = posterior_cov[0:10, 0:10]

# Crop the data to the first 10 sources and time points


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


plot_active_sources_single_time_step(x, x_hat, active_set, time_step=0, experiment_dir=experiment_dir)
plot_posterior_covariance_matrix(posterior_cov, experiment_dir)

# visualize_active_sources(x_hat, active_set, 'Posterior Mean Over Time')
# visualize_active_sources(x, np.where(x[:, 0] != 0)[0], 'Ground Truth Source Activity Over Time')


x_t0 = x[:, 0]
x_hat_t0 = x_hat[:, 0]

confidence_levels = np.linspace(0.1, 0.99, 10)
CI_count_per_confidence_level = []


os.makedirs(os.path.join(experiment_dir, 'CI'), exist_ok=True)
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

# --------------------------------------------------------------------------------
from mne.datasets import fetch_fsaverage, sample

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

# Ensure the number of vertices matches the data
n_sources = sum(len(v) for v in vertices)
if x.shape[0] != n_sources:
    raise ValueError(f"Data has {x.shape[0]} sources, but source space has {n_sources} sources!")


# # Create SourceEstimate object
# tmin = 0.0 
# stc = mne.SourceEstimate(x, vertices=vertices, tmin=tmin, tstep=1/sfreq)

# # Plot the source time courses
# # initial_time = 0.0  # Time point to plot
# stc.plot(subject='fsaverage',
#          hemi='both', 
#         #  initial_time=initial_time,
#          subjects_dir=subjects_dir)


# mpl_fig = stc.plot(
#     subject='fsaverage',
#     subjects_dir=subjects_dir,
#     # initial_time=initial_time,
#     hemi='rh',
#     backend="matplotlib",
#     # clim=dict(kind='value', lims=[-7, 7, 15]),
#     verbose="error",
#     smoothing_steps=5
# )

# stc_fs = mne.compute_source_morph(
#     stc, "sample", "fsaverage", subjects_dir, smooth=5, verbose="error"
# ).apply(stc)
# brain = stc_fs.plot(
#     subjects_dir=subjects_dir,
#     # initial_time=initial_time,
#     clim=dict(kind="value", lims=[3, 6, 9]),
#     surface="flat",
#     hemi="both",
#     size=(1000, 500),
#     smoothing_steps=5,
#     time_viewer=False,
#     add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
# )

# # to help orient us, let's add a parcellation (red=auditory, green=motor,
# # blue=visual)
# brain.add_annotation("HCPMMP1_combined", borders=2)
# brain.save_movie(time_dilation=20, 
#                  interpolation='linear', framerate=10)


# # 0000000000000 STC EST
# stc_xhat = mne.SourceEstimate(x_hat, vertices=vertices, tmin=tmin, tstep=1/sfreq)

# # Plot the source time courses
# # initial_time = 0.0  # Time point to plot
# stc_xhat.plot(subject='fsaverage',
#          hemi='both', 
#         #  initial_time=initial_time,
#          subjects_dir=subjects_dir)


# mpl_fig_xhat = stc_xhat.plot(
#     subject='fsaverage',
#     subjects_dir=subjects_dir,
#     # initial_time=initial_time,
#     hemi='rh',
#     backend="matplotlib",
#     # clim=dict(kind='value', lims=[-7, 7, 15]),
#     verbose="error",
#     smoothing_steps=5
# )

# stc_fs_xhat = mne.compute_source_morph(
#     stc_xhat, "sample", "fsaverage", subjects_dir, smooth=5, verbose="error"
# ).apply(stc_xhat)

# brain = stc_fs_xhat.plot(
#     subjects_dir=subjects_dir,
#     # initial_time=initial_time,
#     clim=dict(kind="value", lims=[3, 6, 9]),
#     surface="flat",
#     hemi="both",
#     size=(1000, 500),
#     smoothing_steps=5,
#     time_viewer=False,
#     add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
# )

# # to help orient us, let's add a parcellation (red=auditory, green=motor,
# # blue=visual)
# brain.add_annotation("HCPMMP1_combined", borders=2)
# brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16,
#                  interpolation='linear', framerate=10)



# --------------------------------------------------------------------------------
from mne.viz import plot_source_estimates

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

orientations = ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal']

for stc, title in source_estimates:
    brain = stc.plot(hemi="both", subject='fsaverage', subjects_dir=subjects_dir, spacing='ico4', title=title)
    
    for orientation in orientations:
        orientation_dir = os.path.join(experiment_dir, 'brain', orientation)
        os.makedirs(orientation_dir, exist_ok=True)
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
        title=title
    )
    
    brain_fs.save_image(os.path.join(experiment_dir, 'brain', f'{title.replace(" ", "_").lower()}_flat.png'))
    

input("Press Enter to close the plots...")




# stc_fs_xhat_t0 = mne.compute_source_morph(
#     stc_x_hat_t0, "sample", "fsaverage", subjects_dir, smooth=5, verbose="error"
# ).apply(stc_x_hat_t0)

# brain = stc_fs_xhat_t0.plot(
#     subjects_dir=subjects_dir,
#     # initial_time=initial_time,
#     clim=dict(kind="value", lims=[3, 6, 9]),
#     surface="flat",
#     hemi="both",
#     size=(1000, 500),
#     smoothing_steps=5,
#     time_viewer=False,
#     add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
# )

# # to help orient us, let's add a parcellation (red=auditory, green=motor,
# # blue=visual)
# brain.add_annotation("HCPMMP1_combined", borders=2)
# brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16,
#                  interpolation='linear', framerate=10)



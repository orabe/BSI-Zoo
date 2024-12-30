import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, reconstructed_noise
from sklearn.utils import check_random_state

from bsi_zoo.estimators import Solver, gamma_map
from bsi_zoo.data_generator import get_data
from scipy import linalg

from scipy.stats import chi2
from matplotlib.patches import Ellipse
from itertools import combinations

# ----- Main Functions -----

def create_directory_structure(base_dir, params):
    experiment_dir = base_dir
    for key, value in params.items():
        experiment_dir = os.path.join(experiment_dir, f"{key}_{value}")
    # if os.path.exists(experiment_dir):
        # print(f"Directory {experiment_dir} already exists. Skipping creation.")
        # return None
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

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
    results,
    nruns=1):
        
    rng = check_random_state(random_state)       
    seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

    try:
        y, L, x, cov, noise_scaled = get_data(**data_args, seed=seeds)
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
        return None, None, None, None, None, None, None, None
    
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

def compute_confidence_intervals(
    mean: np.ndarray, 
    cov: np.ndarray, 
    orientation_type: str = "fixed", 
    confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for a given mean and covariance matrix.

    Parameters:
    ===========
    mean (np.ndarray): The mean values. Can be a 1D or 2D array.
    cov (np.ndarray): The covariance matrix.
    orientation_type (str, optional): The type of orientation. Can be "fixed" or "free". Default is "fixed".
    confidence_level (float, optional): The confidence level for the intervals. 
                                        Default is 0.95.

    Returns:
    ========
    tuple: A tuple containing two numpy arrays:
        - ci_lower (np.ndarray): The lower bounds of the confidence intervals.
        - ci_upper (np.ndarray): The upper bounds of the confidence intervals.

    Raises:
    =======
    ValueError: If the mean array has unsupported number of dimensions.
    ValueError: If the orientation type is unsupported.

    Notes:
    ======
    - For "fixed" orientation type, the function calculates the confidence intervals 
        directly from the mean and the diagonal of the covariance matrix.
    - For "free" orientation type, the function iterates over the orientations (X, Y, Z) and calculates the confidence intervals for each orientation separately.
    - The Z-score is computed based on the given confidence level using a normal distribution.
    """
    # Z-score for the given confidence level
    z = np.abs(np.percentile(np.random.normal(0, 1, 1000000),
                                [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))
    z = z[1]  # For the upper bound of the confidence interval

    if orientation_type == "fixed":
        # Calculate standard deviations from the covariance matrix
        diag_cov = np.diag(cov)
        std_dev = np.sqrt(diag_cov)

        # Compute confidence intervals
        if mean.ndim == 1:
            ci_lower = mean - z * std_dev
            ci_upper = mean + z * std_dev
        elif mean.ndim == 2:
            ci_lower = mean - z * std_dev[:, np.newaxis]
            ci_upper = mean + z * std_dev[:, np.newaxis]
        else:
            raise ValueError("Mean array has unsupported number of dimensions")
    elif orientation_type == "free":
        n_sources = mean.shape[0] // 3
        ci_lower = np.zeros_like(mean)
        ci_upper = np.zeros_like(mean)

        for i in range(3):  # Iterate over orientations X, Y, Z
            mean_orient = mean[i::3]
            diag_cov_orient = np.diag(cov[i::3, i::3])
            std_dev_orient = np.sqrt(diag_cov_orient)

            if mean_orient.ndim == 1:
                ci_lower[i::3] = mean_orient - z * std_dev_orient
                ci_upper[i::3] = mean_orient + z * std_dev_orient
            elif mean_orient.ndim == 2:
                ci_lower[i::3] = mean_orient - z * std_dev_orient[:, np.newaxis]
                ci_upper[i::3] = mean_orient + z * std_dev_orient[:, np.newaxis]
            else:
                raise ValueError("Mean array has unsupported number of dimensions")
    else:
        raise ValueError("Unsupported orientation type")

    return ci_lower, ci_upper


def count_values_within_ci(
    x: np.ndarray, 
    ci_lower: np.ndarray, 
    ci_upper: np.ndarray, 
    orientation_type: str = "fixed"
) -> int:
    """
    Count the number of ground truth values that lie within the confidence intervals.

    Parameters:
    ===========
    x (np.ndarray): The ground truth source activities, shape (n_sources,) or (n_sources, n_times).
    ci_lower (np.ndarray): The lower bounds of the confidence intervals, shape (n_sources,) or (n_sources, n_times).
    ci_upper (np.ndarray): The upper bounds of the confidence intervals, shape (n_sources,) or (n_sources, n_times).
    orientation_type (str, optional): The type of orientation. Can be "fixed" or "free". Default is "fixed".

    Returns:
    ========
    int: The count of ground truth values within the confidence intervals.

    Raises:
    =======
    ValueError: If the orientation type is unsupported.

    Notes:
    ======
    - For "fixed" orientation type, the function counts the values directly from the mean and the diagonal of the covariance matrix.
    - For "free" orientation type, the function iterates over the orientations (X, Y, Z) and counts the values for each orientation separately.
    """
    if orientation_type == "fixed":
        if x.ndim == 1:
            count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper))
        else:
            count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper), axis=0)
    
    elif orientation_type == "free":
        count_within_ci = np.zeros(3)
        if x.ndim == 1:
            for i in range(3):  # Iterate over orientations X, Y, Z
                x_orient = x[i::3]
                ci_lower_orient = ci_lower[i::3]
                ci_upper_orient = ci_upper[i::3]
                count_within_ci[i] = np.sum((x_orient >= ci_lower_orient) &
                                            (x_orient <= ci_upper_orient))
        else:
            for i in range(3):  # Iterate over orientations X, Y, Z
                x_orient = x[i::3, :]
                ci_lower_orient = ci_lower[i::3, :]
                ci_upper_orient = ci_upper[i::3, :]
                count_within_ci[i] = np.sum((x_orient >= ci_lower_orient) &
                                            (x_orient <= ci_upper_orient), axis=0)
    else:
        raise ValueError("Unsupported orientation type")

    return count_within_ci

# ----- Visualization Functions -----
def plot_ci_count_per_confidence_level(
    confidence_levels, 
    CI_count_per_confidence_level, 
    orientation_type='fixed',
    experiment_dir=None, 
    filename='ci_count_per_confidence_level'
):
    """
    Plot the count of ground truth sources within confidence intervals at different confidence levels.

    Parameters:
    ===========
    confidence_levels (list): List of confidence levels.
    CI_count_per_confidence_level (list): List of counts of ground truth sources within confidence intervals.
    active_set_size (int): Size of the active set.
    experiment_dir (str, optional): Directory to save the plot. Default is None.
    filename (str, optional): Filename for the saved plot. Default is 'ci_count_per_confidence_level'.
    orientation_type (str, optional): The type of orientation. Can be 'fixed' or 'free'. Default is 'fixed'.

    Returns:
    ========
    None
    """
    if orientation_type == 'free':
        fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, sharey=True)
        orientations = ['X', 'Y', 'Z']
        for i, ax in enumerate(axes):
            ax.bar(confidence_levels, CI_count_per_confidence_level[:, i], width=0.05)
            ax.set_ylabel('Count Within CI')
            ax.grid(True)
            ax.set_xticks(ticks=confidence_levels)
            ax.set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
            ax.set_title(f'Orientation {orientations[i]}')
        
        ax.set_xlabel('Confidence Level')
        fig.suptitle('Count of Ground Truth Sources Within Confidence Intervals at Different Confidence Levels for Free Orientation\n')
        fig.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.bar(confidence_levels, CI_count_per_confidence_level, width=0.05)
        plt.xlabel('Confidence Level')
        plt.ylabel('Count Within CI')
        plt.title('Count of Ground Truth Sources Within Confidence Intervals at Different Confidence Levels')
        plt.grid(True)
        plt.xticks(ticks=confidence_levels, labels=[f'{int(cl*100)}%' for cl in confidence_levels])
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()

def plot_active_sources_single_time_step(
    x: np.ndarray,
    x_hat: np.ndarray,
    active_set: np.ndarray,
    time_step: int = 0,
    orientation_type: str = 'fixed',
    experiment_dir: str = None,
    filename: str = 'active_sources_single_time_step'
) -> None:
    """
    Plot the active sources for a single time step, comparing ground truth and estimated sources.

    Parameters:
    ===========
    x (np.ndarray): Ground truth source activities, shape (n_sources, n_times).
    x_hat (np.ndarray): Estimated source activities, shape (n_sources, n_times).
    active_set (np.ndarray): Indices of active sources in the estimated data.
    time_step (int, optional): The time step to plot. Default is 0.
    orientation_type (str, optional): The type of orientation. Can be 'fixed' or 'free'. Default is 'fixed'.
    experiment_dir (str, optional): Directory to save the plot. Default is None.
    filename (str, optional): Filename for the saved plot. Default is 'active_sources_single_time_step'.

    Returns:
    ========
    None
    
    Notes:
    ======
    - When orientation_type is 'free', the function will plot three subplots corresponding to the X, Y, and Z orientations.
    """
    if orientation_type == 'free':
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        orientations = ['X', 'Y', 'Z']
        for i, ax in enumerate(axes):
            gt_active_sources = np.where(x[:, time_step] != 0)[0] // 3
            gt_amplitudes = x[gt_active_sources * 3 + i, time_step]
            est_amplitudes = x_hat[active_set[i::3], time_step]
            
            ax.scatter(gt_active_sources, gt_amplitudes, color='blue', alpha=0.6, label=f'Ground Truth (nnz={gt_active_sources.size})')
            ax.scatter(active_set[i::3] // 3, est_amplitudes, color='red', marker='x', alpha=0.6, label=f'Estimated (nnz={active_set[i::3].size})')
            ax.set_xlabel('Source Index')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Orientation {orientations[i]}')
            ax.set_xlim(0, x.shape[0] // 3)
            ax.legend(loc='upper left')
            all_active_sources = np.union1d(gt_active_sources, active_set[i::3] // 3)
            ax.set_xticks(all_active_sources)
            ax.set_xticklabels([f'{idx}' for idx in all_active_sources], rotation=45, ha='right')
            ax.grid(True, alpha=0.5)
        fig.suptitle(f"Active Sources for GT and Estimated Sources\n")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()
    else:
        gt_active_sources = np.where(x[:, time_step] != 0)[0]
        gt_amplitudes = x[gt_active_sources, time_step]
        est_amplitudes = x_hat[active_set, time_step]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(gt_active_sources, gt_amplitudes, color='blue', alpha=0.6, label=f'Ground Truth (nnz={gt_active_sources.size})')
        plt.scatter(active_set, est_amplitudes, color='red', marker='x', alpha=0.6, label=f'Estimated (nnz={active_set.size})')
        plt.xlabel('Source Index')
        plt.ylabel('Amplitude')
        plt.title('Active Sources for GT and Estimated Sources')
        plt.xlim(0, x.shape[0])
        plt.tight_layout()
        plt.legend(loc='upper left')
        all_active_sources = np.union1d(gt_active_sources, active_set)
        plt.xticks(all_active_sources, rotation=45, ha='right')
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()
    
    
def plot_posterior_covariance_matrix(
    posterior_cov: np.ndarray,
    orientation_type: str = 'fixed',
    experiment_dir: str = None,
    filename: str = 'posterior_covariance_matrix'
) -> None:
    """
    Plot the posterior covariance matrix.

    Parameters:
    ===========
    posterior_cov (np.ndarray): The posterior covariance matrix, shape (n_sources, n_sources).
    orientation_type (str, optional): The type of orientation. Can be 'fixed' or 'free'. Default is 'fixed'.
    experiment_dir (str, optional): Directory to save the plot. Default is None.
    filename (str, optional): Filename for the saved plot. Default is 'posterior_covariance_matrix'.

    Returns:
    ========
    None

    Notes:
    ======
    - When orientation_type is 'free', the function will plot three subplots corresponding to the X, Y, and Z orientations.
    - The covariance matrix is split into three sub-matrices for each orientation.
    - The color scale is consistent across all subplots to allow for comparison.
    """
    if orientation_type == 'free':
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        orientations = ['X', 'Y', 'Z']
        vmin = np.min(posterior_cov)
        vmax = np.max(posterior_cov)
        for i, ax in enumerate(axes):
            cov_matrix = posterior_cov[i::3, i::3]
            im = ax.imshow(cov_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'Orientation {orientations[i]}')
            ax.set_xlabel('Sources')
            ax.set_ylabel('Sources')
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.ax.set_ylabel('Covariance Value')
        fig.suptitle(f"Posterior Covariance Matrix\n")

    else:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(posterior_cov, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(im, label='Covariance Value')
        plt.title('Posterior Covariance Matrix')
        plt.xlabel('Sources')
        plt.ylabel('Sources')
        
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))


# def plot_active_sources(x_hat, active_set, title='Posterior Mean Over Time', experiment_dir=None, filename='active_sources_over_time'):
#     plt.figure(figsize=(10, 6))
#     for i in active_set:
#         plt.plot(x_hat[i, :], label=f'Source {i+1}')
#     plt.xlabel('Time Points')
#     plt.ylabel('Source Activity')
#     plt.title(title)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., ncol=2)
#     plt.grid(True)
#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     total_sources = x_hat.shape[0]
#     plt.text(0.95, 0.95, f'Total Sources: {total_sources}', 
#              horizontalalignment='right', verticalalignment='top', 
#              transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
#     plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()

def plot_ci_times(
    x: np.ndarray, 
    x_hat: np.ndarray, 
    active_set: list, 
    ci_lower: np.ndarray, 
    ci_upper: np.ndarray, 
    figsize: tuple, 
    title: str = 'Estimated Source Activity with Confidence Intervals', 
    experiment_dir: str = None, 
    filename: str = 'all_ci_times',
    orientation_type: str = 'fixed',
):
    """
    Plots the estimated source activity with confidence intervals.

    Parameters:
    ===========
    x (np.ndarray): Ground truth source activity, shape (n_sources,).
    x_hat (np.ndarray): Posterior mean estimates of source activity, shape (n_sources,).
    active_set (list): List of active sources.
    ci_lower (np.ndarray): Lower bounds of the confidence intervals, shape (n_sources,).
    ci_upper (np.ndarray): Upper bounds of the confidence intervals, shape (n_sources,).
    figsize (tuple): Size of the figure.
    title (str, optional): Title of the plot. Default is 'Estimated Source Activity with Confidence Intervals'.
    experiment_dir (str, optional): Directory to save the plot. Default is None.
    filename (str, optional): Filename to save the plot. Default is 'all_ci_times'.
    orientation_type (str, optional): The type of orientation. Can be 'fixed' or 'free'. Default is 'fixed'.

    Returns:
    ========
    None
    
    Note:
    =====
    - When orientation_type is 'free', the function will plot three subplots corresponding to the X, Y, and Z orientations.

    """
    n_sources = x_hat.shape[0]

    if orientation_type == 'free':
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharey=True)
        orientations = ['X', 'Y', 'Z']
        for j, (ax, orient) in enumerate(zip(axes, orientations)):
            for i in range(n_sources // 3):
                idx = i * 3 + j
                ax.scatter(i, x_hat[idx], marker='x', color='red', label='Posterior Mean' if i == 0 else "")  
                ax.fill_between(
                    [i - 0.5, i + 0.5], 
                    ci_lower[idx], 
                    ci_upper[idx], 
                    alpha=0.3,
                    label='Confidence Interval' if i == 0 else ""
                )
                ax.scatter(i, x[idx], s=10, color='blue', label='Ground Truth' if i == 0 else "")
            ax.set_title(f'Orientation {orient}')
            ax.axhline(0, color='grey', lw=0.8, ls='-')
            ax.legend(loc='upper right', title=f'(Total Sources: {n_sources // 3})')
            ax.set_xticks(np.arange(n_sources // 3))
            ax.set_xticklabels([i for i in active_set[::3] // 3], rotation=45)
        fig.suptitle(title)
        fig.text(0.5, 0.04, 'Source index', ha='center')
        fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')
        fig.suptitle('Estimated Source Activity with Confidence Intervals for free orientation\n', fontsize=20)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n_sources):             
            ax.scatter(i, x_hat[i], marker='x', color='red', label='Posterior Mean' if i == 0 else "")  
            ax.fill_between(
                [i - 0.5, i + 0.5], 
                ci_lower[i], 
                ci_upper[i], 
                alpha=0.3,
                label='Confidence Interval' if i == 0 else ""
            )
            ax.scatter(i, x[i], s=10, color='blue', label='Ground Truth' if i == 0 else "")
        ax.set_xticks(np.arange(n_sources))
        ax.set_xticklabels([f'{idx}' for idx in active_set], rotation=45)
        ax.set_title(title)
        ax.axhline(0, color='grey', lw=0.8, ls='-')
        ax.legend(loc='upper right', title=f'(Total Sources: {n_sources}, Active Sources: {len(active_set)})')
        fig.text(0.5, 0.04, 'Source index', ha='center')
        fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()
 
# def plot_ci_sources(x_hat, ci_lower, ci_upper, x, figsize, experiment_dir=None, filename='all_ci_sources'):
#     n_sources = x_hat.shape[0]
#     n_times = x_hat.shape[1]
    
#     n_cols = min(5, n_sources)
#     n_rows = (n_sources + n_cols - 1) // n_cols
#     time_points = np.linspace(0, 1, n_times)

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False, constrained_layout=True)
#     axes = axes.flatten()

#     for i in range(n_sources):
#         axes[i].plot(time_points, x_hat[i, :], marker='o', label='Posterior Mean')  
#         axes[i].fill_between(
#             time_points,
#             ci_lower[i, :],
#             ci_upper[i, :],
#             alpha=0.3,
#             label='Confidence Interval'
#         )
#         axes[i].scatter(time_points, x[i, :], color='red', s=10, label='Ground Truth')
#         axes[i].set_title(f'Source {i + 1}')
#         axes[i].axhline(0, color='grey', lw=0.8, ls='--')
#         axes[i].grid()
#         axes[i].set_xlabel('Time') 

#     for ax in axes:
#         ax.set_ylabel('')  

#     fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')

#     for j in range(n_sources, n_rows * n_cols):
#         fig.delaxes(axes[j])

#     fig.suptitle('Estimated Source Activity with Confidence Intervals')
#     plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
#     plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
#     plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
#     # plt.show()
def plot_proportion_of_hits(
    confidence_levels, 
    CI_count_per_confidence_level, 
    total_sources, 
    orientation_type='fixed',
    experiment_dir=None, 
    filename='proportion_of_hits', 
):
    """
    Plot the proportion of hits within confidence intervals at different confidence levels.

    Parameters:
    ===========
    confidence_levels (list): List of confidence levels.
    CI_count_per_confidence_level (list): List of counts of ground truth sources within confidence intervals.
    total_sources (int): Total number of sources.
    experiment_dir (str, optional): Directory to save the plot. Default is None.
    filename (str, optional): Filename for the saved plot. Default is 'proportion_of_hits'.
    orientation_type (str, optional): The type of orientation. Can be 'fixed' or 'free'. Default is 'fixed'.

    Returns:
    ========
    None
    """
    if orientation_type == 'free':
        fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, sharey=True)
        orientations = ['X', 'Y', 'Z']
        for i, ax in enumerate(axes):
            hits = np.array(CI_count_per_confidence_level[:, i])
            misses = total_sources // 3 - hits
            proportions = hits / (hits + misses)
            ax.bar(confidence_levels, proportions, width=0.05)
            ax.set_ylabel('Proportion of Hits')
            ax.grid(True)
            ax.set_xticks(ticks=confidence_levels)
            ax.set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
            ax.set_title(f'Orientation {orientations[i]}')
        
        ax.set_xlabel('Confidence Level')
        fig.suptitle('Proportion of Hits Within Confidence Intervals at Different Confidence Levels for Free Orientation\n')
        fig.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()
    else:
        hits = np.array(CI_count_per_confidence_level)
        misses = total_sources - hits
        proportions = hits / (hits + misses)

        plt.figure(figsize=(10, 6))
        plt.bar(confidence_levels, proportions, width=0.05)
        plt.xlabel('Confidence Level')
        plt.ylabel('Proportion of Hits (hits / (hits + misses))')
        plt.title('Proportion of Hits Within Confidence Intervals')
        plt.grid(True)
        plt.xticks(ticks=confidence_levels, labels=[f'{int(cl*100)}%' for cl in confidence_levels])
        plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
        # plt.show()
    # plt.show()
  
    


    
# ------------- Confidence Ellipses (eigenvalues) ------------

def plot_sorted_variances(cov, experiment_dir, filename="sorted_variances", top_k=None):
    """
    Plot the sorted variances from the covariance matrix, highlighting the top-k variances.

    Parameters:
    cov (array): Posterior covariance matrix of shape (n, n).
    top_k (int, optional): Number of top variances to highlight. If None, highlights all.

    Returns:
    None
    """
    # Extract variances (diagonal of covariance matrix)
    variances = np.diag(cov)
    
    # Sort variances in descending order
    sorted_indices = np.argsort(variances)[::-1]
    sorted_variances = variances[sorted_indices]
    
    # Plot the variances
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_variances)), sorted_variances, color='skyblue', edgecolor='blue')
    
    # Highlight top-k variances if specified
    if top_k is not None:
        for bar in bars[:top_k]:
            bar.set_color('orange')
    
    plt.xticks(range(len(sorted_variances)), sorted_indices, rotation=45)
    plt.xlabel("Source Index")
    plt.ylabel("Variance")
    plt.title(f"Sorted Posterior Variances (Top-{top_k if top_k else len(variances)} Highlighted)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))   
       
def compute_top_covariance_pairs(cov, top_k=None):
    """
    Compute and optionally sort the magnitudes of covariances for all pairs of dimensions.

    Parameters:
        cov (array-like): Covariance matrix of shape (n, n).
        top_k (int, optional): Number of top pairs to return. If None, return all pairs.

    Returns:
        list: A sorted list of tuples. Each tuple contains:
              - A pair of indices (i, j).
              - The absolute magnitude of their covariance.
    """
    # Ensure covariance matrix is a NumPy array
    cov = np.asarray(cov)

    # Get all unique pairs of indices
    n = cov.shape[0]
    pairs = list(combinations(range(n), 2))

    # Compute magnitudes of covariances for each pair
    pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]

    # Sort by covariance magnitude in descending order
    sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)

    # Return top-k pairs if specified
    if top_k is not None:
        return sorted_pairs[:top_k]
    return sorted_pairs

def visualize_sorted_covariances(cov, experiment_dir, filename="sorted_covariances", top_k=None):
    """
    Visualize the sorted magnitudes of covariances for all pairs of dimensions.

    Parameters:
        cov (array-like): Covariance matrix of shape (n, n).
        top_k (int, optional): Number of top pairs to visualize. If None, visualize all pairs.

    Returns:
        None
    """
    # Compute sorted pairs and magnitudes
    sorted_pairs = compute_top_covariance_pairs(cov, top_k=top_k)
    
    # Extract pairs and magnitudes
    pairs = [f"({i},{j})" for (i, j), _ in sorted_pairs]
    magnitudes = [magnitude for _, magnitude in sorted_pairs]
    
    # Plot the sor ted magnitudes
    plt.figure(figsize=(10, 6))
    plt.bar(pairs, magnitudes, color='skyblue')
    plt.xlabel('Pairs of Dimensions')
    plt.ylabel('Covariance Magnitude')
    plt.title(f"Top-{top_k if top_k else len(magnitudes)} Sorted Covariance Magnitudes")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    

def make_psd(cov, epsilon=1e-6):
    """
    Ensure that the covariance matrix is positive semi-definite by adding epsilon to the diagonal.
    """
    print("Regularizing covariance matrix...")
    while not np.all(np.linalg.eigvals(cov) >= 0):
        cov += np.eye(cov.shape[0]) * epsilon
        epsilon *= 10  # Gradually increase epsilon if needed
    return cov

def compute_confidence_ellipse(mean, cov, confidence_level=0.95):
    """
    Compute the parameters of a confidence ellipse for a given mean and covariance matrix.
    """
    # Validate covariance matrix
    condition_number = np.linalg.cond(cov)
    if condition_number > 1e10:
        print("Covariance matrix is ill-conditioned")
    
    # Regularize covariance matrix if not positive definite by adding gradually increasing epsilon to the diagonal.
    if not np.all(np.linalg.eigvals(cov) > 0):
        cov = make_psd(cov, epsilon=1e-6)
    
    chi2_val = chi2.ppf(confidence_level, df=2)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    if np.all(eigenvals > 0):
        print("Covariance matrix is now positive definite.")
    else:
        print("Covariance matrix is still not positive definite.")

    order = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    width, height = 2 * np.sqrt(eigenvals * chi2_val)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # TODO: replace nan values with small values to enable plotting
    # if np.isnan(width):
    #     width = 1e-6 
    #     print("Warning: Replacing NaN width with a small value.")
    
    # if np.isnan(height):
    #     height = 1e-6 
    #     print("Warning: Replacing NaN height with a small value.")
        
    return width, height, angle


def plot_confidence_ellipse(mean, width, height, angle, ax=None, **kwargs):
    """
    Plot a confidence ellipse for given parameters.

    Parameters:
    - mean: array-like, shape (2,)
        The mean of the data in the two dimensions being plotted.
    - width: float
        The width of the ellipse (related to variance along the major axis).
    - height: float
        The height of the ellipse (related to variance along the minor axis).
    - angle: float
        The rotation angle of the ellipse in degrees.
    - ax: matplotlib.axes.Axes, optional
        The axis on which to plot the ellipse. If None, creates a new figure.
    - **kwargs: additional keyword arguments for matplotlib.patches.Ellipse.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Add ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    ax.scatter(*mean, color='blue', label='Mean')
    
    # Set axis labels
    ax.set_xlabel("Principal Component 1 (Variance in Dim 1)")
    ax.set_ylabel("Principal Component 2 (Variance in Dim 2)")

    # Set title
    ax.set_title("Confidence Ellipse (Width and Height Indicate Variance)")
    ax.grid()
    ax.legend()
    

def plot_top_relevant_CE_pairs(mean, cov, experiment_dir, filename="top_relevant_CE_pairs", top_k=5, confidence_level=0.95):
    """
    Identify the top-k relevant pairs of dimensions (based on covariance magnitude)
    and plot their confidence ellipses.
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    n = len(mean)
    pairs = list(combinations(range(n), 2))
        
    # Compute magnitudes of covariances for each pair
    pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]
    
    # Sort by covariance magnitude (descending order)
    sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)
    top_pairs = [pair for pair, _ in sorted_pairs[:top_k]]
    
    # Dynamic grid for subplots
    n_cols = min(3, top_k)
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    # Debug: Print sorted pairs and their covariances
    for idx, (pair, magnitude) in enumerate(sorted_pairs[:top_k]):
        print(f"Pair {idx + 1}: Dimensions {pair} | Covariance Magnitude: {magnitude}")
    
    # Debug: Print covariance submatrices
    for idx, (i, j) in enumerate(top_pairs):
        print(f"Pair {idx + 1}: Dimensions {i, j}")
        print(f"Covariance submatrix:\n{cov[np.ix_([i, j], [i, j])]}")
        
    for idx, (i, j) in enumerate(top_pairs):
        mean_ij = mean[[i, j]]
        cov_ij = cov[np.ix_([i, j], [i, j])]

        # Plot confidence ellipse
        width, height, angle = compute_confidence_ellipse(mean_ij, cov_ij, confidence_level)
        plot_confidence_ellipse(mean_ij, width, height, angle, ax=axes[idx], edgecolor='blue', alpha=0.5)
        
        axes[idx].set_title(f"Dimensions {i} & {j}")

    # Remove unused axes
    for ax in axes[len(top_pairs):]:
        fig.delaxes(ax)

    fig.suptitle("Top Relevant Dimensional Pairs with Confidence Ellipses", fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))   

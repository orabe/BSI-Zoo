import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, reconstructed_noise
from sklearn.utils import check_random_state

from bsi_zoo.estimators import Solver, gamma_map
from bsi_zoo.data_generator import get_data
from scipy import linalg

# ----- Main Functions -----

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
        y, L, x, cov, _ = get_data(**data_args, seed=seeds)
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

def compute_confidence_intervals(mean, cov, confidence_level=0.95):
    # Z-score for the given confidence level
    z = np.abs(np.percentile(np.random.normal(0, 1, 1000000),
                             [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))
    z = z[1]  # For the upper bound of the confidence interval

    # Calculate standard deviations from the covariance matrix
    std_dev = np.sqrt(np.diag(cov))

    # Compute confidence intervals
    if mean.ndim == 1:
        ci_lower = mean - z * std_dev
        ci_upper = mean + z * std_dev
    elif mean.ndim == 2:
        ci_lower = mean - z * std_dev[:, np.newaxis]
        ci_upper = mean + z * std_dev[:, np.newaxis]
    else:
        raise ValueError("Mean array has unsupported number of dimensions")

    return ci_lower, ci_upper

def count_values_within_ci(x, ci_lower, ci_upper):
    """
    Count the number of ground truth values that lie within the confidence intervals.

    Parameters:
    x (np.ndarray): The ground truth source activities, shape (n_sources, n_times).
    ci_lower (np.ndarray): The lower bounds of the confidence intervals, shape (n_sources, n_times).
    ci_upper (np.ndarray): The upper bounds of the confidence intervals, shape (n_sources, n_times).

    Returns:
    int: The count of ground truth values within the confidence intervals.
    """
    if x.ndim == 1:
        count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper))
    else:
        count_within_ci = np.zeros(x.ndim)
        for i in range(x.ndim):
            count_within_ci[i] = np.sum((x[:, i] >= ci_lower[:, i]) &
                                        (x[:, i] <= ci_upper[:, i]))
    return count_within_ci

# ----- Visualization Functions -----
def plot_ci_count_per_confidence_level(confidence_levels, CI_count_per_confidence_level, active_set_size, experiment_dir=None, filename='ci_count_per_confidence_level'):
    plt.figure(figsize=(10, 6))
    plt.bar(confidence_levels, CI_count_per_confidence_level, width=0.05)
    plt.xlabel('Confidence Level')
    plt.ylabel('Count Within CI')
    plt.title('Count of Ground Truth Sources Within Confidence Intervals at Different Confidence Levels')
    plt.grid(True)
    plt.xticks(ticks=confidence_levels, labels=[f'{int(cl*100)}%' for cl in confidence_levels])
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()

def plot_active_sources_single_time_step(x, x_hat, active_set, time_step=0, experiment_dir=None, filename='active_sources_single_time_step'):
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
    
def plot_posterior_covariance_matrix(posterior_cov, experiment_dir=None, filename='posterior_covariance_matrix'):
    plt.figure(figsize=(10, 8))
    plt.imshow(posterior_cov, cmap='viridis', aspect='auto')
    plt.colorbar(label='Covariance Value')
    plt.title('Posterior Covariance Matrix')
    plt.xlabel('Sources')
    plt.ylabel('Sources')
    plt.grid(False)
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()

def plot_active_sources(x_hat, active_set, title='Posterior Mean Over Time', experiment_dir=None, filename='active_sources_over_time'):
    plt.figure(figsize=(10, 6))
    for i in active_set:
        plt.plot(x_hat[i, :], label=f'Source {i+1}')
    plt.xlabel('Time Points')
    plt.ylabel('Source Activity')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., ncol=2)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    total_sources = x_hat.shape[0]
    plt.text(0.95, 0.95, f'Total Sources: {total_sources}', 
             horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()
    
def plot_ci_times(x_hat, ci_lower, ci_upper, x, active_set, figsize, title='Estimated Source Activity with Confidence Intervals', experiment_dir=None, filename='all_ci_times'):
    n_sources = x_hat.shape[0]
    n_times = x_hat.shape[1] 

    fig, axes = plt.subplots(1, n_times, figsize=figsize, sharey=False)
    if n_times == 1:
        axes = [axes]

    for t in range(n_times):
        for i in range(n_sources):             
            axes[t].scatter(i, x_hat[i, t], marker='x', color='red', label='Posterior Mean' if i == 0 else "")  
            axes[t].fill_betweenx(
                [ci_lower[i, t], ci_upper[i, t]], 
                i - 0.5, 
                i + 0.5, 
                alpha=0.3,
                label='Confidence Interval' if i == 0 else ""
            )
            axes[t].scatter(i, x[i, t], s=10, color='blue', label='Ground Truth' if i == 0 else "")
        axes[t].set_title(title)
        axes[t].set_xticks(np.arange(n_sources))
        axes[t].set_xticklabels([f'Source {idx}' for idx in active_set], rotation=45)
        axes[t].axhline(0, color='grey', lw=0.8, ls='-')

    fig.suptitle('Estimated Source Activity with Confidence Intervals')
    fig.text(0.5, 0.04, 'Sources', ha='center')
    fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()

def plot_ci_sources(x_hat, ci_lower, ci_upper, x, figsize, experiment_dir=None, filename='all_ci_sources'):
    n_sources = x_hat.shape[0]
    n_times = x_hat.shape[1]
    
    n_cols = min(5, n_sources)
    n_rows = (n_sources + n_cols - 1) // n_cols
    time_points = np.linspace(0, 1, n_times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False, constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_sources):
        axes[i].plot(time_points, x_hat[i, :], marker='o', label='Posterior Mean')  
        axes[i].fill_between(
            time_points,
            ci_lower[i, :],
            ci_upper[i, :],
            alpha=0.3,
            label='Confidence Interval'
        )
        axes[i].scatter(time_points, x[i, :], color='red', s=10, label='Ground Truth')
        axes[i].set_title(f'Source {i + 1}')
        axes[i].axhline(0, color='grey', lw=0.8, ls='--')
        axes[i].grid()
        axes[i].set_xlabel('Time') 

    for ax in axes:
        ax.set_ylabel('')  

    fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')

    for j in range(n_sources, n_rows * n_cols):
        fig.delaxes(axes[j])

    fig.suptitle('Estimated Source Activity with Confidence Intervals')
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.savefig(os.path.join(experiment_dir, f'{filename}.png'))
    # plt.show()

def plot_proportion_of_hits(confidence_levels, CI_count_per_confidence_level, total_sources, experiment_dir=None, filename='proportion_of_hits'):
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
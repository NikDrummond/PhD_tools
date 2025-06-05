import pandas as pd
import numpy as np
import GeoJax as gj
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from jax import random

@jit
def resultant_vector_length(arr: jnp.ndarray) -> float:

    n = arr.shape[0]
    sum_cos = jnp.sum(jnp.cos(arr))
    sum_sin = jnp.sum(jnp.sin(arr))
    R = jnp.sqrt(sum_cos**2 + sum_sin**2) / n
    return R

def unpack_groups(df, group_col, value_col) -> list:
    groups = df[group_col].to_numpy()
    values = df[value_col].to_numpy()
    unique_groups = np.unique(groups)
    return [values[groups == g] for g in unique_groups]

def unpack_groups_perm(values: np.ndarray, group_ids: np.ndarray) -> list:
    unique_groups = np.unique(group_ids)
    return [values[group_ids == g] for g in unique_groups]


def shuffle_group_labels(group_labels: np.ndarray) -> np.ndarray:
    return np.random.permutation(group_labels)

def between_group_dispersion(data_by_group, circmean_func, resultant_vector_length_func):
    
    # Flatten all data to compute grand mean
    all_data = jnp.concatenate(data_by_group)
    grand_mean = circmean_func(all_data)

    ss_between = 0.0
    for group_data in data_by_group:
        n_i = group_data.shape[0]
        theta_i = circmean_func(group_data)
        R_i = resultant_vector_length_func(group_data)
        ss_between += n_i * R_i * jnp.cos(theta_i - grand_mean)

    return ss_between

def within_group_dispersion(data_by_group, circmean_func):
    ss_within = 0.0

    for group_data in data_by_group:
        theta_i = circmean_func(group_data)
        diffs = group_data - theta_i
        ss_i = jnp.sum(1 - jnp.cos(diffs))
        ss_within += ss_i

    return ss_within

def compute_circular_omega2(data_by_group, circmean_func, rvl_func):
    k = len(data_by_group)
    n_total = sum(len(g) for g in data_by_group)

    ss_between = between_group_dispersion(data_by_group, circmean_func, rvl_func)
    ss_within = within_group_dispersion(data_by_group, circmean_func)

    ms_within = ss_within / (n_total - k)
    ss_total = ss_between + ss_within

    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return omega2

def bootstrap_omega2(
    data_by_group, circmean_func, rvl_func, n_boot=1000, seed=None
):
    rng = np.random.default_rng(seed)
    boot_estimates = np.zeros(n_boot)

    for i in tqdm(range(n_boot)):
        resampled = [
            rng.choice(group, size=len(group), replace=True)
            for group in data_by_group
        ]
        boot_estimates[i] = compute_circular_omega2(resampled, circmean_func, rvl_func)

    lower = np.percentile(boot_estimates, 2.5)
    upper = np.percentile(boot_estimates, 97.5)

    return boot_estimates, (lower, upper)

def permutation_multi_group_circular(df, group_col, value_col, n_perms = 1000):
    # get observed groups
    obs_groups = unpack_groups(df, group_col, value_col)
    # get observed test statistic
    SS_between_obs = between_group_dispersion(obs_groups, gj.circmean, resultant_vector_length)

    # permutation
    perm_test_stat = np.zeros(n_perms)

    group_labels = df['Type'].to_numpy()
    values = jnp.array(df['PC1_Angle'].to_numpy())

    # seed = 42

    for i in tqdm(range(n_perms)):
        # permute labels
        labels = shuffle_group_labels(group_labels)

        # permuted groups
        perm_groups = unpack_groups_perm(values, labels)
        # calculate test statistic
        perm_val = between_group_dispersion(perm_groups, gj.circmean, resultant_vector_length)
        # update perm_test_stat
        perm_test_stat[i] = perm_val

    p_value = np.mean(perm_test_stat >= SS_between_obs)

    return SS_between_obs, p_value

def circ_omega_square(df, group_col, val_col, n_bootstraps = 1000, return_all = False):
    obs_groups = unpack_groups(df, group_col, val_col)
    omega2 = compute_circular_omega2(obs_groups, gj.circmean, resultant_vector_length)
    boot_vals, ci = bootstrap_omega2(obs_groups, gj.circmean, resultant_vector_length, n_boot = n_bootstraps)
    
    if return_all:
        return omega2, ci, boot_vals
    else:
        return omega2, ci
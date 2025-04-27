import numpy as np

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit
from itertools import combinations
import scipy.stats as stats
from itertools import combinations

from jax.ops import segment_sum
from scipy.stats import rankdata
from itertools import combinations

def edf(data, alpha=0.05, x0=None, x1=None, bins=None):
    """
    Calculate the empirical distribution function and confidence intervals.

    Parameters
    ----------
    data : np.ndarray
        1D array of sample data.
    alpha : float, optional
        Confidence level for the interval (default is 0.05).
    x0 : float, optional
        Lower bound of the x-range (default is min(data) with buffer).
    x1 : float, optional
        Upper bound of the x-range (default is max(data) with buffer).
    bins : int, optional
        Number of evaluation points (adaptive if None).

    Returns
    -------
    x : np.ndarray
        Evaluation points.
    y : np.ndarray
        EDF values.
    l : np.ndarray
        Lower bound of confidence interval.
    u : np.ndarray
        Upper bound of confidence interval.
    """

    # Input validation
    assert isinstance(data, np.ndarray), "data must be a NumPy array"
    assert data.ndim == 1, "data must be a 1D array"
    assert 0 < alpha < 1, "alpha must be in (0,1)"
    
    data = np.sort(data)  # Sort data for faster search
    N = len(data)
    
    # Define x range with optional buffer
    x0 = np.min(data) if x0 is None else x0
    x1 = np.max(data) if x1 is None else x1
    buffer = 0.05 * (x1 - x0)  # Small buffer around observed range
    x0, x1 = x0 - buffer, x1 + buffer

    # Adaptive bin selection using Freedman-Diaconis rule
    if bins is None:
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / np.cbrt(N)  # Freedman-Diaconis rule
        bins = max(10, int((x1 - x0) / bin_width))  # Ensure reasonable number

    x = np.linspace(x0, x1, bins)
    
    # Compute EDF efficiently
    y = np.searchsorted(data, x, side='right') / N
    
    # Confidence interval calculation
    e = np.sqrt(np.log(2.0 / alpha) / (2 * N))
    l = np.maximum(y - e, 0)
    u = np.minimum(y + e, 1)
    
    return x, y, l, u

def epdf(data, alpha=0.05, x0=None, x1=None, bins=None, n_bootstrap=1000):
    """
    Estimate the empirical probability density function (EPDF) and confidence intervals.

    Parameters
    ----------
    data : np.ndarray
        1D array of sample data.
    alpha : float, optional
        Confidence level (default is 0.05 for 95% confidence intervals).
    x0 : float, optional
        Lower bound of the x-axis range (default: min(data)).
    x1 : float, optional
        Upper bound of the x-axis range (default: max(data)).
    bins : int, optional
        Number of bins (default: automatic selection using Freedman-Diaconis rule).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default is 1000).

    Returns
    -------
    x : np.ndarray
        Bin centers where the density is evaluated.
    y : np.ndarray
        Estimated density values.
    l : np.ndarray
        Lower bound of the confidence interval.
    u : np.ndarray
        Upper bound of the confidence interval.
    """

    # Validate input
    assert isinstance(data, np.ndarray) and data.ndim == 1, "data must be a 1D NumPy array"
    assert 0 < alpha < 1, "alpha must be in (0,1)"

    # Define x0 and x1 with optional buffer
    x0 = np.min(data) if x0 is None else x0
    x1 = np.max(data) if x1 is None else x1
    buffer = 0.05 * (x1 - x0)  # Small buffer outside observed range
    x0, x1 = x0 - buffer, x1 + buffer

    # Determine number of bins
    if bins is None:
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
        bins = max(10, int((x1 - x0) / bin_width))  # Ensure at least 10 bins

    # Compute histogram (EPDF)
    bin_edges = np.linspace(x0, x1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_values, _ = np.histogram(data, bins=bin_edges, density=True)

    # Bootstrap confidence intervals
    boot_samples = np.zeros((n_bootstrap, bins))
    for i in range(n_bootstrap):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        boot_samples[i], _ = np.histogram(resampled_data, bins=bin_edges, density=True)

    # Compute confidence intervals
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(boot_samples, 100 * (1 - alpha / 2), axis=0)

    return bin_centers, hist_values, lower_bound, upper_bound

def asymmetric_mad(data):
    """
    Computes asymmetric MAD for each list in a dictionary or each column of a 2D array.

    Parameters:
        data (dict or np.ndarray):
            - dict: keys are labels, values are 1D lists/arrays (may have different lengths)
            - ndarray: shape (n_samples, n_features)

    Returns:
        If dict: dict with keys 'median', 'mad_low', 'mad_high' -> subdicts by original keys
        If array: tuple of (median, mad_low, mad_high) as ndarrays
    """
    if isinstance(data, dict):
        result = {"median": {}, "mad_low": {}, "mad_high": {}}
        for key, values in data.items():
            arr = np.asarray(values)
            median = np.nanmedian(arr)
            diffs = np.abs(arr - median)
            below = arr < median
            above = arr > median

            mad_low = np.nanmedian(diffs[below]) if np.any(below) else np.nan
            mad_high = np.nanmedian(diffs[above]) if np.any(above) else np.nan

            result["median"][key] = median
            result["mad_low"][key] = mad_low
            result["mad_high"][key] = mad_high
        return result

    # Original behavior for 2D array
    median = np.nanmedian(data, axis=0)
    below = data < median
    above = data > median

    mad_low = np.full(data.shape[1], np.nan)
    mad_high = np.full(data.shape[1], np.nan)

    for i in range(data.shape[1]):
        diffs = np.abs(data[:, i] - median[i])
        if np.any(below[:, i]):
            mad_low[i] = np.nanmedian(diffs[below[:, i]])
        if np.any(above[:, i]):
            mad_high[i] = np.nanmedian(diffs[above[:, i]])

    return median, mad_low, mad_high


def bootstrap_ci_mean(data, n_boot=1000, ci=95):
    """
    Bootstrap confidence intervals for mean of each list in a dictionary or column of a 2D array.

    Parameters:
        data : dict or ndarray
            - dict: keys map to lists of numbers (different lengths allowed)
            - ndarray: shape (n_samples, n_features)
        n_boot : int
            Number of bootstrap samples.
        ci : float
            Confidence interval width (e.g. 95 for 95% CI).

    Returns:
        If dict: dict with keys 'mean', 'lower', 'upper' -> subdicts by original keys
        If array: tuple of (mean, lower_error, upper_error)
    """
    if isinstance(data, dict):
        result = {"mean": {}, "lower": {}, "upper": {}}
        lower_percentile = (100 - ci) / 2
        upper_percentile = 100 - lower_percentile

        for key, values in data.items():
            arr = np.asarray(values)
            boot_means = np.zeros(n_boot)
            n = len(arr)

            for i in range(n_boot):
                sample = arr[np.random.randint(0, n, size=n)]
                boot_means[i] = np.nanmean(sample)

            mean = np.nanmean(arr)
            lower = np.percentile(boot_means, lower_percentile)
            upper = np.percentile(boot_means, upper_percentile)

            result["mean"][key] = mean
            result["lower"][key] = mean - lower
            result["upper"][key] = upper - mean
        return result

    # Original behavior for 2D array
    n, m = data.shape
    means = np.zeros((n_boot, m))

    for i in range(n_boot):
        sample = data[np.random.randint(0, n, size=n), :]
        means[i] = np.nanmean(sample, axis=0)

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    mean = np.nanmean(data, axis=0)
    lower = np.percentile(means, lower_percentile, axis=0)
    upper = np.percentile(means, upper_percentile, axis=0)

    return mean, mean - lower, upper - mean




def kruskal_ANOVA(
    df, group_col, value_col, rank_col=None, rank_value=None):
    """
    Perform Kruskal-Wallis test and calculate Epsilon-squared effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing values and group labels.
    group_col : str
        Column name for group labels.
    value_col : str
        Column name for the measured values.
    rank_col : str, optional
        Column name for eigenvalue rank if filtering is needed, by default None.
    rank_value : int, optional
        Value of eigenvalue rank to filter on, by default None.

    Returns
    -------
    tuple
        Kruskal-Wallis H-statistic (float), Epsilon-squared effect size (float), and p-value (float).

    Raises
    ------
    ValueError
        If the input DataFrame does not contain specified columns.
    """

    # 1. Optional filter by eigenvalue rank
    if rank_col and rank_value is not None:
        df = df[df[rank_col] == rank_value]

    # 2. Prepare groups
    group_names = df[group_col].unique()
    groups = [df[df[group_col] == g][value_col] for g in group_names]

    # 3. Kruskal-Wallis Test
    h_stat, p_kruskal = stats.kruskal(*groups)

    # 4. calculate effect size
    k = len(group_names)
    n = len(df)
    ef = (h_stat - k + 1) / (n - k)

    return h_stat, ef, p_kruskal

def Mann_Whitney_pairwise(df, value_col, group_col, rank_col=None, rank_value=None, correction = 'bonferroni'):
    """
    Perform pairwise Mann-Whitney U-tests with multiple testing correction.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing values and group labels.
    value_col : str
        Column name for the measured values.
    group_col : str
        Column name for group labels.
    rank_col : str, optional
        Column name for eigenvalue rank if filtering is needed, by default None.
    rank_value : int, optional
        Value of eigenvalue rank to filter on, by default None.
    correction : str, optional
        Method for multiple comparisons correction, 'bonferroni' or 'holm', by default 'bonferroni'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing group pairs, U-statistic, raw p-values, and corrected p-values.

    Raises
    ------
    ValueError
        If the correction method specified is not 'bonferroni' or 'holm'.
    """
    # 1. Optional filter by eigenvalue rank
    if rank_col and rank_value is not None:
        df = df[df[rank_col] == rank_value]

    # 2. Prepare groups
    group_names = df[group_col].unique()

    # Pairwise Mann-Whitney tests
    pairwise_results = []

    for g1, g2 in combinations(group_names, 2):
        data1 = df[df[group_col] == g1][value_col]
        data2 = df[df[group_col] == g2][value_col]
        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        pairwise_results.append((g1, g2, u_stat, p_val))

    results_df = pd.DataFrame(
        pairwise_results, columns=["Group1", "Group2", "U_statistic", "p_value"]
    )

    # 5. Multiple comparisons correction
    if correction == "bonferroni":
        m = len(results_df)
        results_df["p_value_corrected"] = np.minimum(results_df["p_value"] * m, 1.0)
    elif correction == "holm":
        # Holm-Bonferroni
        results_df = results_df.sort_values("p_value")
        m = len(results_df)
        holm_p = []
        for i, p in enumerate(results_df["p_value"]):
            corrected = min(p * (m - i), 1.0)
            holm_p.append(corrected)
        results_df["p_value_corrected"] = holm_p
        results_df = results_df.sort_index()  # return to original order
    else:
        raise ValueError("Correction method must be 'bonferroni' or 'holm'.")

    return results_df


@jit
def compute_anova_f_stat(y, groups, unique_groups):
    """Compute the F-statistic for given data and group labels."""
    overall_mean = jnp.mean(y)
    
    group_sum = segment_sum(y, groups, num_segments=unique_groups.shape[0])
    group_count = segment_sum(jnp.ones_like(y), groups, num_segments=unique_groups.shape[0])
    group_mean = group_sum / group_count

    ss_between = jnp.sum(group_count * (group_mean - overall_mean)**2)

    y_mean_centered = y - group_mean[groups]
    ss_within = jnp.sum(y_mean_centered ** 2)

    n_total = y.shape[0]
    df_between = unique_groups.shape[0] - 1
    df_within = n_total - unique_groups.shape[0]

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F = ms_between / ms_within

    return F, ss_between, ss_within, n_total

def permutation_anova(df, group_col, value_col, num_permutations=1000, seed=42, rank_based=False):
    """
    Perform permutation-based one-way ANOVA (optionally rank-based) with effect size calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str
        Column name for group labels.
    value_col : str
        Column name for the measured values.
    num_permutations : int, optional
        Number of random permutations, by default 1000.
    seed : int, optional
        Random seed for reproducibility.
    rank_based : bool, optional
        If True, perform rank-based permutation ANOVA, by default False.

    Returns
    -------
    dict
        Dictionary with F-statistic, p-value, eta-squared, and omega-squared.
    """

    # Prepare data
    y = df[value_col].values

    # Optional rank-transform
    if rank_based:
        y = rankdata(y)

    y = jnp.array(y)

    # Encode group labels numerically
    group_labels, uniques = pd.factorize(df[group_col])
    groups = jnp.array(group_labels)
    unique_groups = jnp.arange(len(uniques))

    # Observed F-statistic and sums of squares
    F_obs, ss_between, ss_within, n_total = compute_anova_f_stat(y, groups, unique_groups)

    # Effect sizes
    ss_total = ss_between + ss_within
    df_between = unique_groups.shape[0] - 1
    df_within = n_total - unique_groups.shape[0]
    ms_within = ss_within / df_within

    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)

    # Permutation testing
    key = random.PRNGKey(seed)
    keys = random.split(key, num_permutations)

    @jit
    def permuted_f(k):
        permuted_groups = random.permutation(k, groups)
        F_perm, _, _, _ = compute_anova_f_stat(y, permuted_groups, unique_groups)
        return F_perm

    permuted_F = vmap(permuted_f)(keys)

    p_value = jnp.mean(permuted_F >= F_obs)

    return {
        'F_statistic': float(F_obs),
        'p_value': float(p_value),
        'eta_squared': float(eta_squared),
        'omega_squared': float(omega_squared),
        'rank_based': rank_based
    }



@jit
def compute_difference_mean_and_d(group1_vals, group2_vals):
    """Compute absolute difference of means and Cohen's d."""
    diff = jnp.abs(jnp.mean(group1_vals) - jnp.mean(group2_vals))
    pooled_sd = jnp.sqrt((jnp.var(group1_vals, ddof=1) + jnp.var(group2_vals, ddof=1)) / 2)
    d = diff / pooled_sd
    return diff, d

@jit
def compute_mwu_and_r(group1_vals, group2_vals):
    """Compute Mann-Whitney U and rank-biserial correlation."""
    n1 = group1_vals.shape[0]
    n2 = group2_vals.shape[0]
    u = jnp.sum(jnp.expand_dims(group1_vals, 0) > group2_vals[:, None]) + 0.5 * jnp.sum(jnp.expand_dims(group1_vals, 0) == group2_vals[:, None])
    u = jnp.minimum(u, n1 * n2 - u)  # smaller U
    r = 1 - (2 * u) / (n1 * n2)
    return u, r

@jit
def compute_cliff_delta(group1_vals, group2_vals):
    """Compute Cliff's delta."""
    n1 = group1_vals.shape[0]
    n2 = group2_vals.shape[0]
    greater = jnp.sum(jnp.expand_dims(group1_vals, 0) > group2_vals[:, None])
    less = jnp.sum(jnp.expand_dims(group1_vals, 0) < group2_vals[:, None])
    delta = (greater - less) / (n1 * n2)
    return delta, delta  # same value returned as "statistic" and "effect size"

def permutation_posthoc_pairwise(df, group_col, value_col, num_permutations=5000, seed=42, correction='bonferroni', test='mean'):
    """
    Perform permutation-based post-hoc pairwise comparisons between groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str
        Column name for group labels.
    value_col : str
        Column name for measured values.
    num_permutations : int, optional
        Number of permutations, by default 5000.
    seed : int, optional
        Random seed, by default 42.
    correction : str, optional
        'bonferroni' or 'holm', by default 'bonferroni'.
    test : str, optional
        'mean', 'mwu', or 'cliff', by default 'mean'.

    Returns
    -------
    pd.DataFrame
        DataFrame with Group1, Group2, ObservedStatistic, EffectSize, p_value, and p_value_corrected.
    """

    group_labels, uniques = pd.factorize(df[group_col])
    groups = jnp.array(group_labels)
    uniques = np.array(uniques)
    values = jnp.array(df[value_col].values)

    group_ids = jnp.unique(groups)
    results = []
    key = random.PRNGKey(seed)

    for idx, (g1, g2) in enumerate(combinations(group_ids, 2)):
        mask = (groups == g1) | (groups == g2)
        y_pair = values[mask]
        groups_pair = groups[mask]

        group1_vals = y_pair[groups_pair == g1]
        group2_vals = y_pair[groups_pair == g2]

        # Choose which comparison to run
        if test == 'mean':
            observed_stat, effect_size = compute_difference_mean_and_d(group1_vals, group2_vals)
        elif test == 'mwu':
            observed_stat, effect_size = compute_mwu_and_r(group1_vals, group2_vals)
        elif test == 'cliff':
            observed_stat, effect_size = compute_cliff_delta(group1_vals, group2_vals)
        else:
            raise ValueError("Test must be 'mean', 'mwu', or 'cliff'.")

        n_total = y_pair.shape[0]
        n1 = group1_vals.shape[0]
        n2 = group2_vals.shape[0]
        subkey = random.fold_in(key, idx)
        keys = random.split(subkey, num_permutations)

        @jit
        def permute_once(k):
            permuted_idx = random.permutation(k, n_total)
            permuted_y = y_pair[permuted_idx]

            perm_group1 = permuted_y[:n1]
            perm_group2 = permuted_y[n1:n1+n2]

            if test == 'mean':
                perm_stat, _ = compute_difference_mean_and_d(perm_group1, perm_group2)
            elif test == 'mwu':
                perm_stat, _ = compute_mwu_and_r(perm_group1, perm_group2)
            elif test == 'cliff':
                perm_stat, _ = compute_cliff_delta(perm_group1, perm_group2)
            return perm_stat

        permuted_stats = vmap(permute_once)(keys)

        # Correct two-sided permutation
        p_value = jnp.mean(jnp.abs(permuted_stats) >= jnp.abs(observed_stat))

        results.append((uniques[g1], uniques[g2], float(observed_stat), float(effect_size), float(p_value)))

    results_df = pd.DataFrame(results, columns=["Group1", "Group2", "ObservedStatistic", "EffectSize", "p_value"])

    # Multiple comparisons correction
    m = len(results_df)
    if correction == 'bonferroni':
        results_df["p_value_corrected"] = np.minimum(results_df["p_value"] * m, 1.0)
    elif correction == 'holm':
        results_df = results_df.sort_values("p_value")
        holm_p = []
        for i, p in enumerate(results_df["p_value"]):
            corrected = min(p * (m - i), 1.0)
            holm_p.append(corrected)
        results_df["p_value_corrected"] = holm_p
        results_df = results_df.sort_index()
    else:
        raise ValueError("Correction must be 'bonferroni' or 'holm'.")

    return results_df


# PERT and distance functions - commenting this out for now

# #### NOTE PDF DOES NOT RETURN A PROBABILITY - WILL OFTEN BE GREATER THAN 1

# class PERT:
#     """PERT probability distribution.

#     Implements the PERT (Program Evaluation and Review Technique) probability distribution. 
#     Allows specifying the min, mode, and max values to define the distribution.

#     Attributes:
#         a: Minimum value
#         b: Mode value 
#         c: Maximum value
#         lamb: Shape parameter
#         alpha: PERT distribution alpha parameter
#         beta: PERT distribution beta parameter 
#         mean: Mean of distribution
#         var: Variance of distribution

#     Methods:
#         build: Calculates PERT distribution parameters
#         median: Returns median of distribution
#         pdf: Evaluates the probability density function    
#     """
#     def __init__(self, min_val:float, mode:float, max_val:float, lamb:float = 4.0):
#         self.a = min_val
#         self.b = mode
#         self.c = max_val
#         self.lamb = lamb

#         # some checks
#         assert lamb > 0, 'lamb parameter should be greater than 0.'
#         assert self.b > self.a, 'minimum value must be lower than mode.'
#         assert self.c > self.b, 'maximum values must be greater than mode.'
#         assert not isclose(self.a,self.b) | isclose(self.b,self.c) | isclose(self.a,self.c), 'minimum, mode, and maximum values should be different.'

#         self.build()
        
#     def build(self):

                
#         self.alpha = 1 + (self.lamb * ((self.b-self.a) / (self.c-self.a)))
#         self.beta = 1 + (self.lamb * ((self.c-self.b) / (self.c-self.a)))
        
#         self.mean = (self.a + (self.lamb*self.b) + self.c) / (2+self.lamb)
#         self.var = ((self.mean-self.a) * (self.c-self.mean)) / (self.lamb+3)
        
#     @property
#     def range(self):
#         """ Calculates the min-max range
        
#         Returns
#         -------
#         Array:
#             Array of range values of max-min.
#         """
#         return np.asarray(self.c - self.a)
    
#     def median(self) -> float:
#         median = (beta_dist(self.alpha, self.beta).median() * self.range) + self.a
#         return median
    
#     def std(self) -> float:
#         return np.sqrt(self.var)
    
#     def pdf(self, val:np.ndarray) -> np.ndarray:
#         x = ((val - self.a) / self.range).clip(0,1)
#         pdf_val = beta_dist.pdf(x, self.alpha, self.beta) / self.range
#         return pdf_val


# def mahalanobis_distance(x:float, expectancy:float, cov:float, absolute:bool = False) -> float:
#     """Calculates the Mahalanobis distance between a vector x and a distribution
#     with mean expectancy and covariance matrix cov. 

#     NOTE THIS IS WRONG - NEED TO FIX BUT NOT CURRENTLY USING IT.

#     Parameters:
#         x (np.ndarray): Vector to calculate distance for
#         expectancy (np.ndarray): Mean of distribution
#         cov (np.ndarray): Covariance matrix of distribution
#         abs (bool): Whether to take absolute value of distance

#     Returns:
#         dist (np.ndarray): Mahalanobis distance 
#     """
#     dist = (x - expectancy) / cov
#     if absolute:
#         dist = abs(dist)
#     return dist 

# ## Generalised Histogram thresholding


# # If this code is shared with paper publication, add that this essentially copied from GHT paper

# # cumulative sum
# csum = lambda z:np.cumsum(z)[:-1]
# #de-cumulative sum
# dsum = lambda z:np.cumsum(z[::-1])[-2::-1]

# # Use the mean for ties .
# argmax = lambda x , f:np.mean(x[:-1][f==np.max(f)]) 
# clip = lambda z:np.maximum(1e-30,z)

# def image_hist(N, range = None):
#     """ Generate bins and counts """
#     image = N.ravel().astype(int)
#     n = np.bincount(image,minlength = np.max(image) - np.min(image) + 1)
#     if range != None:
#         if len(range) != 2:
#             raise TypeError('Range is not length 2')
#         else:
#             x = np.arange(range[0], range[1])    
#     elif range == None:
#         x = np.arange(0,np.max(image) + 1)
#     else:
#         raise TypeError("range input type not supported")
    
#     return n,x

# def preliminaries(n, x):
#     """ Some math that is shared across each algorithm - refering to GHT, weighted percentile and Otsu"""
#     assert np . all(n >= 0)
#     x = np.arange(len(n), dtype=n.dtype) if x is None else x
#     assert np . all(x[1:] >= x[: -1])
#     w0 = clip(csum(n))
#     w1 = clip(dsum(n))
#     p0 = w0/(w0 + w1)
#     p1 = w1/(w0 + w1)
#     mu0 = csum(n*x)/w0
#     mu1 = dsum(n*x)/w1
#     d0 = csum(n*x**2)-w0*mu0**2
#     d1 = dsum(n*x**2)-w1*mu1**2
#     return x, w0, w1, p0, p1, mu0, mu1, d0, d1


# def GHT(n, x, nu=None, tau=0.1, kappa=0.1, omega=0.5):
#     """ GHT implementation"""
#     if nu == None:
#         nu = abs(len(x)/2)
#     assert nu >= 0
#     assert tau >= 0
#     assert kappa >= 0
#     assert omega >= 0 and omega <= 1
#     x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
#     v0 = clip((p0*nu*tau**2+d0)/(p0*nu+w0))
#     v1 = clip((p1*nu*tau**2+d1)/(p1*nu+w1))
#     f0 = - d0/v0-w0*np.log(v0) + 2*(w0+kappa*omega)*np.log(w0)
#     f1 = - d1/v1-w1*np.log(v1)+2*\
#         (w1+kappa*(1-omega))*np.log(w1)
#     return argmax(x, f0+f1), f0+f1

# ###




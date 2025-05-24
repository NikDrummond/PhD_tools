import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any, Callable, Dict
from scipy.stats import norm
import warnings  # For more controllable warnings


def compute_grid(
    data: np.ndarray,
    x0: Optional[float],
    x1: Optional[float],
    bins: Optional[int],
    buffer: float,
) -> np.ndarray:
    """Computes a data-driven grid of points.

    The grid is typically used for evaluating distributions (ECDF, PDF, etc.).
    If `bins` is not provided, Freedman-Diaconis rule is used to determine
    an appropriate number of bins. A buffer can be added to the data range.

    Parameters
    ----------
    data : np.ndarray
        The input data array used to determine grid boundaries and/or bin widths.
    x0 : Optional[float]
        Pre-defined lower bound for the grid. If None, derived from data minimum.
    x1 : Optional[float]
        Pre-defined upper bound for the grid. If None, derived from data maximum.
    bins : Optional[int]
        The number of points (bins) in the grid. If None, determined automatically.
    buffer : float
        Fraction of the data range (x1 - x0) to add as a buffer on
        each side of the grid. E.g., 0.05 for a 5% buffer.

    Returns
    -------
    np.ndarray
        A 1D array representing the computed grid points (typically bin centers).

    Notes
    -----
    The Freedman-Diaconis rule for bin width is `2 * IQR / n^(1/3)`.
    A minimum of 10 bins is enforced if `bins` is auto-calculated.
    Handles cases with zero IQR or very small data sizes by defaulting to
    a reasonable number of bins.
    """
    data = np.asarray(data)
    if data.size == 0:
        data_min, data_max = 0.0, 1.0
        if x0 is None and x1 is None:
            warnings.warn(
                "compute_grid called with empty data and no x0/x1. Defaulting to [0,1] range.",
                UserWarning,
            )
    else:
        data_min, data_max = np.min(data), np.max(
            data
        )  # np.min/max on empty array would error

    x0_eff = x0 if x0 is not None else data_min
    x1_eff = x1 if x1 is not None else data_max

    if np.isclose(x1_eff, x0_eff):
        if abs(x0_eff) > 1e-3:
            x1_eff = x0_eff * 1.01 if x0_eff != 0 else 0.01
        else:
            x1_eff = x0_eff + 0.01
        if np.isclose(x1_eff, x0_eff):
            x1_eff = x0_eff + 1e-6

    range_val = x1_eff - x0_eff
    buffer_val = buffer * range_val
    x0_final = x0_eff - buffer_val
    x1_final = x1_eff + buffer_val

    if np.isclose(x1_final, x0_final):
        x1_final = x0_final + 1e-6

    if bins is None:
        if data.size < 4:
            bins = 10
        else:
            iqr = np.subtract(*np.percentile(data, [75, 25]))
            cbrt_n = np.cbrt(data.size)

            if iqr > 1e-9 and cbrt_n > 1e-9:
                bin_width = 2 * iqr / cbrt_n
                current_range = x1_final - x0_final
                if bin_width > 1e-9 and current_range > 1e-9:
                    bins = max(10, int(np.ceil(current_range / bin_width)))
                elif current_range <= 1e-9:
                    bins = 10
                else:
                    bins = max(
                        10, min(50, int(data.size / 2)) if data.size >= 4 else 10
                    )
            else:
                bins = max(
                    10, min(50, int(np.sqrt(data.size)) if data.size > 0 else 10)
                )
        if bins <= 0:
            bins = 10

    return np.linspace(x0_final, x1_final, bins)


def dkw_ci(N: int, alpha: float) -> float:
    """Computes the Dvoretzky-Kiefer-Wolfowitz (DKW) confidence interval epsilon."""
    if N <= 0:
        return np.inf
    return np.sqrt(np.log(2.0 / alpha) / (2 * N))


def asymptotic_ci(p: np.ndarray, N: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes asymptotic (Wald) confidence intervals for a proportion."""
    if N <= 0:
        nan_arr = np.full_like(p, np.nan, dtype=float)
        return nan_arr, nan_arr

    z = norm.ppf(1 - alpha / 2)
    p_safe = np.clip(p, 1e-9, 1 - 1e-9)
    se = np.sqrt(p_safe * (1 - p_safe) / N)
    margin = z * se
    return np.clip(p - margin, 0, 1), np.clip(p + margin, 0, 1)


def edf_numpy(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Computes the Empirical Distribution Function (ECDF) using NumPy."""
    if data.size == 0 or x.size == 0:
        return np.full_like(x, np.nan, dtype=float)
    return np.searchsorted(np.sort(data), x, side="right") / data.size


def survival_numpy(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Computes the empirical survival function using NumPy."""
    return 1.0 - edf_numpy(data, x)


def histogram_numpy(
    data: np.ndarray, x_centers: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Computes histogram counts and bin width from bin centers."""
    if x_centers.size == 0:
        raise ValueError("x_centers must not be empty for histogram_numpy.")

    bin_width: float
    edges: np.ndarray
    if x_centers.size == 1:
        half_width = 0.5
        if data.size > 1 and not np.all(np.isclose(data, data[0])):
            data_range = np.max(data) - np.min(data)
            half_width = max(
                0.5 * abs(x_centers[0]) if abs(x_centers[0]) > 1e-3 else 0.5,
                data_range / 2.0 if data_range > 1e-9 else 0.5,
            )

        edges = np.array([x_centers[0] - half_width, x_centers[0] + half_width])
        bin_width = edges[1] - edges[0]
        if bin_width <= 1e-9:
            bin_width = 1.0
    else:
        current_diffs = np.diff(x_centers)
        if np.all(
            np.isclose(current_diffs, 0)
        ):  # All centers are the same or very close
            ref_center = np.mean(x_centers)
            half_width = 0.5 * abs(ref_center) if abs(ref_center) > 1e-3 else 0.5
            bin_width = 2 * half_width
            edges = np.array([ref_center - half_width, ref_center + half_width])
            # For np.histogram, if all centers are same, it needs more than one edge if counts are desired per original center
            # This case means the grid itself is degenerate for distinct bins.
            # We'll make one big bin for simplicity if all centers are same, or use the many-center logic if they slightly differ.
            if x_centers.size > 1:  # If multiple identical centers were passed
                edges = np.zeros(x_centers.size + 1)
                edges[0] = x_centers[0] - half_width
                for i in range(x_centers.size - 1):
                    edges[i + 1] = (x_centers[i] + x_centers[i + 1]) / 2.0 + (
                        1e-7 * i
                    )  # Ensure edges are increasing
                edges[-1] = x_centers[-1] + half_width
                bin_width = np.mean(np.diff(edges))

        else:  # Centers are distinct
            bin_width = np.mean(
                current_diffs[current_diffs > 1e-9]
            )  # Avg of positive diffs
            if np.isnan(bin_width) or bin_width <= 1e-9:
                bin_width = 1.0  # Fallback if all diffs were zero

            edges = np.zeros(x_centers.size + 1)
            half_bw = bin_width / 2.0
            edges[0] = x_centers[0] - half_bw
            edges[1:-1] = (x_centers[:-1] + x_centers[1:]) / 2.0
            edges[-1] = x_centers[-1] + half_bw

    # Ensure edges are monotonically increasing
    sorted_edges = np.sort(np.unique(edges))  # Get unique sorted edges
    if len(sorted_edges) < 2:  # Not enough distinct edges to form bins
        # Fallback: create a single bin encompassing the data or centers
        min_val = x_centers.min() - 0.5
        max_val = x_centers.max() + 0.5
        if np.isclose(min_val, max_val):
            max_val = min_val + 1.0
        sorted_edges = np.array([min_val, max_val])
        bin_width = sorted_edges[1] - sorted_edges[0]

    counts, final_edges = np.histogram(data, bins=sorted_edges)
    # If counts array doesn't match number of centers (due to unique sorted edges)
    # this indicates an issue with x_centers. For now, assume counts aligns with x_centers if bins were derived from it.
    # A more robust approach might involve interpolation if shapes mismatch.
    # However, np.histogram with explicit edges should align counts with num_edges-1.
    # If len(counts) != len(x_centers), something is off.
    # Let's re-evaluate effective bin_width from final_edges used by histogram
    if len(final_edges) > 1:
        final_bin_widths = np.diff(final_edges)
        # If a single bin was used, counts will be len 1.
        # If x_centers was len 1, this is fine.
        # If x_centers was > 1 but all identical, counts might be 1.
        # We need counts to be len(x_centers)
        if len(counts) != len(x_centers):
            # This scenario suggests the initial x_centers were problematic for forming distinct bins.
            # We might need to re-bin or return an error/warning.
            # For now, if counts are fewer than x_centers, it implies some centers were collapsed.
            # This is complex to resolve generically without knowing user intent for degenerate x_centers.
            # A simple approach: if counts has 1 element and x_centers > 1, distribute count or error.
            # Let's assume for now histogram_numpy is called with well-behaved x_centers from compute_grid.
            # The average width of the bins actually used:
            bin_width = np.mean(final_bin_widths) if final_bin_widths.size > 0 else 1.0
            if (
                len(counts) == 1 and len(x_centers) > 1
            ):  # Single bin resulted from edges for multiple centers
                # This can happen if all x_centers were identical.
                # Distribute the single count to the first center, others zero
                new_counts = np.zeros_like(x_centers, dtype=counts.dtype)
                new_counts[0] = counts[0]
                counts = new_counts

    elif len(final_edges) == 1:  # Only one edge, no bins
        bin_width = 1.0
        counts = np.zeros_like(x_centers, dtype=int)
    else:  # empty final_edges
        bin_width = 1.0
        counts = np.zeros_like(x_centers, dtype=int)

    if bin_width <= 1e-9:
        bin_width = 1.0

    return counts, bin_width


# --- Per-ID Matrix Construction ---
def binned_matrix(
    df: pd.DataFrame, y_col: str, id_col: str, x_centers: np.ndarray, kind: str
) -> Tuple[np.ndarray, List[Any]]:
    """Constructs a matrix of binned distributions, one column per ID."""
    if x_centers.size == 0:
        raise ValueError("x_centers must not be empty for binned_matrix.")

    bin_width: float
    edges: np.ndarray
    if x_centers.size == 1:
        half_width = 0.5
        edges = np.array([x_centers[0] - half_width, x_centers[0] + half_width])
        bin_width = edges[1] - edges[0]
        if bin_width <= 1e-9:
            bin_width = 1.0
    else:
        current_diffs = np.diff(x_centers)
        # Check if all centers are effectively the same
        if np.all(np.isclose(current_diffs, 0)):
            ref_center = np.mean(x_centers)
            half_width = 0.5 * abs(ref_center) if abs(ref_center) > 1e-3 else 0.5
            bin_width = 2 * half_width  # Effective width for a single point
            # Create edges that treat this as one effective bin for digitize, but map to multiple later if needed
            edges = np.array([ref_center - half_width, ref_center + half_width])
        else:
            # Create edges assuming x_centers are distinct midpoints
            half_widths = np.diff(x_centers) / 2.0
            # Ensure half_widths has elements if x_centers has at least 2 points
            hw_first = (
                half_widths[0]
                if half_widths.size > 0
                else (0.5 if x_centers.size > 0 else 0)
            )
            hw_last = (
                half_widths[-1]
                if half_widths.size > 0
                else (0.5 if x_centers.size > 0 else 0)
            )

            edges = np.concatenate(
                (
                    [x_centers[0] - hw_first],
                    (x_centers[:-1] + x_centers[1:]) / 2.0,  # Midpoints
                    [x_centers[-1] + hw_last],
                )
            )
            bin_width = np.mean(np.diff(edges))
            if bin_width <= 1e-9:
                bin_width = 1.0

    # Ensure edges are monotonically increasing for np.digitize
    edges = np.sort(np.unique(edges))
    if len(edges) < 2:  # Fallback if edges collapse
        min_val = x_centers.min() - 0.5
        max_val = x_centers.max() + 0.5
        if np.isclose(min_val, max_val):
            max_val = min_val + 1.0
        edges = np.array([min_val, max_val])
        bin_width = edges[1] - edges[0]

    y_data = df[y_col].to_numpy()
    # np.digitize needs bins to be monotonically increasing.
    bin_indices_for_digitize = np.digitize(y_data, edges)

    # Map these indices to 0 to len(x_centers)-1
    # If edges were constructed directly from x_centers, len(edges) = len(x_centers)+1
    # bin_idx from digitize are 1-based for edge array. We subtract 1 for 0-based.
    # Then clip to ensure it's within [0, len(x_centers)-1]
    bin_idx = np.clip(bin_indices_for_digitize - 1, 0, len(x_centers) - 1)

    id_vals = df[id_col].to_numpy()
    id_labels, id_idx_inverse = np.unique(id_vals, return_inverse=True)
    n_bins, n_ids = len(x_centers), len(id_labels)

    if n_ids == 0:
        return np.zeros((n_bins, 0), dtype=float), []

    mat = np.zeros((n_bins, n_ids), dtype=float)
    np.add.at(mat, (bin_idx, id_idx_inverse), 1)

    # For normalization, norm is sum of counts per ID
    norm_per_id = np.sum(mat, axis=0, keepdims=True)  # Shape (1, n_ids)

    # --- This is the critical section for the fix ---
    # Create a 1D boolean mask for columns where norm is 0
    zero_norm_cols_mask_1d = (norm_per_id == 0).flatten()  # Shape (n_ids,)

    if kind == "cumulative":
        mat_cum = np.cumsum(mat, axis=0)
        # Avoid division by zero: where norm_per_id is 0, keep original (cumulative sums of 0s are 0s)
        # For division, use 1 where norm_per_id is 0.
        norm_div = np.where(norm_per_id == 0, 1.0, norm_per_id)
        mat = mat_cum / norm_div
        mat[:, zero_norm_cols_mask_1d] = np.nan  # Set columns with no data to NaN
    elif kind in ("mass", "density"):
        norm_per_id = np.sum(mat, axis=0, keepdims=True)
        zero_norm_cols_mask_1d = (norm_per_id == 0).flatten()
        norm_div = np.where(norm_per_id == 0, 1.0, norm_per_id)
        if kind == "density":
            if bin_width <= 1e-9:
                mat.fill(np.nan)
                warnings.warn(
                    f"Bin width for density in binned_matrix is near zero. Results for density might be all NaNs.",
                    UserWarning,
                )
            else:
                mat = mat / (norm_div * bin_width)
        else:  # kind == "mass"
            mat = mat / norm_div

        mat[:, zero_norm_cols_mask_1d] = np.nan  # Set columns with no data to NaN
    elif kind == "survival":
        mat_cum_surv = np.cumsum(mat[::-1, :], axis=0)[::-1, :]
        norm_div = np.where(norm_per_id == 0, 1.0, norm_per_id)
        mat = mat_cum_surv / norm_div
        mat[:, zero_norm_cols_mask_1d] = np.nan
    else:
        raise ValueError(f"Unknown kind for binned_matrix: {kind}")

    return mat, id_labels.tolist()


# --- Bootstrap over IDs ---
def bootstrap_ci_id(
    mat: np.ndarray, alpha: float, samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes bootstrap confidence intervals for the mean across ID columns."""
    if mat.ndim != 2 or mat.shape[1] == 0:
        nan_arr = np.full(mat.shape[0] if mat.ndim == 2 else 0, np.nan, dtype=float)
        return nan_arr, nan_arr
    if mat.shape[1] < 10:
        warnings.warn(
            f"Number of IDs ({mat.shape[1]}) for bootstrap_ci_id is small; CIs may be unstable.",
            UserWarning,
        )

    n_bins, n_ids = mat.shape
    boot_means = np.zeros((n_bins, samples), dtype=float)

    for b in range(samples):
        resample_idx = rng.integers(0, n_ids, size=n_ids)
        boot_means[:, b] = np.nanmean(mat[:, resample_idx], axis=1)  # Use nanmean

    # error
    # lower_ci = np.percentile(boot_means[~np.isnan(boot_means).all(axis=0)], 100 * alpha / 2, axis=1) if samples > 0 else np.full(n_bins, np.nan)
    # upper_ci = np.percentile(boot_means[~np.isnan(boot_means).all(axis=0)], 100 * (1 - alpha / 2), axis=1) if samples > 0 else np.full(n_bins, np.nan)
    if samples > 0:
        lower_ci = np.nanpercentile(boot_means, 100 * alpha / 2, axis=1)
        upper_ci = np.nanpercentile(boot_means, 100 * (1 - alpha / 2), axis=1)
    else:
        lower_ci = np.full(n_bins, np.nan, dtype=float)
        upper_ci = np.full(n_bins, np.nan, dtype=float)

    # The comments about np.percentile of all NaNs being NaN are still relevant for np.nanpercentile.
    # If an entire row of boot_means (all samples for a specific bin) is NaN,
    # np.nanpercentile will correctly return NaN for that bin's CI.

    return lower_ci, upper_ci
    # Handle cases where all bootstrap means for a bin might be NaN
    # (e.g. if all IDs in resamples had NaN for that bin)
    # np.percentile of all NaNs is NaN, which is fine.

    return lower_ci, upper_ci


# --- Bootstrap for a generic statistic (non-ID case) ---
def _bootstrap_ci_statistic(
    data_values: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    x_grid: np.ndarray,
    alpha: float,
    samples: int,
    rng: np.random.Generator,
    **statistic_fn_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes bootstrap CIs for a statistic applied to resampled data."""
    if data_values.size == 0:
        nan_arr = np.full_like(x_grid, np.nan, dtype=float)
        return nan_arr, nan_arr
    if data_values.size < 10:
        warnings.warn(
            f"Sample size ({data_values.size}) for _bootstrap_ci_statistic is small; CIs may be unstable.",
            UserWarning,
        )

    boot_stats_list = []  # Use list to handle potential varying NaN patterns
    for _ in range(samples):
        resample_data = rng.choice(data_values, size=data_values.size, replace=True)
        stat = statistic_fn(resample_data, x_grid, **statistic_fn_kwargs)
        boot_stats_list.append(stat)

    # Stack, being careful about all-NaN slices if statistic_fn can return them
    try:
        boot_stats_arr = np.array(boot_stats_list, dtype=float)
    except (
        ValueError
    ):  # Happens if arrays have inconsistent shapes (should not with this logic)
        # Fallback: try to make a common shape, padding with NaNs if needed
        # This is complex; for now, assume statistic_fn returns consistent shapes or NaNs
        max_len = 0
        if x_grid is not None:
            max_len = len(x_grid)
        else:
            max_len = max(len(s) for s in boot_stats_list if hasattr(s, "__len__"))

        boot_stats_arr = np.full((len(boot_stats_list), max_len), np.nan, dtype=float)
        for i, stat_arr in enumerate(boot_stats_list):
            if hasattr(stat_arr, "__len__") and len(stat_arr) <= max_len:
                boot_stats_arr[i, : len(stat_arr)] = stat_arr
            elif not hasattr(stat_arr, "__len__") and max_len == 1:  # scalar
                boot_stats_arr[i, 0] = stat_arr

    # Calculate percentiles ignoring columns (axis=0) that are all NaN
    # np.nanpercentile handles NaNs within each column (percentile calculation)
    lower_ci = np.nanpercentile(boot_stats_arr, 100 * alpha / 2, axis=0)
    upper_ci = np.nanpercentile(boot_stats_arr, 100 * (1 - alpha / 2), axis=0)

    return lower_ci, upper_ci


# --- Envelope Bands ---
def compute_envelope(
    arr: np.ndarray,
    method: str = "percentile",
    q_lower: float = 0.025,
    q_upper: float = 0.975,
    mad_scale: float = 1.4826,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes envelope bands for a set of curves."""
    if arr.ndim != 2 or arr.shape[1] == 0:
        nan_arr = np.full(arr.shape[0] if arr.ndim == 2 else 0, np.nan, dtype=float)
        return nan_arr, nan_arr

    if method == "percentile":
        return np.nanpercentile(arr, 100 * q_lower, axis=1), np.nanpercentile(
            arr, 100 * q_upper, axis=1
        )
    elif method == "minmax":
        return np.nanmin(arr, axis=1), np.nanmax(arr, axis=1)
    elif method == "mad":
        med = np.nanmedian(arr, axis=1)
        abs_dev = np.abs(arr - med[:, None])
        mad_val = np.nanmedian(abs_dev, axis=1)
        lower_band = med - mad_scale * mad_val
        return np.clip(lower_band, 0, None), med + mad_scale * mad_val
    elif method == "asymmetric_mad":
        med = np.nanmedian(arr, axis=1)  # Median per row (bin)
        mad_lower_vals = np.full_like(med, np.nan, dtype=float)
        mad_upper_vals = np.full_like(med, np.nan, dtype=float)

        for i in range(arr.shape[0]):
            current_row_values = arr[i, :]
            current_median = med[i]  # This median might be NaN if all in row are NaN

            if np.isnan(current_median):  # If median is NaN, MADs are NaN
                continue

            # Filter out NaNs before comparison for deviations
            valid_row_values = current_row_values[~np.isnan(current_row_values)]
            if valid_row_values.size == 0:
                continue

            values_below_median = valid_row_values[valid_row_values < current_median]
            if values_below_median.size > 0:
                mad_lower_vals[i] = np.nanmedian(
                    np.abs(values_below_median - current_median)
                )

            values_at_or_above_median = valid_row_values[
                valid_row_values >= current_median
            ]
            if values_at_or_above_median.size > 0:
                mad_upper_vals[i] = np.nanmedian(
                    np.abs(values_at_or_above_median - current_median)
                )

        lower_band = med - mad_scale * mad_lower_vals
        upper_band = med + mad_scale * mad_upper_vals
        return np.clip(lower_band, 0, None), upper_band
    else:
        raise ValueError(f"Unknown envelope method: {method}")


# --- Main API ---
class DistributionResult:
    """Stores and plots the result of a distribution estimation."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        group_labels: List[Any],
        kind: str,
        y_col_name: str,
        env_l: Optional[np.ndarray] = None,
        env_u: Optional[np.ndarray] = None,
        log_data: bool = False,
    ):
        self.x = x
        self.y = y
        self.l = l
        self.u = u
        self.group_labels = group_labels
        self.kind = kind
        self.y_col_name = y_col_name
        self.env_l = env_l
        self.env_u = env_u
        self.log_data = log_data
        self.log_note = (
            f"Data ('{self.y_col_name}') were log-transformed before analysis."
            if log_data
            else f"Data ('{self.y_col_name}') were used in original units."
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        colors: Optional[List[Any]] = None,
        ci_alpha: float = 0.3,
        env_alpha: float = 0.15,
        show_envelope: bool = True,
        plot_log_x: bool = False,
        plot_log_y: bool = False,
        group_labels: Optional[List[str]] = None,  # Added parameter
        plot_kwargs: Optional[Dict[str, Any]] = None,  # Added parameter
        fill_kwargs: Optional[Dict[str, Any]] = None,  # Added parameter
        legend_kwargs: Optional[Dict[str, Any]] = None,  # Added parameter
        **kwargs,  # Kept for backward compatibility or other potential uses
    ) -> plt.Axes:
        """
        Plots the estimated distribution(s).

        Parameters
        ----------
        ax : Optional[plt.Axes], default=None
            The matplotlib Axes object to plot on. If None, a new figure and Axes
            are created.
        colors : Optional[List[Any]], default=None
            A list of colors to use for plotting each group. If None, the default
            matplotlib color cycle is used.
        ci_alpha : float, default=0.3
            The alpha (transparency) value for the confidence interval fill.
        env_alpha : float, default=0.15
            The alpha (transparency) value for the envelope fill.
        show_envelope : bool, default=True
            Whether to plot the envelope (e.g., prediction interval).
        plot_log_x : bool, default=False
            If True, the x-axis will be set to a logarithmic scale.
        plot_log_y : bool = False
            If True, the y-axis will be set to a logarithmic scale.
        group_labels : Optional[List[str]], default=None
            Manually set the labels for each plotted group. If None, the labels
            from `self.group_labels` will be used.
        plot_kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments passed directly to `ax.plot()` for the
            main distribution lines. For example: `{'linestyle': '--', 'linewidth': 2}`.
        fill_kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments passed directly to `ax.fill_between()` for
            both confidence intervals and envelopes. These will override `ci_alpha`
            and `env_alpha` if `alpha` is specified within `fill_kwargs`.
            For example: `{'edgecolor': 'blue', 'alpha': 0.2}`.
        legend_kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments passed directly to `ax.legend()`.
            For example: `{'frameon': False, 'loc': 'upper left'}`.
        **kwargs
            Additional keyword arguments. These are primarily kept for potential
            future extensions or specific uses not covered by `plot_kwargs`
            or `fill_kwargs`. Note that if keys overlap with `plot_kwargs`
            or `fill_kwargs`, the dedicated dictionaries will take precedence.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object with the plotted distribution(s).

        Raises
        ------
        ValueError
            If `plot_log_x` is True and x-values (possibly already log-transformed)
            contain non-positive entries.
        """
        if ax is None:
            _, ax = plt.subplots()

        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Initialize kwargs dictionaries
        plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        fill_kwargs = fill_kwargs if fill_kwargs is not None else {}
        legend_kwargs = legend_kwargs if legend_kwargs is not None else {}

        # Use provided group_labels or fall back to self.group_labels
        current_group_labels = (
            group_labels if group_labels is not None else self.group_labels
        )

        y_plot = self.y if self.y.ndim == 2 else self.y[:, np.newaxis]
        l_plot = self.l if self.l.ndim == 2 else self.l[:, np.newaxis]
        u_plot = self.u if self.u.ndim == 2 else self.u[:, np.newaxis]

        env_l_plot = None
        if self.env_l is not None:
            env_l_plot = (
                self.env_l if self.env_l.ndim == 2 else self.env_l[:, np.newaxis]
            )
        env_u_plot = None
        if self.env_u is not None:
            env_u_plot = (
                self.env_u if self.env_u.ndim == 2 else self.env_u[:, np.newaxis]
            )

        for i, label in enumerate(current_group_labels):
            if i >= y_plot.shape[1]:  # Should not happen if data constructed correctly
                warnings.warn(
                    f"Group label index {i} exceeds available data columns ({y_plot.shape[1]}). Skipping plot for '{label}'.",
                    UserWarning,
                )
                continue

            color = colors[i % len(colors)]
            ax.plot(self.x, y_plot[:, i], label=str(label), color=color, **plot_kwargs)

            # Only plot fill_between if CIs are not all NaN for this group
            if not np.all(np.isnan(l_plot[:, i])) and not np.all(
                np.isnan(u_plot[:, i])
            ):
                ci_fill_kwargs = {
                    "alpha": ci_alpha,
                    "color": color,
                    "edgecolor": "none",
                }
                ci_fill_kwargs.update(fill_kwargs)  # fill_kwargs can override defaults
                ax.fill_between(
                    self.x,
                    l_plot[:, i],
                    u_plot[:, i],
                    **ci_fill_kwargs,
                )

            if (
                show_envelope
                and env_l_plot is not None
                and env_u_plot is not None
                and i < env_l_plot.shape[1]
                and i < env_u_plot.shape[1]
                and not np.all(np.isnan(env_l_plot[:, i]))
                and not np.all(np.isnan(env_u_plot[:, i]))
            ):
                env_fill_kwargs = {
                    "alpha": env_alpha,
                    "color": color,
                    "linestyle": "--",
                    "edgecolor": "none",
                }
                env_fill_kwargs.update(fill_kwargs)  # fill_kwargs can override defaults
                ax.fill_between(
                    self.x,
                    env_l_plot[:, i],
                    env_u_plot[:, i],
                    **env_fill_kwargs,
                )

        xlabel_text = self.y_col_name
        if self.log_data:
            xlabel_text = f"log({self.y_col_name})"

        if plot_log_x:
            if self.log_data:
                warnings.warn(
                    f"Plotting log of x-axis which represents log-transformed data: log(log({self.y_col_name})). Ensure this is intended.",
                    UserWarning,
                )
            if np.any(self.x[~np.isnan(self.x)] <= 0):  # Check non-NaN values
                raise ValueError(
                    "Cannot use log-x axis: x-values (possibly already log-transformed) "
                    "contain non-positive entries."
                )
            ax.set_xscale("log")

        ax.set_xlabel(xlabel_text)

        if plot_log_y:
            plottable_y_data = [
                arr for arr in [y_plot, l_plot, u_plot] if arr is not None
            ]
            if any(np.any(arr_y[~np.isnan(arr_y)] <= 0) for arr_y in plottable_y_data):
                warnings.warn(
                    "Some y-values or CI bounds are non-positive. Log-y axis may clip or omit these.",
                    UserWarning,
                )
            ax.set_yscale("log")

        ylabel_text = self.kind.capitalize()
        ax.set_ylabel(ylabel_text)

        if current_group_labels and (
            len(current_group_labels) > 1
            or (len(current_group_labels) == 1 and current_group_labels[0] != "all")
        ):
            ax.legend(**legend_kwargs)  # Pass legend_kwargs here
        ax.set_title(self.log_note)
        return ax


def compute_distribution(
    df: pd.DataFrame,
    y_col: str,
    kind: str = "cumulative",
    group_col: Optional[str] = None,
    id_col: Optional[str] = None,
    ci_method: str = "bootstrap",
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    envelope_method: str = "percentile",
    envelope_quantiles: Tuple[float, float] = (0.025, 0.975),
    envelope_mad_scale: float = 1.4826,
    x_grid_input: Optional[np.ndarray] = None,
    x0: Optional[float] = None,
    x1: Optional[float] = None,
    bins: Optional[int] = None,
    weight_by_obs: bool = False,
    log_data: bool = False,
    grid_buffer: float = 0.0,
) -> DistributionResult:
    """Computes a distribution with confidence intervals and optional envelopes."""
    df_proc = df.copy()

    if y_col not in df_proc.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")

    if df_proc[y_col].isnull().any():
        warnings.warn(
            f"NaNs found in '{y_col}'. Rows with NaN '{y_col}' will be dropped for this analysis.",
            UserWarning,
        )
        df_proc.dropna(subset=[y_col], inplace=True)

    if df_proc.empty:
        warnings.warn(
            f"DataFrame is empty after processing '{y_col}'. Returning empty/NaN result.",
            UserWarning,
        )
        dummy_x_grid = (
            x_grid_input
            if x_grid_input is not None and x_grid_input.size > 0
            else np.array([0.0, 1.0])
        )
        num_groups_expected = (
            len(df[group_col].unique())
            if group_col and df[group_col].nunique() > 0
            else 1
        )

        return DistributionResult(
            x=dummy_x_grid,
            y=np.full((dummy_x_grid.size, num_groups_expected), np.nan),
            l=np.full((dummy_x_grid.size, num_groups_expected), np.nan),
            u=np.full((dummy_x_grid.size, num_groups_expected), np.nan),
            group_labels=(
                df[group_col].unique().tolist()
                if group_col and df[group_col].nunique() > 0
                else ["all"]
            ),
            kind=kind,
            y_col_name=y_col,
            log_data=log_data,
        )

    series_for_grid_computation = df_proc[y_col].to_numpy()

    x_grid: np.ndarray
    # Grid handling logic:
    # 1. If x_grid_input is provided, use it.
    # 2. If log_data is True, data in df_proc[y_col] is transformed.
    #    x_grid_input, if provided, should be on this transformed scale.
    #    If x_grid_input is NOT provided, x0/x1 (original scale) are transformed, then grid is computed.

    _log_data_applied_to_df_proc = False
    if log_data:
        # Check original data for loggability before transforming df_proc[y_col]
        # series_for_grid_computation currently holds original non-logged data if x_grid_input is None
        # OR it holds already logged data if x_grid_input is given AND log_data was True (ambiguous intent by user)

        # To avoid double logging or logging user-provided transformed grid values:
        # Apply log to df_proc[y_col] only ONCE, and do it before grid decision if grid is auto.
        if np.any(
            df_proc[y_col].to_numpy(dtype=float) <= 0
        ):  # Check current state of y_col
            raise ValueError(
                f"Cannot log-transform data in column '{y_col}': it contains non-positive values."
            )
        df_proc[y_col] = np.log(df_proc[y_col].to_numpy(dtype=float))
        _log_data_applied_to_df_proc = True
        # If grid is to be computed, update its source
        if x_grid_input is None:
            series_for_grid_computation = df_proc[y_col].to_numpy()

    if x_grid_input is not None:
        x_grid = np.asarray(x_grid_input)
        if x_grid.ndim != 1 or x_grid.size == 0:
            raise ValueError("x_grid_input must be a non-empty 1D array.")
        # If log_data, assume x_grid_input is already on the log scale
    else:  # Compute grid internally
        current_x0, current_x1 = x0, x1
        if (
            log_data and not _log_data_applied_to_df_proc
        ):  # This case should not be hit due to above logic
            # This implies x_grid_input was None, log_data True. series_for_grid_computation is original.
            # Transform x0, x1 for grid computation.
            if current_x0 is not None:
                if current_x0 <= 0:
                    raise ValueError("x0 must be positive when log_data=True.")
                current_x0 = np.log(current_x0)
            if current_x1 is not None:
                if current_x1 <= 0:
                    raise ValueError("x1 must be positive when log_data=True.")
                current_x1 = np.log(current_x1)
        elif (
            log_data and _log_data_applied_to_df_proc
        ):  # x_grid_input was None, log_data True, df_proc[y_col] is logged
            # x0, x1 (if provided) need to be logged for compute_grid
            if current_x0 is not None:
                if x0 <= 0:
                    raise ValueError(
                        "Original x0 must be positive when log_data=True."
                    )  # Check original x0
                current_x0 = np.log(x0)  # Log original x0
            if current_x1 is not None:
                if x1 <= 0:
                    raise ValueError(
                        "Original x1 must be positive when log_data=True."
                    )  # Check original x1
                current_x1 = np.log(x1)  # Log original x1

        x_grid = compute_grid(
            series_for_grid_computation, current_x0, current_x1, bins, grid_buffer
        )

    if x_grid.size == 0:
        raise RuntimeError("Computed x_grid is empty. This should not happen.")

    rng = np.random.default_rng()
    group_labels_list = df_proc[group_col].unique().tolist() if group_col else ["all"]
    if not group_labels_list:
        group_labels_list = ["all"]

    y_out, l_out, u_out = [], [], []
    env_l_out, env_u_out = [], []

    for g_label in group_labels_list:
        df_g = (
            df_proc[df_proc[group_col] == g_label]
            if group_col and g_label != "all"
            else df_proc
        )

        y_g = np.full_like(x_grid, np.nan, dtype=float)
        l_g = np.full_like(x_grid, np.nan, dtype=float)
        u_g = np.full_like(x_grid, np.nan, dtype=float)
        current_env_l = np.full_like(x_grid, np.nan, dtype=float)
        current_env_u = np.full_like(x_grid, np.nan, dtype=float)

        if (
            df_g.empty or df_g[y_col].isnull().all()
        ):  # Check if group is empty or all NaNs for y_col
            warnings.warn(
                f"No valid data available for group '{g_label}'. Results for this group will be NaN.",
                UserWarning,
            )
        elif id_col:
            mat, id_labels_from_binned_matrix = binned_matrix(
                df_g, y_col, id_col, x_grid, kind
            )
            if mat.ndim != 2 or mat.shape[1] == 0:
                warnings.warn(
                    f"No valid IDs or data processed for group '{g_label}' with id_col '{id_col}'. Results for this group will be NaN.",
                    UserWarning,
                )
            else:
                if weight_by_obs:
                    obs_counts = df_g.groupby(id_col)[y_col].count()
                    aligned_weights = (
                        obs_counts.reindex(id_labels_from_binned_matrix)
                        .fillna(0)
                        .to_numpy()
                    )
                    sum_weights = np.sum(aligned_weights)
                    if sum_weights > 0:
                        weights = aligned_weights / sum_weights
                        y_g = np.nansum(
                            mat * weights[np.newaxis, :], axis=1
                        )  # Use nansum for weighted average
                    else:
                        warnings.warn(
                            f"Sum of weights is 0 for group '{g_label}' with id_col '{id_col}'. Defaulting to unweighted nanmean.",
                            UserWarning,
                        )
                        y_g = np.nanmean(mat, axis=1)
                else:
                    y_g = np.nanmean(mat, axis=1)  # Use nanmean for robustness

                l_g, u_g = bootstrap_ci_id(mat, alpha, bootstrap_samples, rng)
                current_env_l, current_env_u = compute_envelope(
                    mat,
                    method=envelope_method,
                    q_lower=envelope_quantiles[0],
                    q_upper=envelope_quantiles[1],
                    mad_scale=envelope_mad_scale,
                )
        else:
            y_vals = df_g[y_col].to_numpy()

            if kind == "cumulative":
                y_g = edf_numpy(y_vals, x_grid)
            elif kind == "survival":
                y_g = survival_numpy(y_vals, x_grid)
            else:
                counts, bin_width = histogram_numpy(y_vals, x_grid)
                if kind == "density":
                    if y_vals.size == 0 or bin_width <= 1e-9:
                        y_g.fill(np.nan)
                    else:
                        y_g = counts / (y_vals.size * bin_width)
                elif kind == "mass":
                    if y_vals.size == 0:
                        y_g.fill(np.nan)
                    else:
                        y_g = counts / y_vals.size
                elif kind == "count":
                    y_g = counts.astype(float)
                else:
                    raise ValueError(f"Unknown kind: {kind}")

            if not np.all(np.isnan(y_g)):  # Only compute CIs if y_g is not all NaN
                if ci_method == "dkw" and kind == "cumulative":
                    eps = dkw_ci(len(y_vals), alpha)
                    l_g, u_g = np.clip(y_g - eps, 0, 1), np.clip(y_g + eps, 0, 1)
                elif ci_method == "asymptotic" and kind == "cumulative":
                    l_g, u_g = asymptotic_ci(y_g, len(y_vals), alpha)
                elif ci_method == "bootstrap":

                    def _stat_fn_bootstrap(data_sample, eval_grid):
                        if data_sample.size == 0:
                            val = np.full_like(eval_grid, np.nan, dtype=float)
                            if kind == "count":
                                val = np.zeros_like(eval_grid, dtype=float)
                            return val
                        if kind == "cumulative":
                            return edf_numpy(data_sample, eval_grid)
                        if kind == "survival":
                            return survival_numpy(data_sample, eval_grid)
                        s_counts, s_width = histogram_numpy(data_sample, eval_grid)
                        if kind == "density":
                            return (
                                (s_counts / (data_sample.size * s_width))
                                if (data_sample.size * s_width > 1e-9)
                                else np.full_like(s_counts, np.nan, dtype=float)
                            )
                        if kind == "mass":
                            return (
                                (s_counts / data_sample.size)
                                if data_sample.size > 0
                                else np.full_like(s_counts, np.nan, dtype=float)
                            )
                        if kind == "count":
                            return s_counts.astype(float)
                        raise ValueError(f"Unsupported kind '{kind}' for bootstrap CI.")

                    l_g, u_g = _bootstrap_ci_statistic(
                        y_vals,
                        _stat_fn_bootstrap,
                        x_grid,
                        alpha,
                        bootstrap_samples,
                        rng,
                    )
                else:
                    warnings.warn(
                        f"CI method '{ci_method}' for kind '{kind}' (non-ID) not supported. No CIs computed.",
                        UserWarning,
                    )

        y_out.append(y_g)
        l_out.append(l_g)
        u_out.append(u_g)
        env_l_out.append(current_env_l)
        env_u_out.append(current_env_u)

    final_y = (
        np.column_stack(y_out)
        if y_out and any(item is not None for item in y_out)
        else np.full((x_grid.size, len(group_labels_list)), np.nan)
    )
    final_l = (
        np.column_stack(l_out)
        if l_out and any(item is not None for item in l_out)
        else np.full((x_grid.size, len(group_labels_list)), np.nan)
    )
    final_u = (
        np.column_stack(u_out)
        if u_out and any(item is not None for item in u_out)
        else np.full((x_grid.size, len(group_labels_list)), np.nan)
    )

    has_envelope_data_l = env_l_out and any(
        arr is not None and not np.all(np.isnan(arr)) for arr in env_l_out
    )
    has_envelope_data_u = env_u_out and any(
        arr is not None and not np.all(np.isnan(arr)) for arr in env_u_out
    )

    final_env_l = np.column_stack(env_l_out) if has_envelope_data_l else None
    final_env_u = np.column_stack(env_u_out) if has_envelope_data_u else None

    return DistributionResult(
        x=x_grid,
        y=final_y,
        l=final_l,
        u=final_u,
        group_labels=group_labels_list,
        kind=kind,
        y_col_name=y_col,
        env_l=final_env_l,
        env_u=final_env_u,
        log_data=log_data,
    )


# ### Permutation ANOVA and post-hoc
# # -----------------------------------------------------------------------------
# # JAX ANOVA CORE
# # -----------------------------------------------------------------------------

# def compute_ss_jax(y, groups, num_groups, eps=1e-12):
#     return _compute_ss_static(y, groups, int(num_groups), eps)

# @partial(jit, static_argnums=(2,))
# def _compute_ss_static(y, groups, num_groups, eps=1e-12):
#     μ = jnp.mean(y)
#     gs = segment_sum(y, groups, num_segments=num_groups)
#     gc = segment_sum(jnp.ones_like(y), groups, num_segments=num_groups)
#     gm = gs / gc
#     resid_within = y - gm[groups]
#     resid_total = y - μ
#     ssw = jnp.sum(resid_within**2) + eps
#     sst = jnp.sum(resid_total**2) + eps
#     return ssw, sst, μ


# def compute_anova_stats_jax(y, groups, num_groups):
#     return _compute_anova_stats_static(y, groups, int(num_groups))

# @partial(jit, static_argnums=(2,))
# def _compute_anova_stats_static(y, groups, num_groups):
#     ssw, sst, μ = _compute_ss_static(y, groups, num_groups)
#     n = y.shape[0]
#     dfb = num_groups - 1
#     dfw = n - num_groups
#     ssb = sst - ssw
#     msb = ssb / dfb
#     msw = ssw / dfw
#     F = msb / msw
#     omega2 = (ssb - dfb * msw) / (sst + msw)
#     return F, omega2, μ


# @jit
# def compute_logL_jax(ss, n):
#     return -0.5 * n * (jnp.log(2 * jnp.pi) + jnp.log(ss / n) + 1)


# def compute_permuted_F_jax(resid, μ, key, groups, num_groups):
#     return _compute_permuted_F_static(resid, μ, key, groups, int(num_groups))

# @partial(jit, static_argnums=(4,))
# def _compute_permuted_F_static(resid, μ, key, groups, num_groups):
#     pr = random.permutation(key, resid)
#     yp = μ + pr
#     Fp, _, _ = _compute_anova_stats_static(yp, groups, num_groups)
#     return Fp


# def jax_permutation_anova(y, groups, num_groups, perm_keys):
#     F_obs, omega2_obs, μ = _compute_anova_stats_static(y, groups, int(num_groups))
#     resid = y - μ
#     perm_Fs = vmap(lambda key: _compute_permuted_F_static(resid, μ, key, groups, int(num_groups)))(perm_keys)
#     return F_obs, perm_Fs, omega2_obs


# # -----------------------------------------------------------------------------
# # NumPy Stratified Bootstrap Indexing (Efficient CPU logic)
# # -----------------------------------------------------------------------------

# def generate_numpy_stratified_bootstrap_indices(labels: np.ndarray, num_bootstrap: int, seed: int) -> np.ndarray:
#     rng = np.random.default_rng(seed)
#     unique_groups = np.unique(labels)
#     group_indices = [np.where(labels == g)[0] for g in unique_groups]

#     sampled_indices = [
#         rng.choice(idx, size=(num_bootstrap, len(idx)), replace=True)
#         for idx in group_indices
#     ]

#     return np.hstack(sampled_indices)


# def compute_omega2_numpy(y: np.ndarray, groups: np.ndarray) -> float:
#     overall_mean = np.mean(y)
#     ss_total = np.sum((y - overall_mean) ** 2)
#     uniqs = np.unique(groups)
#     ss_within = sum(
#         np.sum((y[groups == g] - np.mean(y[groups == g])) ** 2) for g in uniqs
#     )
#     n = len(y)
#     k = len(uniqs)
#     dfb = k - 1
#     dfw = n - k
#     msw = ss_within / dfw
#     ssb = ss_total - ss_within
#     omega2 = (ssb - dfb * msw) / (ss_total + msw)
#     return omega2


# def compute_omega2_bootstrap_numpy(y: np.ndarray, labels: np.ndarray, num_bootstrap: int, seed: int) -> np.ndarray:
#     indices = generate_numpy_stratified_bootstrap_indices(labels, num_bootstrap, seed)
#     n = len(y)
#     boot_omegas = [
#         compute_omega2_numpy(y[idx], labels[idx]) for idx in indices.reshape(num_bootstrap, n)
#     ]
#     return np.array(boot_omegas)


# # -----------------------------------------------------------------------------
# # High-Level API
# # -----------------------------------------------------------------------------

# def residual_permutation_anova(
#     df: pd.DataFrame,
#     group_col: str,
#     value_col: str,
#     num_permutations: int = 1000,
#     num_bootstrap: int = 1000,
#     ci: float = 0.95,
#     seed: int = 42,
#     return_distributions: bool = False,
# ) -> dict:
#     if df[group_col].nunique() < 2:
#         raise ValueError("Need at least two groups.")

#     y = pd.to_numeric(df[value_col], errors="raise").values
#     labels, uniques = pd.factorize(df[group_col])
#     n, k = len(y), len(uniques)

#     # Permutation-based F and omega² in JAX
#     y_jax = jnp.array(y)
#     grp_jax = jnp.array(labels)
#     perm_keys = random.split(random.PRNGKey(seed), num_permutations)
#     F_obs_j, perm_Fs_j, omega2_obs_j = jax_permutation_anova(y_jax, grp_jax, k, perm_keys)

#     F_obs = float(F_obs_j)
#     perm_Fs = np.array(perm_Fs_j)
#     p_value = float((perm_Fs >= F_obs).sum() + 1) / (num_permutations + 1)
#     alpha = 1 / np.sqrt(n)
#     significant = p_value < alpha

#     # Efficient NumPy stratified bootstrap
#     omega2_boot = compute_omega2_bootstrap_numpy(y, labels, num_bootstrap, seed + 1)
#     ci_lo = float(np.percentile(omega2_boot, (1 - ci) / 2 * 100))
#     ci_hi = float(np.percentile(omega2_boot, (1 + ci) / 2 * 100))

#     # AIC/BIC computation in JAX
#     ssw_j, sst_j, _ = _compute_ss_static(y_jax, grp_jax, int(k))
#     logL_null = float(compute_logL_jax(sst_j, n))
#     logL_full = float(compute_logL_jax(ssw_j, n))
#     AIC_null = 2 * 1 - 2 * logL_null
#     AIC_full = 2 * k - 2 * logL_full
#     BIC_null = np.log(n) * 1 - 2 * logL_null
#     BIC_full = np.log(n) * k - 2 * logL_full

#     result = {
#         "F_statistic": F_obs,
#         "p_value": p_value,
#         "omega_squared": float(omega2_obs_j),
#         "omega2_ci": (ci_lo, ci_hi),
#         "alpha_adaptive": alpha,
#         "significant_adaptive": significant,
#         "AIC_null": AIC_null,
#         "AIC_full": AIC_full,
#         "delta_AIC": AIC_null - AIC_full,
#         "BIC_null": BIC_null,
#         "BIC_full": BIC_full,
#         "delta_BIC": BIC_null - BIC_full,
#     }

#     if return_distributions:
#         result["perm_Fs"] = perm_Fs
#         result["omega2_bootstrap"] = omega2_boot

#     return result


# # -----------------------------------------------------------------------------
# # JAX Pairwise Post-Hoc (static group sizes to avoid tracer errors)
# # -----------------------------------------------------------------------------


# @partial(jit, static_argnums=(5, 6))
# def jax_pairwise_permutation_cohend(
#     y1: jnp.ndarray,
#     y2: jnp.ndarray,
#     perm_keys: jnp.ndarray,
#     boot_keys: jnp.ndarray,
#     ci: float,
#     c1: int,
#     c2: int,
# ) -> tuple[
#     jnp.ndarray,
#     jnp.ndarray,
#     jnp.ndarray,
#     jnp.ndarray,
#     jnp.ndarray,
#     jnp.ndarray,
#     jnp.ndarray,
# ]:
#     """
#     For one pair of groups (sizes c1, c2):
#      - permutation test of mean diff
#      - bootstrap CI for Cohen's d
#     Returns:
#       obs_diff, perm_diffs, p_raw,
#       obs_d,    d_boot,    ci_lo, ci_hi
#     """
#     # Observed stats
#     obs_diff = jnp.mean(y1) - jnp.mean(y2)
#     obs_d = obs_diff / jnp.sqrt(
#         ((c1 - 1) * jnp.var(y1, ddof=1) + (c2 - 1) * jnp.var(y2, ddof=1))
#         / (c1 + c2 - 2)
#     )

#     # Permutation distribution
#     pooled = jnp.concatenate([y1, y2])

#     def _perm(key):
#         p = random.permutation(key, pooled)
#         return jnp.mean(p[:c1]) - jnp.mean(p[c1 : c1 + c2])

#     perm_diffs = vmap(_perm)(perm_keys)
#     p_raw = (jnp.sum(jnp.abs(perm_diffs) >= jnp.abs(obs_diff)) + 1) / (
#         perm_keys.shape[0] + 1
#     )

#     # Bootstrap Cohen's d
#     def _boot(key):
#         b1 = random.choice(key, y1, shape=(c1,), replace=True)
#         b2 = random.choice(key, y2, shape=(c2,), replace=True)
#         s1 = jnp.var(b1, ddof=1)
#         s2 = jnp.var(b2, ddof=1)
#         psd = jnp.sqrt(((c1 - 1) * s1 + (c2 - 1) * s2) / (c1 + c2 - 2))
#         return (jnp.mean(b1) - jnp.mean(b2)) / psd

#     d_boot = vmap(_boot)(boot_keys)
#     ci_lo = jnp.percentile(d_boot, (1 - ci) / 2 * 100)
#     ci_hi = jnp.percentile(d_boot, (1 + ci) / 2 * 100)

#     return obs_diff, perm_diffs, p_raw, obs_d, d_boot, ci_lo, ci_hi


# # -----------------------------------------------------------------------------
# # P-value Adjustment
# # -----------------------------------------------------------------------------


# def adjust_pvalues(pvals: np.ndarray, method: str = "bonferroni") -> np.ndarray:
#     p = np.asarray(pvals)
#     m = len(p)
#     if method == "bonferroni":
#         return np.minimum(p * m, 1.0)
#     idx = np.argsort(p)
#     sorted_p = p[idx]
#     cummin = np.minimum.accumulate((m / np.arange(1, m + 1)) * sorted_p[::-1])[::-1]
#     out = np.empty(m)
#     out[idx] = np.minimum(cummin, 1.0)
#     return out


# # -----------------------------------------------------------------------------
# # High-level Post-Hoc Wrapper
# # -----------------------------------------------------------------------------


# def posthoc_pairwise_permutation(
#     df: pd.DataFrame,
#     group_col: str,
#     value_col: str,
#     num_permutations: int = 1000,
#     num_bootstrap: int = 1000,
#     ci: float = 0.95,
#     p_adjust: str = "bonferroni",
#     seed: int = 42,
# ) -> pd.DataFrame:
#     y = pd.to_numeric(df[value_col], errors="raise").values
#     labels, uniques = pd.factorize(df[group_col])

#     perm_keys = random.split(random.PRNGKey(seed), num_permutations)
#     boot_keys = random.split(random.PRNGKey(seed + 1), num_bootstrap)

#     records = []
#     for i, j in combinations(range(len(uniques)), 2):
#         y1 = jnp.array(y[labels == i])
#         y2 = jnp.array(y[labels == j])
#         c1, c2 = int(y1.shape[0]), int(y2.shape[0])

#         (obs_diff, perm_diffs, p_raw, obs_d, d_boot, ci_lo, ci_hi) = (
#             jax_pairwise_permutation_cohend(y1, y2, perm_keys, boot_keys, ci, c1, c2)
#         )

#         records.append(
#             {
#                 "group1": uniques[i],
#                 "group2": uniques[j],
#                 "mean_diff": float(obs_diff),
#                 "cohen_d": float(obs_d),
#                 "ci_lower": float(ci_lo),
#                 "ci_upper": float(ci_hi),
#                 "p_raw": float(p_raw),
#             }
#         )

#     df_res = pd.DataFrame(records)
#     df_res["p_adj"] = adjust_pvalues(df_res["p_raw"].values, method=p_adjust)
#     return df_res

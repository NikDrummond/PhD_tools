import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Any, Tuple, Literal
from scipy.stats import norm


import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.ops import segment_sum
from functools import partial
from itertools import combinations

# Optional JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from jax import random as jrandom
    from functools import partial

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from scipy.stats import norm


# -------------------
# Grid computation
# -------------------
def _compute_x(
    data: np.ndarray, x0: Optional[float], x1: Optional[float], bins: Optional[int]
) -> np.ndarray:
    data_min, data_max = np.min(data), np.max(data)
    x0 = data_min if x0 is None else x0
    x1 = data_max if x1 is None else x1
    buffer = 0.05 * (x1 - x0)
    x0, x1 = x0 - buffer, x1 + buffer
    if bins is None:
        N = data.size
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        width = 2 * iqr / np.cbrt(N)
        bins = max(10, int((x1 - x0) / width))
    return np.linspace(x0, x1, bins)


# -------------------
# EDF computation
# -------------------


def _edf_values_numpy(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    sorted_data = np.sort(data)
    return np.searchsorted(sorted_data, x, side="right") / data.size


if JAX_AVAILABLE:

    @jit
    def _edf_values_jax(data: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sorted_data = jnp.sort(data)
        return jnp.searchsorted(sorted_data, x, side="right") / data.shape[0]


# -------------------
# Confidence intervals (numpy)
# -------------------


def _ci_dkw_numpy(N: int, alpha: float) -> float:
    return np.sqrt(np.log(2.0 / alpha) / (2 * N))


def _ci_asymptotic_numpy(
    y: np.ndarray, N: int, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    zval = norm.ppf(1 - alpha / 2)
    std = np.sqrt(y * (1 - y) / N)
    half = zval * std
    return np.clip(y - half, 0, 1), np.clip(y + half, 0, 1)


def _ci_bootstrap_numpy(
    data: np.ndarray,
    x: np.ndarray,
    alpha: float,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    N = data.size
    boot = np.empty((x.size, samples))
    for b in range(samples):
        samp = rng.choice(data, size=N, replace=True)
        boot[:, b] = _edf_values_numpy(samp, x)
    low_q, high_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return np.percentile(boot, low_q, axis=1), np.percentile(boot, high_q, axis=1)


# -------------------
# Confidence intervals (JAX)
# -------------------
if JAX_AVAILABLE:

    @jit
    def _single_boot_edf_jax(
        key: jnp.ndarray, data: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        idx = jrandom.randint(key, (data.shape[0],), 0, data.shape[0])
        samp = data[idx]
        sd = jnp.sort(samp)
        return jnp.searchsorted(sd, x, side="right") / samp.shape[0]

    @partial(jit, static_argnums=(3,))
    def _ci_bootstrap_jax_jit(
        data: jnp.ndarray,
        x: jnp.ndarray,
        alpha: float,
        samples: int,
        base_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = jrandom.split(base_key, samples)
        boots = vmap(lambda k: _single_boot_edf_jax(k, data, x))(keys)
        low_q = 100 * (alpha / 2)
        high_q = 100 * (1 - alpha / 2)
        lower = jnp.percentile(boots, low_q, axis=0)
        upper = jnp.percentile(boots, high_q, axis=0)
        return lower, upper

    def _ci_bootstrap_jax(
        data_np: np.ndarray, x_np: np.ndarray, alpha: float, samples: int, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        data_j = jnp.array(data_np)
        x_j = jnp.array(x_np)
        base_key = jrandom.PRNGKey(seed)
        low_j, high_j = _ci_bootstrap_jax_jit(data_j, x_j, alpha, samples, base_key)
        return np.array(low_j), np.array(high_j)


# -------------------
# Envelope methods (numpy)
# -------------------


def _env_percentile_numpy(
    arr: np.ndarray, low_q: float, high_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    return np.percentile(arr, 100 * low_q, axis=1), np.percentile(
        arr, 100 * high_q, axis=1
    )


def _env_minmax_numpy(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return arr.min(axis=1), arr.max(axis=1)


def _env_mad_numpy(arr: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(arr, axis=1)
    mad = np.median(np.abs(arr - med[:, None]), axis=1)
    return np.clip(med - scale * mad, 0, 1), np.clip(med + scale * mad, 0, 1)


def _env_asymmad_numpy(
    arr: np.ndarray, scale_lo: float, scale_hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(arr, axis=1)
    dev = arr - med[:, None]
    mad_lo = np.median(np.abs(np.where(dev < 0, dev, 0)), axis=1)
    mad_hi = np.median(np.abs(np.where(dev > 0, dev, 0)), axis=1)
    return np.clip(med - scale_lo * mad_lo, 0, 1), np.clip(
        med + scale_hi * mad_hi, 0, 1
    )


def _env_bootstrap_numpy(
    arr: np.ndarray, low_q: float, high_q: float, samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    xsize, nids = arr.shape
    boot_low = np.empty((xsize, samples))
    boot_high = np.empty((xsize, samples))
    for b in range(samples):
        sel = rng.integers(0, nids, size=nids)
        samp_arr = arr[:, sel]
        boot_low[:, b] = np.percentile(samp_arr, 100 * low_q, axis=1)
        boot_high[:, b] = np.percentile(samp_arr, 100 * high_q, axis=1)
    return boot_low.mean(axis=1), boot_high.mean(axis=1)


# -------------------
# Envelope methods (JAX)
# -------------------
if JAX_AVAILABLE:

    @jit
    def _single_env_boot_jax(
        key: jnp.ndarray, arr: jnp.ndarray, low_q: float, high_q: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nids = arr.shape[1]
        idx = jrandom.randint(key, (nids,), 0, nids)
        samp = arr[:, idx]
        low = jnp.percentile(samp, 100 * low_q, axis=1)
        high = jnp.percentile(samp, 100 * high_q, axis=1)
        return low, high

    @partial(jit, static_argnums=(3,))
    def _env_bootstrap_jax_jit(
        arr: jnp.ndarray,
        low_q: float,
        high_q: float,
        samples: int,
        base_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = jrandom.split(base_key, samples)
        boots = vmap(lambda k: _single_env_boot_jax(k, arr, low_q, high_q))(keys)
        # boots: (samples, 2, n_bins)
        lows = jnp.stack([b[0] for b in boots], axis=1)
        highs = jnp.stack([b[1] for b in boots], axis=1)
        return lows.mean(axis=1), highs.mean(axis=1)

    def _env_bootstrap_jax(
        arr: np.ndarray, low_q: float, high_q: float, samples: int, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr_j = jnp.array(arr)
        base_key = jrandom.PRNGKey(seed)
        low_j, high_j = _env_bootstrap_jax_jit(arr_j, low_q, high_q, samples, base_key)
        return np.array(low_j), np.array(high_j)


# -------------------
# Result container
# -------------------


class EDFResult:
    """
    Container for empirical distribution function results.

    Attributes
    ----------
    x : np.ndarray, shape (n_bins,)
        Shared evaluation grid.
    y : np.ndarray, shape (n_bins, n_groups)
        EDF values for each group.
    l : np.ndarray, shape (n_bins, n_groups)
        Lower confidence bounds.
    u : np.ndarray, shape (n_bins, n_groups)
        Upper confidence bounds.
    group_labels : list
        Labels for the group dimension.
    y_id : np.ndarray or None, shape (n_bins, n_ids)
        EDF curves for each ID if provided.
    id_labels : list or None
        Labels for the ID dimension.
    env_lower : np.ndarray or None, shape (n_bins, n_groups)
        Lower envelope of ID variability, if computed.
    env_upper : np.ndarray or None, shape (n_bins, n_groups)
        Upper envelope of ID variability, if computed.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        group_labels: List[Any],
        y_id: Optional[np.ndarray] = None,
        id_labels: Optional[List[Any]] = None,
        env_lower: Optional[np.ndarray] = None,
        env_upper: Optional[np.ndarray] = None,
    ):
        self.x = x
        self.y = y
        self.l = l
        self.u = u
        self.group_labels = group_labels
        self.y_id = y_id
        self.id_labels = id_labels
        self.env_lower = env_lower
        self.env_upper = env_upper

    def __repr__(self) -> str:
        desc = (
            f"EDFResult(x={self.x.shape}, y={self.y.shape}, groups={self.group_labels})"
        )
        if self.id_labels is not None:
            desc = desc[:-1] + f", ids={self.id_labels})"
        return desc

    def plot(
        self,
        ax=None,
        colors: Optional[List[str]] = None,
        ci_alpha: Optional[float] = None,
        envelope: bool = True,
        **plot_kwargs,
    ):
        """
        Plot group-level empirical distribution functions with confidence intervals and optional envelope.

        Plots one curve per group, shades the confidence interval band, and optionally overlays the ID variability envelope.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw the plot on. If None, a new figure and axis are created.
        colors : list of str, optional
            List of colors for each group curve. Cycled if fewer than number of groups.
        ci_alpha : float, optional
            Transparency level for the confidence interval shading. Defaults to 0.3.
        envelope : bool, default True
            If True and ID variability envelope is available, overlay the envelope band.
        **plot_kwargs
            Additional keyword arguments passed to `ax.plot` for the EDF curves.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes with the plot.
        """

        if ax is None:
            fig, ax = plt.subplots()
        # Default colors from matplotlib
        if colors is None:
            prop_cycle = plt.rcParams.get("axes.prop_cycle")
            colors = prop_cycle.by_key().get("color", None)
        n_groups = self.y.shape[1]
        # Default transparency for CI
        alpha_ci = ci_alpha if ci_alpha is not None else 0.3

        for idx, label in enumerate(self.group_labels):
            col = colors[idx % len(colors)] if colors is not None else None
            # Plot EDF curve
            ax.plot(self.x, self.y[:, idx], label=str(label), color=col, **plot_kwargs)
            # Shade confidence interval
            ax.fill_between(
                self.x, self.l[:, idx], self.u[:, idx], alpha=alpha_ci, color=col
            )
            # Optional envelope
            if envelope and self.env_lower is not None:
                ax.fill_between(
                    self.x,
                    self.env_lower[:, idx],
                    self.env_upper[:, idx],
                    alpha=alpha_ci / 2,
                    color=col,
                    linestyle="--",
                    linewidth=0,
                )
        ax.legend()
        return ax


# -------------------
# Main EDF function
# -------------------


def edf(
    df,
    y: str,
    id: Optional[str] = None,
    group: Optional[str] = None,
    ci_method: Literal["dkw", "asymptotic", "bootstrap"] = "dkw",
    ci_alpha: float = 0.05,
    ci_bootstrap_samples: int = 1000,
    envelope_method: Literal[
        "percentile", "minmax", "bootstrap", "mad", "asymmad"
    ] = "percentile",
    env_quantiles: Tuple[float, float] = (0.025, 0.975),
    env_bootstrap_samples: int = 500,
    envelope_scale: float = 1.0,
    envelope_scale_lower: float = 1.0,
    envelope_scale_upper: float = 1.0,
    x0: Optional[float] = None,
    x1: Optional[float] = None,
    bins: Optional[int] = None,
    use_jax: bool = False,
) -> EDFResult:
    """
    Compute empirical distribution functions (EDFs) with optional grouping, ID curves, and confidence/envelope bands.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data in long format.
    y : str
        Column name of the values variable.
    id : str, optional
        Column name for individual identifiers. If provided, EDFs are computed per ID.
    group : str, optional
        Column name for grouping. If provided, EDFs and CIs are computed per group.
    ci_method : {'dkw', 'asymptotic', 'bootstrap'}, default 'dkw'
        Method for computing confidence intervals:
        - 'dkw': Dvoretzky–Kiefer–Wolfowitz bound (uniform).
        - 'asymptotic': normal approximation at each x.
        - 'bootstrap': nonparametric bootstrap.
    ci_alpha : float, default 0.05
        Significance level (alpha) for confidence intervals.
    ci_bootstrap_samples : int, default 1000
        Number of bootstrap replicates if ci_method='bootstrap'.
    envelope_method : {'percentile', 'minmax', 'bootstrap', 'mad', 'asymmad'}, default 'percentile'
        Method for computing the ID variability envelope.
    env_quantiles : tuple of float, default (0.025, 0.975)
        Lower and upper quantiles for percentile or bootstrap envelopes.
    env_bootstrap_samples : int, default 500
        Number of bootstrap replicates if envelope_method='bootstrap'.
    envelope_scale : float, default 1.0
        Scale factor for MAD envelope.
    envelope_scale_lower : float, default 1.0
        Lower-scale factor for asymmetric MAD envelope.
    envelope_scale_upper : float, default 1.0
        Upper-scale factor for asymmetric MAD envelope.
    x0, x1 : float, optional
        Lower and upper bounds for evaluation grid. Defaults to data range with 5% buffer.
    bins : int, optional
        Number of grid points. If None, uses Freedman–Diaconis rule.
    use_jax : bool, default False
        If True and JAX is available, uses JAX-accelerated computations for EDF and bootstrap.

    Returns
    -------
    result : EDFResult
        Object containing:
        - x: evaluation grid
        - y, l, u: EDF curves and confidence bands per group
        - y_id: EDF curves per ID (if id provided)
        - env_lower, env_upper: ID variability envelope per group (if id provided)
    """
    # Preserve column names
    val_col = y
    id_col = id
    grp_col = group

    # Base data array
    values = df[val_col].to_numpy(dtype=float)
    x = _compute_x(values, x0, x1, bins)
    n_bins = x.size

    # Primary grouping keys
    if grp_col is not None:
        grp_keys = df[grp_col].unique().tolist()
    elif id_col is not None:
        grp_keys = df[id_col].unique().tolist()
    else:
        grp_keys = ["all"]
    n_groups = len(grp_keys)

    # Optional ID labels
    if id_col is not None:
        id_labels = df[id_col].unique().tolist()
    else:
        id_labels = None

    # Precompute JAX arrays if needed
    if use_jax and JAX_AVAILABLE:
        x_jax = jnp.array(x)

    # Initialize outputs
    y_out = np.empty((n_bins, n_groups))
    l_out = np.empty((n_bins, n_groups))
    u_out = np.empty((n_bins, n_groups))
    rng = np.random.default_rng()

    # Group-level EDF and CI
    for j, key in enumerate(grp_keys):
        # Subset data for this "group"
        if grp_col is None and id_col is None:
            data_j = values
        elif grp_col is not None:
            data_j = df[df[grp_col] == key][val_col].to_numpy(dtype=float)
        else:
            data_j = df[df[id_col] == key][val_col].to_numpy(dtype=float)

        # EDF
        if use_jax and JAX_AVAILABLE:
            data_jax = jnp.array(data_j)
            y_vals = np.array(_edf_values_jax(data_jax, x_jax))
        else:
            y_vals = _edf_values_numpy(data_j, x)
        y_out[:, j] = y_vals

        # CI
        if ci_method == "dkw":
            half = _ci_dkw_numpy(data_j.size, ci_alpha)
            l_out[:, j] = np.clip(y_vals - half, 0, 1)
            u_out[:, j] = np.clip(y_vals + half, 0, 1)
        elif ci_method == "asymptotic":
            l_out[:, j], u_out[:, j] = _ci_asymptotic_numpy(
                y_vals, data_j.size, ci_alpha
            )
        else:
            if use_jax and JAX_AVAILABLE:
                l_out[:, j], u_out[:, j] = _ci_bootstrap_jax(
                    data_j, x, ci_alpha, ci_bootstrap_samples
                )
            else:
                l_out[:, j], u_out[:, j] = _ci_bootstrap_numpy(
                    data_j, x, ci_alpha, ci_bootstrap_samples, rng
                )

    # ID-level curves
    if id_col is not None:
        y_id = np.empty((n_bins, len(id_labels)))
        for i, iid in enumerate(id_labels):
            data_i = df[df[id_col] == iid][val_col].to_numpy(dtype=float)
            if use_jax and JAX_AVAILABLE:
                y_id[:, i] = np.array(_edf_values_jax(jnp.array(data_i), x_jax))
            else:
                y_id[:, i] = _edf_values_numpy(data_i, x)
    else:
        y_id = None

    # Envelope if IDs present
    env_lower = None
    env_upper = None
    if id_col is not None:
        env_lower = np.empty_like(y_out)
        env_upper = np.empty_like(y_out)
        for j, key in enumerate(grp_keys):
            # collect relevant ID curves
            if grp_col is not None:
                ids_in_g = df[df[grp_col] == key][id_col].unique().tolist()
                idxs = [id_labels.index(iid) for iid in ids_in_g]
            else:
                idxs = [j]
            arr = y_id[:, idxs]
            if envelope_method == "percentile":
                env_lower[:, j], env_upper[:, j] = _env_percentile_numpy(
                    arr, env_quantiles[0], env_quantiles[1]
                )
            elif envelope_method == "minmax":
                env_lower[:, j], env_upper[:, j] = _env_minmax_numpy(arr)
            elif envelope_method == "mad":
                env_lower[:, j], env_upper[:, j] = _env_mad_numpy(arr, envelope_scale)
            elif envelope_method == "asymmad":
                env_lower[:, j], env_upper[:, j] = _env_asymmad_numpy(
                    arr, envelope_scale_lower, envelope_scale_upper
                )
            else:
                if use_jax and JAX_AVAILABLE:
                    env_lower[:, j], env_upper[:, j] = _env_bootstrap_jax(
                        arr, env_quantiles[0], env_quantiles[1], env_bootstrap_samples
                    )
                else:
                    env_lower[:, j], env_upper[:, j] = _env_bootstrap_numpy(
                        arr,
                        env_quantiles[0],
                        env_quantiles[1],
                        env_bootstrap_samples,
                        rng,
                    )

    return EDFResult(
        x=x,
        y=y_out,
        l=l_out,
        u=u_out,
        group_labels=grp_keys,
        y_id=y_id,
        id_labels=id_labels,
        env_lower=env_lower,
        env_upper=env_upper,
    )


# -------------------
# Histogram-based EPDF helpers
# -------------------


def _epdf_histogram_numpy(data: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute histogram-based density estimates on grid midpoints x.

    Returns
    -------
    f : np.ndarray, shape (n_bins,)
        Estimated density at each x.
    width : float
        Bin width.
    """
    edges = np.linspace(x[0], x[-1], x.size + 1)
    counts, _ = np.histogram(data, bins=edges)
    width = edges[1] - edges[0]
    f = counts / (data.size * width)
    return f, width


def _ci_bootstrap_epdf_numpy(
    data: np.ndarray,
    x: np.ndarray,
    alpha: float,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence intervals for histogram-based EPDF.
    """
    N = data.size
    edges = np.linspace(x[0], x[-1], x.size + 1)
    width = edges[1] - edges[0]
    boot = np.empty((x.size, samples))
    for b in range(samples):
        samp = rng.choice(data, size=N, replace=True)
        counts, _ = np.histogram(samp, bins=edges)
        boot[:, b] = counts / (N * width)
    low_q, high_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return np.percentile(boot, low_q, axis=1), np.percentile(boot, high_q, axis=1)


# -------------------
# EPDF bootstrap (JAX)
# -------------------
if JAX_AVAILABLE:

    @jit
    def _single_boot_epdf_jax(
        key: jnp.ndarray, data: jnp.ndarray, edges: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Single bootstrap replicate for histogram-based EPDF in JAX.
        """
        N = data.shape[0]
        # sample indices with replacement
        idx = jrandom.randint(key, (N,), 0, N)
        samp = data[idx]
        # assign to bins and count
        bin_idx = jnp.digitize(samp, edges)
        counts = jnp.bincount(bin_idx, length=edges.shape[0] + 1)[1:-1]
        width = edges[1] - edges[0]
        return counts / (N * width)

    @partial(jit, static_argnums=(3,))
    def _ci_bootstrap_epdf_jax_jit(
        data: jnp.ndarray,
        x: jnp.ndarray,
        alpha: float,
        samples: int,
        base_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-accelerated bootstrap CI for EPDF in JAX.
        """
        # construct bin edges
        edges = jnp.linspace(x[0], x[-1], x.size + 1)
        # split keys for replicates
        keys = jrandom.split(base_key, samples)
        # vectorize single bootstrap
        boots = vmap(lambda k: _single_boot_epdf_jax(k, data, edges))(keys)
        low_q = 100 * (alpha / 2)
        high_q = 100 * (1 - alpha / 2)
        lower = jnp.percentile(boots, low_q, axis=0)
        upper = jnp.percentile(boots, high_q, axis=0)
        return lower, upper

    def _ci_bootstrap_epdf_jax(
        data_np: np.ndarray, x_np: np.ndarray, alpha: float, samples: int, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entry-point for JAX-bootstrapped EPDF CIs: converts inputs and calls JIT.
        """
        data_j = jnp.array(data_np)
        x_j = jnp.array(x_np)
        base_key = jrandom.PRNGKey(seed)
        low_j, high_j = _ci_bootstrap_epdf_jax_jit(
            data_j, x_j, alpha, samples, base_key
        )
        return np.array(low_j), np.array(high_j)


# -------------------
# EPDF envelope (JAX)
# -------------------
if JAX_AVAILABLE:

    @jit
    def _single_env_boot_jax(
        key: jnp.ndarray, arr: jnp.ndarray, low_q: float, high_q: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single bootstrap replicate for ID-level envelope in JAX.
        """
        nids = arr.shape[1]
        idx = jrandom.randint(key, (nids,), 0, nids)
        samp = arr[:, idx]
        low = jnp.percentile(samp, 100 * low_q, axis=1)
        high = jnp.percentile(samp, 100 * high_q, axis=1)
        return low, high

    @partial(jit, static_argnums=(3,))
    def _env_bootstrap_jax_jit(
        arr: jnp.ndarray,
        low_q: float,
        high_q: float,
        samples: int,
        base_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-accelerated bootstrap envelope for ID variability in JAX.
        """
        keys = jrandom.split(base_key, samples)
        boots = vmap(lambda k: _single_env_boot_jax(k, arr, low_q, high_q))(keys)
        lows = jnp.stack([b[0] for b in boots], axis=1)
        highs = jnp.stack([b[1] for b in boots], axis=1)
        return lows.mean(axis=1), highs.mean(axis=1)

    def _env_bootstrap_jax(
        arr_np: np.ndarray, low_q: float, high_q: float, samples: int, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entry-point for JAX-bootstrapped ID envelope: converts inputs and calls JIT.
        """
        arr_j = jnp.array(arr_np)
        base_key = jrandom.PRNGKey(seed)
        low_j, high_j = _env_bootstrap_jax_jit(arr_j, low_q, high_q, samples, base_key)
        return np.array(low_j), np.array(high_j)


# -------------------
# histogram computation helpers
# -------------------


def _compute_group_histogram(
    data: np.ndarray,
    x: np.ndarray,
    ci_method: Literal["dkw", "asymptotic", "bootstrap"],
    ci_alpha: float,
    ci_bootstrap_samples: int,
    rng: np.random.Generator,
    use_jax: bool,
    seed: int,
    type: Literal["mass", "density", "count"] = "mass",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute histogram-based estimates and CIs for a single group.

    Depending on `type`, returns one of:
      - 'mass':  p_i = count_i / count.sum()        (∑ p_i = 1)
      - 'density': f_i = count_i / (N * width)      (∫ f = 1)
      - 'count':          raw count_i               (∑ count_i = N)

    Confidence‐intervals are scaled into the same units.

    Parameters
    ----------
    data : np.ndarray
        1D array of observations for this group.
    x : np.ndarray
        Grid midpoints (length = n_bins).
    ci_method : {'dkw','asymptotic','bootstrap'}
        Method for pointwise CIs.
    ci_alpha : float
        Two‐sided significance level.
    ci_bootstrap_samples : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        NumPy RNG for bootstrap.
    use_jax : bool
        Whether to use JAX for bootstrap (if available).
    seed : int
        Seed for JAX PRNGKey.
    type : {'mass','density','count'}, default 'mass'
        What to return in `f_vals`.

    Returns
    -------
    f_vals : np.ndarray, shape (n_bins,)
        Estimated mass, density, or counts.
    l : np.ndarray, shape (n_bins,)
        Lower confidence bound (same units).
    u : np.ndarray, shape (n_bins,)
        Upper confidence bound.
    """
    # build bin edges and counts
    edges = np.linspace(x[0], x[-1], x.size + 1)
    counts, _ = np.histogram(data, bins=edges)
    N = data.size
    width = edges[1] - edges[0]

    # choose output scale
    if type == "count":
        f_vals = counts.astype(float)
    elif type == "mass":
        total = counts.sum()
        f_vals = counts / total if total > 0 else np.zeros_like(counts, float)
    else:  # density
        f_vals = counts / (N * width)

    # build CI in same scale
    if ci_method == "dkw":
        eps = np.sqrt(np.log(2.0 / ci_alpha) / (2 * N))
        if type == "density":
            half = eps / width
        elif type == "mass":
            half = eps
        else:  # count
            half = eps * N
        l = np.clip(f_vals - half, 0, None)
        u = f_vals + half

    elif ci_method == "asymptotic":
        p = counts / N
        if type == "density":
            se = np.sqrt(p * (1 - p) / N) / width
        elif type == "mass":
            se = np.sqrt(p * (1 - p) / N)
        else:  # count
            se = np.sqrt(p * (1 - p) * N)
        half = norm.ppf(1 - ci_alpha / 2) * se
        l = np.clip(f_vals - half, 0, None)
        u = f_vals + half

    else:  # bootstrap
        if use_jax and JAX_AVAILABLE:
            low, high = _ci_bootstrap_epdf_jax(
                data, x, ci_alpha, ci_bootstrap_samples, seed
            )
        else:
            low, high = _ci_bootstrap_epdf_numpy(
                data, x, ci_alpha, ci_bootstrap_samples, rng
            )

        # low/high returned in *density* units by bootstrap helper
        if type == "count":
            low = low * N * width
            high = high * N * width
        elif type == "mass":
            # convert densities→counts→normalize
            c_low = low * N * width
            c_high = high * N * width
            denom_low = c_low.sum() or 1.0
            denom_high = c_high.sum() or 1.0
            low = c_low / denom_low
            high = c_high / denom_high

        l, u = low, high

    return f_vals, l, u


def _compute_id_histogram(
    df,
    y: str,
    id_col: str,
    x: np.ndarray,
    type: Literal["mass", "density", "count"] = "mass",
) -> Tuple[np.ndarray, List[Any]]:
    """
    Compute per-ID histogram estimates (mass, density, or counts).

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
        Column of values.
    id_col : str
        Column of ID labels.
    x : np.ndarray
        Grid midpoints.
    type : {'mass','density','count'}, default 'mass'

    Returns
    -------
    f_id : np.ndarray, shape (n_bins, n_ids)
        Histogram estimates for each ID.
    id_labels : list
        Unique IDs.
    """
    id_labels = df[id_col].unique().tolist()
    n_bins = x.size
    f_id = np.empty((n_bins, len(id_labels)))
    edges = np.linspace(x[0], x[-1], n_bins + 1)
    width = edges[1] - edges[0]

    for i, iid in enumerate(id_labels):
        data_i = df[df[id_col] == iid][y].to_numpy(dtype=float)
        counts, _ = np.histogram(data_i, bins=edges)
        N = data_i.size

        if type == "count":
            f_id[:, i] = counts.astype(float)
        elif type == "mass":
            total = counts.sum()
            f_id[:, i] = counts / total if total > 0 else np.zeros_like(counts, float)
        else:  # density
            f_id[:, i] = counts / (N * width)

    return f_id, id_labels


def _compute_envelope(
    arr: np.ndarray,
    grp_keys: List[Any],
    df,
    id_col: str,
    id_labels: List[Any],
    envelope_method: Literal["percentile", "minmax", "bootstrap", "mad", "asymmad"],
    env_quantiles: Tuple[float, float],
    env_bootstrap_samples: int,
    envelope_scale: float,
    envelope_scale_lower: float,
    envelope_scale_upper: float,
    rng: np.random.Generator,
    use_jax: bool,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ID-level variability envelope for each group.
    """
    n_bins, n_groups = arr.shape[0], len(grp_keys)
    env_l = np.empty((n_bins, n_groups))
    env_u = np.empty((n_bins, n_groups))
    for j, key in enumerate(grp_keys):
        ids = (
            df[df[id_col] == key][id_col].unique().tolist()
            if key in df.columns
            else id_labels
        )
        idxs = [id_labels.index(i) for i in ids]
        sub = arr[:, idxs]
        if envelope_method == "percentile":
            env_l[:, j], env_u[:, j] = np.percentile(
                sub, [100 * env_quantiles[0], 100 * env_quantiles[1]], axis=1
            )
        elif envelope_method == "minmax":
            env_l[:, j], env_u[:, j] = sub.min(axis=1), sub.max(axis=1)
        elif envelope_method == "mad":
            med = np.median(sub, axis=1)
            mad = np.median(np.abs(sub - med[:, None]), axis=1)
            env_l[:, j] = np.clip(med - envelope_scale * mad, 0, None)
            env_u[:, j] = np.clip(med + envelope_scale * mad, 0, None)
        elif envelope_method == "asymmad":
            med = np.median(sub, axis=1)
            dev = sub - med[:, None]
            mad_lo = np.median(np.abs(np.where(dev < 0, dev, 0)), axis=1)
            mad_hi = np.median(np.abs(np.where(dev > 0, dev, 0)), axis=1)
            env_l[:, j] = np.clip(med - envelope_scale_lower * mad_lo, 0, None)
            env_u[:, j] = np.clip(med + envelope_scale_upper * mad_hi, 0, None)
        else:
            if use_jax and JAX_AVAILABLE:
                env_l[:, j], env_u[:, j] = _env_bootstrap_jax(
                    sub, env_quantiles[0], env_quantiles[1], env_bootstrap_samples, seed
                )
            else:
                env_l[:, j], env_u[:, j] = _env_bootstrap_numpy(
                    sub, env_quantiles[0], env_quantiles[1], env_bootstrap_samples, rng
                )
    return env_l, env_u


# -------------------
# EPDFResult container
# -------------------


class HistogramResult:
    """
    Container for histogram-based estimates (mass, density, or counts).

    Attributes
    ----------
    x : np.ndarray, shape (n_bins,)
        Grid midpoints.
    f : np.ndarray, shape (n_bins, n_groups)
        Estimated mass/density/count per group.
    l : np.ndarray, shape (n_bins, n_groups)
        Lower CI bounds.
    u : np.ndarray, shape (n_bins, n_groups)
        Upper CI bounds.
    group_labels : list
        Labels for each group.
    type : str
        One of 'mass','density','count'.
    y_id : np.ndarray or None, shape (n_bins, n_ids)
        Individual-ID histograms, if requested.
    id_labels : list or None
        Labels for IDs.
    env_lower : np.ndarray or None, shape (n_bins, n_groups)
        Lower envelope of ID variability.
    env_upper : np.ndarray or None, shape (n_bins, n_groups)
        Upper envelope of ID variability.
    """

    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        group_labels: List[Any],
        type: str,
        y_id: Optional[np.ndarray] = None,
        id_labels: Optional[List[Any]] = None,
        env_lower: Optional[np.ndarray] = None,
        env_upper: Optional[np.ndarray] = None,
    ):
        self.x = x
        self.f = f
        self.l = l
        self.u = u
        self.group_labels = group_labels
        self.type = type
        self.y_id = y_id
        self.id_labels = id_labels
        self.env_lower = env_lower
        self.env_upper = env_upper

    def __repr__(self) -> str:
        base = f"HistogramResult(type={self.type!r}, x={self.x.shape}, f={self.f.shape}, groups={self.group_labels})"
        if self.id_labels is not None:
            base = base[:-1] + f", ids={self.id_labels})"
        return base

    def plot(
        self,
        ax=None,
        colors: Optional[List[str]] = None,
        ci_alpha: Optional[float] = None,
        envelope: bool = True,
        **plot_kwargs,
    ):
        """
        Plot group-level empirical probability density functions with confidence intervals and optional ID envelope.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis object to draw the plot on. If None, a new figure and axis are created.
        colors : list of str, optional
            List of colors for each group curve. Colors are cycled if fewer than number of groups.
        ci_alpha : float, optional
            Alpha transparency level for confidence interval shading. Default is 0.3.
        envelope : bool, default True
            If True and ID variability envelope is available, overlay the envelope band.
        **plot_kwargs :
            Additional keyword arguments passed to `ax.plot` for the density curves.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """

        if ax is None:
            fig, ax = plt.subplots()
        if colors is None:
            prop_cycle = plt.rcParams.get("axes.prop_cycle")
            colors = prop_cycle.by_key().get("color", None)
        n_groups = self.f.shape[1]
        alpha_ci = ci_alpha if ci_alpha is not None else 0.3

        for idx, label in enumerate(self.group_labels):
            col = colors[idx % len(colors)] if colors is not None else None
            ax.plot(self.x, self.f[:, idx], label=str(label), color=col, **plot_kwargs)
            ax.fill_between(
                self.x, self.l[:, idx], self.u[:, idx], alpha=alpha_ci, color=col
            )
            if envelope and self.env_lower is not None:
                ax.fill_between(
                    self.x,
                    self.env_lower[:, idx],
                    self.env_upper[:, idx],
                    alpha=alpha_ci / 2,
                    color=col,
                    linestyle="--",
                    linewidth=0,
                )
        ax.legend()
        return ax


# -------------------
# Main EPDF function
# -------------------


def histogram(
    df,
    y: str,
    id: Optional[str] = None,
    group: Optional[str] = None,
    ci_method: Literal["dkw", "asymptotic", "bootstrap"] = "dkw",
    ci_alpha: float = 0.05,
    ci_bootstrap_samples: int = 1000,
    envelope_method: Literal[
        "percentile", "minmax", "bootstrap", "mad", "asymmad"
    ] = "percentile",
    env_quantiles: Tuple[float, float] = (0.025, 0.975),
    env_bootstrap_samples: int = 500,
    envelope_scale: float = 1.0,
    envelope_scale_lower: float = 1.0,
    envelope_scale_upper: float = 1.0,
    x0: Optional[float] = None,
    x1: Optional[float] = None,
    bins: Optional[int] = None,
    use_jax: bool = False,
    seed: int = 0,
    type: Literal["mass", "density", "count"] = "mass",
) -> HistogramResult:
    """
    Compute a grouped histogram estimate (mass, density, or count) with optional ID curves,
    confidence intervals, and envelope bands.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
    id : str or None
    group : str or None
    ci_method, ci_alpha, ci_bootstrap_samples : see _compute_group_histogram
    envelope_method, env_quantiles, env_bootstrap_samples, envelope_scale… : see _compute_group_histogram
    x0, x1, bins : as in _compute_x
    use_jax : bool
    seed : int
    type : {'mass','density','count'}, default 'mass'

    Returns
    -------
    result : HistogramResult
    """
    values = df[y].to_numpy(dtype=float)
    x = _compute_x(values, x0, x1, bins)

    # determine group keys…
    if group is not None:
        grp_keys = df[group].unique().tolist()
    elif id is not None:
        grp_keys = df[id].unique().tolist()
    else:
        grp_keys = ["all"]

    # allocate
    f_out = np.empty((x.size, len(grp_keys)))
    l_out = np.empty_like(f_out)
    u_out = np.empty_like(f_out)
    rng = np.random.default_rng(seed)

    # per-group
    for j, key in enumerate(grp_keys):
        if group is None and id is None:
            data_j = values
        elif group is not None:
            data_j = df[df[group] == key][y].to_numpy(dtype=float)
        else:
            data_j = df[df[id] == key][y].to_numpy(dtype=float)

        f_out[:, j], l_out[:, j], u_out[:, j] = _compute_group_histogram(
            data_j,
            x,
            ci_method,
            ci_alpha,
            ci_bootstrap_samples,
            rng,
            use_jax,
            seed,
            type,
        )

    # per-ID curves & envelope… exactly as before, passing along `type`
    if id is not None:
        f_id, id_labels = _compute_id_histogram(df, y, id, x, type)
        env_l, env_u = _compute_envelope(
            f_id,
            grp_keys,
            df,
            id,
            id_labels,
            envelope_method,
            env_quantiles,
            env_bootstrap_samples,
            envelope_scale,
            envelope_scale_lower,
            envelope_scale_upper,
            rng,
            use_jax,
            seed,
        )
    else:
        f_id = id_labels = env_l = env_u = None

    return HistogramResult(
        x=x,
        f=f_out,
        l=l_out,
        u=u_out,
        group_labels=grp_keys,
        type=type,
        y_id=f_id,
        id_labels=id_labels,
        env_lower=env_l,
        env_upper=env_u,
    )


### Permutation ANOVA and post-hoc


# -----------------------------------------------------------------------------
# Core JAX ANOVA helpers
# -----------------------------------------------------------------------------


@partial(jit, static_argnums=(2,))
def compute_ss_jax(
    y: jnp.ndarray, groups: jnp.ndarray, num_groups: int, eps: float = 1e-12
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    μ = jnp.mean(y)
    gs = segment_sum(y, groups, num_segments=num_groups)
    gc = segment_sum(jnp.ones_like(y), groups, num_segments=num_groups)
    gm = gs / gc
    resid_within = y - gm[groups]
    resid_total = y - μ
    ssw = jnp.sum(resid_within**2) + eps
    sst = jnp.sum(resid_total**2) + eps
    return ssw, sst, μ


@partial(jit, static_argnums=(1,))
def compute_logL_jax(ss: jnp.ndarray, n: int) -> jnp.ndarray:
    return -0.5 * n * (jnp.log(2 * jnp.pi) + jnp.log(ss / n) + 1)


@partial(jit, static_argnums=(2,))
def compute_anova_stats_jax(
    y: jnp.ndarray, groups: jnp.ndarray, num_groups: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ssw, sst, μ = compute_ss_jax(y, groups, num_groups)
    n = y.shape[0]
    k = num_groups
    dfb = k - 1
    dfw = n - k
    ssb = sst - ssw
    msb = ssb / dfb
    msw = ssw / dfw
    F = msb / msw
    omega2 = (ssb - dfb * msw) / (sst + msw)
    return F, omega2, μ


# -----------------------------------------------------------------------------
# FUSED ANOVA (one jit, one compile)
# -----------------------------------------------------------------------------


@partial(jit, static_argnums=(2,))
def jax_anova_all(
    y: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    perm_keys: jnp.ndarray,
    boot_keys: jnp.ndarray,
    ci: float = 0.95,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    # observed F & ω²
    F_obs, ω2_obs, μ = compute_anova_stats_jax(y, groups, num_groups)

    # 1) F permutations (Freedman–Lane residual)
    resid = y - μ

    def _oneF(k):
        pr = random.permutation(k, resid)
        yp = μ + pr
        Fp, _, _ = compute_anova_stats_jax(yp, groups, num_groups)
        return Fp

    perm_Fs = vmap(_oneF)(perm_keys)

    # 2) ω² bootstrap
    def _oneω2(k):
        rb = random.choice(k, resid, shape=resid.shape, replace=True)
        yb = μ + rb
        _, w2b, _ = compute_anova_stats_jax(yb, groups, num_groups)
        return w2b

    ω2_boot = vmap(_oneω2)(boot_keys)

    # 3) percentile CI
    lo = jnp.percentile(ω2_boot, (1 - ci) / 2 * 100)
    hi = jnp.percentile(ω2_boot, (1 + ci) / 2 * 100)

    return F_obs, perm_Fs, ω2_obs, ω2_boot, lo, hi


# -----------------------------------------------------------------------------
# High-level ANOVA wrapper
# -----------------------------------------------------------------------------


def residual_permutation_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    num_permutations: int = 1000,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    return_distributions: bool = False,
) -> dict:
    if df[group_col].nunique() < 2:
        raise ValueError("Need at least two groups.")
    y = pd.to_numeric(df[value_col], errors="raise").values
    labels, uniques = pd.factorize(df[group_col])
    n, k = len(y), len(uniques)

    y_jax = jnp.array(y)
    grp_jax = jnp.array(labels)

    perm_keys = random.split(random.PRNGKey(seed), num_permutations)
    boot_keys = random.split(random.PRNGKey(seed + 1), num_bootstrap)

    F_obs_j, perm_Fs_j, ω2_obs_j, ω2_boot_j, lo_j, hi_j = jax_anova_all(
        y_jax, grp_jax, k, perm_keys, boot_keys, ci
    )

    # back to Python scalars/arrays
    F_obs = float(F_obs_j)
    ω2_obs = float(ω2_obs_j)
    ci_lo = float(lo_j)
    ci_hi = float(hi_j)
    perm_Fs = np.array(perm_Fs_j)
    ω2_boot = np.array(ω2_boot_j)

    # AIC / BIC as before
    logL0 = float(compute_logL_jax(jnp.array(np.sum((y - y.mean()) ** 2)), n))
    logL1 = float(compute_logL_jax(jnp.array(np.sum((y - y_jax.mean()) ** 2)), n))
    AIC0, AIC1 = 2 * 1 - 2 * logL0, 2 * k - 2 * logL1
    BIC0, BIC1 = np.log(n) * 1 - 2 * logL0, np.log(n) * k - 2 * logL1

    p_value = float((perm_Fs >= F_obs).sum() + 1) / (num_permutations + 1)
    alpha = 1 / np.sqrt(n)
    sig = p_value < alpha

    out = {
        "F_statistic": F_obs,
        "p_value": p_value,
        "omega_squared": ω2_obs,
        "omega2_ci": (ci_lo, ci_hi),
        "alpha_adaptive": alpha,
        "significant_adaptive": sig,
        "AIC_null": AIC0,
        "AIC_full": AIC1,
        "delta_AIC": AIC0 - AIC1,
        "BIC_null": BIC0,
        "BIC_full": BIC1,
        "delta_BIC": BIC0 - BIC1,
    }
    if return_distributions:
        out["perm_Fs"] = perm_Fs
        out["omega2_bootstrap"] = ω2_boot
    return out


# -----------------------------------------------------------------------------
# Pairwise post-hoc kernel (static sizes fix tracer error)
# -----------------------------------------------------------------------------


@partial(jit, static_argnums=(5, 6))
def jax_pairwise_permutation_cohend(
    y1: jnp.ndarray,
    y2: jnp.ndarray,
    perm_keys: jnp.ndarray,
    boot_keys: jnp.ndarray,
    ci: float,
    c1: int,
    c2: int,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    c1, c2 are Python ints (group sizes) and marked static_argnums.
    """
    obs_diff = jnp.mean(y1) - jnp.mean(y2)
    obs_d = obs_diff / jnp.sqrt(
        ((c1 - 1) * jnp.var(y1, ddof=1) + (c2 - 1) * jnp.var(y2, ddof=1))
        / (c1 + c2 - 2)
    )

    pooled = jnp.concatenate([y1, y2])

    # permutation distribution
    def _perm(key):
        p = random.permutation(key, pooled)
        return jnp.mean(p[:c1]) - jnp.mean(p[c1 : c1 + c2])

    perm_diffs = vmap(_perm)(perm_keys)
    p_raw = (jnp.sum(jnp.abs(perm_diffs) >= jnp.abs(obs_diff)) + 1) / (
        perm_keys.shape[0] + 1
    )

    # bootstrap Cohen’s d
    def _boot(key):
        b1 = random.choice(key, y1, shape=(c1,), replace=True)
        b2 = random.choice(key, y2, shape=(c2,), replace=True)
        s1 = jnp.var(b1, ddof=1)
        s2 = jnp.var(b2, ddof=1)
        psd = jnp.sqrt(((c1 - 1) * s1 + (c2 - 1) * s2) / (c1 + c2 - 2))
        return (jnp.mean(b1) - jnp.mean(b2)) / psd

    d_boot = vmap(_boot)(boot_keys)
    lo = jnp.percentile(d_boot, (1 - ci) / 2 * 100)
    hi = jnp.percentile(d_boot, (1 + ci) / 2 * 100)

    return obs_diff, perm_diffs, p_raw, obs_d, d_boot, lo, hi


# -----------------------------------------------------------------------------
# P-value adjustment
# -----------------------------------------------------------------------------


def adjust_pvalues(pvals: np.ndarray, method: str = "bonferroni") -> np.ndarray:
    p = np.asarray(pvals)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    idx = np.argsort(p)
    sp = p[idx]
    cum = np.minimum.accumulate((m / np.arange(1, m + 1)) * sp[::-1])[::-1]
    out = np.empty(m)
    out[idx] = np.minimum(cum, 1.0)
    return out


# -----------------------------------------------------------------------------
# High-level post-hoc wrapper
# -----------------------------------------------------------------------------


def posthoc_pairwise_permutation(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    num_permutations: int = 1000,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
    p_adjust: str = "bonferroni",
    seed: int = 42,
) -> pd.DataFrame:
    y = pd.to_numeric(df[value_col], errors="raise").values
    labels, uniques = pd.factorize(df[group_col])

    perm_keys = random.split(random.PRNGKey(seed), num_permutations)
    boot_keys = random.split(random.PRNGKey(seed + 1), num_bootstrap)

    rows = []
    for i, j in combinations(range(len(uniques)), 2):
        y1 = jnp.array(y[labels == i])
        y2 = jnp.array(y[labels == j])
        c1, c2 = y1.shape[0], y2.shape[0]

        od, pdiffs, p_raw, od_d, dboot, lo, hi = jax_pairwise_permutation_cohend(
            y1, y2, perm_keys, boot_keys, ci, c1, c2
        )

        rows.append(
            {
                "group1": uniques[i],
                "group2": uniques[j],
                "mean_diff": float(od),
                "cohen_d": float(od_d),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "p_raw": float(p_raw),
            }
        )

    df_out = pd.DataFrame(rows)
    df_out["p_adj"] = adjust_pvalues(df_out["p_raw"].values, method=p_adjust)
    return df_out

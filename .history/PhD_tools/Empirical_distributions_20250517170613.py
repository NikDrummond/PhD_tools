import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any
from scipy.stats import norm

# --- Grid & CI Utilities ---
def compute_grid(data: np.ndarray, x0: Optional[float], x1: Optional[float], bins: Optional[int]) -> np.ndarray:
    data_min, data_max = np.min(data), np.max(data)
    x0 = x0 if x0 is not None else data_min
    x1 = x1 if x1 is not None else data_max
    buffer = 0.05 * (x1 - x0)
    x0 -= buffer
    x1 += buffer

    if bins is None:
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        bin_width = 2 * iqr / np.cbrt(data.size)
        bins = max(10, int((x1 - x0) / bin_width))

    return np.linspace(x0, x1, bins)

def dkw_ci(N: int, alpha: float) -> float:
    return np.sqrt(np.log(2.0 / alpha) / (2 * N))

def asymptotic_ci(p: np.ndarray, N: int, alpha: float, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / N) * scale
    margin = z * se
    return np.clip(p - margin, 0, None), p + margin

# --- Histogram and EDF Functions ---
def edf_numpy(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(data), x, side="right") / data.size

def histogram_numpy(data: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    edges = np.linspace(x[0], x[-1], len(x) + 1)
    counts, _ = np.histogram(data, bins=edges)
    return counts, edges[1] - edges[0]

# --- Per-ID Matrix Construction ---
def binned_matrix(df: pd.DataFrame, y: str, id_col: str, x: np.ndarray, kind: str) -> Tuple[np.ndarray, List[Any]]:
    edges = np.linspace(x[0], x[-1], len(x) + 1)
    bin_idx = np.digitize(df[y].to_numpy(), edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(x) - 1)

    id_vals = df[id_col].to_numpy()
    id_labels, id_idx = np.unique(id_vals, return_inverse=True)
    n_bins, n_ids = len(x), len(id_labels)

    mat = np.zeros((n_bins, n_ids), dtype=float)
    np.add.at(mat, (bin_idx, id_idx), 1)

    if kind == "cumulative":
        norm = np.sum(mat, axis=0)
        mat = np.cumsum(mat, axis=0)
        norm[norm == 0] = 1
        mat /= norm
    elif kind in ("mass", "density"):
        norm = np.sum(mat, axis=0)
        norm[norm == 0] = 1
        if kind == "density":
            mat /= norm * (edges[1] - edges[0])
        else:
            mat /= norm
    return mat, id_labels.tolist()

# --- Bootstrap over IDs ---
def bootstrap_ci_id(mat: np.ndarray, alpha: float, samples: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n_bins, n_ids = mat.shape
    boot = np.zeros((n_bins, samples))
    for b in range(samples):
        resample_idx = rng.integers(0, n_ids, n_ids)
        boot[:, b] = mat[:, resample_idx].mean(axis=1)
    return np.percentile(boot, 100 * alpha / 2, axis=1), np.percentile(boot, 100 * (1 - alpha / 2), axis=1)

# --- Envelope Bands ---
def compute_envelope(arr: np.ndarray, method: str = "percentile", q: Tuple[float, float] = (0.025, 0.975), scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if method == "percentile":
        return np.percentile(arr, 100 * q[0], axis=1), np.percentile(arr, 100 * q[1], axis=1)
    elif method == "minmax":
        return arr.min(axis=1), arr.max(axis=1)
    elif method == "mad":
        med = np.median(arr, axis=1)
        mad = np.median(np.abs(arr - med[:, None]), axis=1)
        return np.clip(med - scale * mad, 0, None), med + scale * mad
    elif method == "asymmetric_mad":
        med = np.median(arr, axis=1)
        mad_lower = np.median(np.abs(arr - med[:, None]) * (arr < med[:, None]), axis=1)
        mad_upper = np.median(np.abs(arr - med[:, None]) * (arr >= med[:, None]), axis=1)
        return np.clip(med - scale * mad_lower, 0, None), med + scale * mad_upper
    else:
        raise ValueError(f"Unknown envelope method: {method}")

# --- Main API ---
class DistributionResult:
    def __init__(self, x, y, l, u, group_labels, kind, env_l=None, env_u=None):
        self.x = x
        self.y = y
        self.l = l
        self.u = u
        self.group_labels = group_labels
        self.kind = kind
        self.env_l = env_l
        self.env_u = env_u

    def plot(self, ax=None, colors=None, alpha=0.3, envelope=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, label in enumerate(self.group_labels):
            color = colors[i % len(colors)]
            ax.plot(self.x, self.y[:, i], label=str(label), color=color, **kwargs)
            ax.fill_between(self.x, self.l[:, i], self.u[:, i], alpha=alpha, color=color)
            if envelope and self.env_l is not None:
                ax.fill_between(self.x, self.env_l[:, i], self.env_u[:, i], alpha=alpha/2, color=color, linestyle='--')

        ax.set_ylabel(self.kind)
        ax.legend()
        return ax

def compute_distribution(df: pd.DataFrame, y: str, kind: str = "cumulative", group: Optional[str] = None, id_col: Optional[str] = None,
                          ci_method: str = "bootstrap", alpha: float = 0.05, samples: int = 1000,
                          envelope_method: str = "percentile", x0=None, x1=None, bins=None,
                          weight_by_obs: bool = False) -> DistributionResult:

    data = df[y].to_numpy()
    x = compute_grid(data, x0, x1, bins)
    rng = np.random.default_rng()
    groups = df[group].unique().tolist() if group else ["all"]

    y_out, l_out, u_out = [], [], []
    env_l_out, env_u_out = [], []

    for g in groups:
        df_g = df[df[group] == g] if group else df

        if id_col:
            mat, _ = binned_matrix(df_g, y, id_col, x, kind)
            if weight_by_obs:
                weights = df_g.groupby(id_col)[y].count().reindex(mat.shape[1]*[None]).fillna(0).to_numpy()
                weights = weights / weights.sum()
                y_g = mat @ weights
            else:
                y_g = mat.mean(axis=1)
            l_g, u_g = bootstrap_ci_id(mat, alpha, samples, rng)
            env_l_g, env_u_g = compute_envelope(mat, method=envelope_method)
        else:
            y_vals = df_g[y].to_numpy()
            if kind == "cumulative":
                y_g = edf_numpy(y_vals, x)
            else:
                counts, width = histogram_numpy(y_vals, x)
                if kind == "density":
                    y_g = counts / (y_vals.size * width)
                elif kind == "mass":
                    y_g = counts / y_vals.size
                elif kind == "count":
                    y_g = counts

            if ci_method == "dkw" and kind == "cumulative":
                eps = dkw_ci(len(y_vals), alpha)
                l_g = np.clip(y_g - eps, 0, None)
                u_g = y_g + eps
            elif ci_method == "asymptotic" and kind == "cumulative":
                l_g, u_g = asymptotic_ci(y_g, len(y_vals), alpha)
            else:
                l_g, u_g = bootstrap_ci_id(np.expand_dims(y_g, 1), alpha, samples, rng)
            env_l_g = env_u_g = np.full_like(y_g, np.nan)

        y_out.append(y_g)
        l_out.append(l_g)
        u_out.append(u_g)
        env_l_out.append(env_l_g)
        env_u_out.append(env_u_g)

    return DistributionResult(x=x,
                              y=np.column_stack(y_out),
                              l=np.column_stack(l_out),
                              u=np.column_stack(u_out),
                              group_labels=groups,
                              kind=kind,
                              env_l=np.column_stack(env_l_out),
                              env_u=np.column_stack(env_u_out))


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

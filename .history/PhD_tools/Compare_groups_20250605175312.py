"""
refactored_stats.py
Author: <you>

Fast JAX-backed one-way ANOVA for linear and circular data
(with permutation testing, ω², and bootstrap CIs).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
from scipy.stats import f as scipy_f
import GeoJax as gj        
from functools import partial

# --------------------------------------------------------------------------- #
# Public result container
# --------------------------------------------------------------------------- #

@dataclass
class StatisticalResult:
    test_statistic: float
    p_value: float | None
    effect_size: float
    confidence_interval: Tuple[float, float] | None = None
    bootstraps: np.ndarray | None = None
    method: str = ""


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _encode_groups(groups_np: np.ndarray) -> np.ndarray:
    """Map arbitrary group labels → contiguous int32 codes (stable order)."""
    _, codes = np.unique(groups_np, return_inverse=True)
    return codes.astype(np.int32)


def _split_by_codes(values: jnp.ndarray, codes: np.ndarray) -> List[jnp.ndarray]:
    """Return list of jnp.ndarray slices grouped by int codes (0..k-1)."""
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    sorted_vals = values[order]

    if len(sorted_vals) == 0:
        return []

    # locate boundaries where code changes
    idx = np.nonzero(sorted_codes[:-1] != sorted_codes[1:])[0] + 1
    splits = np.split(sorted_vals, idx)          # returns List[np.ndarray]
    return [jnp.asarray(s) for s in splits]      # make JAX arrays


# --------------------------------------------------------------------------- #
# Circular core
# --------------------------------------------------------------------------- #

@jit
def _rvl(x: jnp.ndarray) -> float:
    """Mean resultant vector length (R̄)."""
    n = x.size
    if n == 0:
        return 0.0
    s = jnp.sum(jnp.sin(x))
    c = jnp.sum(jnp.cos(x))
    return jnp.sqrt(s**2 + c**2) / n


@partial(jit, static_argnames=("circmean", "rvl"))
def _ss_circular_between(groups: List[jnp.ndarray],
                         circmean: Callable[[jnp.ndarray], float] = gj.circmean,
                         rvl: Callable[[jnp.ndarray], float] = _rvl) -> float:
    """Dispersion analogue of SS_between (Berens 2018)."""
    if not groups:
        return 0.0
    non_empty = [g for g in groups if g.size]
    if not non_empty:
        return 0.0
    grand_mean = circmean(jnp.concatenate(non_empty))
    def term(g):
        n = g.size
        if n == 0:
            return 0.0
        return n * rvl(g) * jnp.cos(circmean(g) - grand_mean)
    return jnp.sum(jnp.array([term(g) for g in groups]))


@partial(jit, static_argnames=("circmean",))
def _ss_circular_within(groups: List[jnp.ndarray],
                        circmean: Callable[[jnp.ndarray], float] = gj.circmean) -> float:
    """Dispersion analogue of SS_within."""
    def term(g):
        if g.size == 0:
            return 0.0
        theta = circmean(g)
        return jnp.sum(1 - jnp.cos(g - theta))
    return jnp.sum(jnp.array([term(g) for g in groups]))


# --------------------------------------------------------------------------- #
# Linear core (JAX)
# --------------------------------------------------------------------------- #

@jit
def _ss_linear(groups: List[jnp.ndarray]) -> Tuple[float, float, int, int]:
    valid = [g for g in groups if g.size]
    if not valid:
        return 0.0, 0.0, 0, 0
    all_vals = jnp.concatenate(valid)
    grand = jnp.mean(all_vals)
    ss_between = jnp.sum(jnp.array([g.size * (jnp.mean(g) - grand)**2 for g in valid]))
    ss_within  = jnp.sum(jnp.array([jnp.sum((g - jnp.mean(g))**2) for g in valid]))
    return ss_between, ss_within, all_vals.size, len(valid)


# --------------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------------- #


@jit
def _omega2(ss_b, ss_w, n, k):
    """JAX-compatible ω² effect size."""
    df_b = k - 1
    df_w = n - k

    ms_w = jnp.where(df_w > 0, ss_w / df_w, jnp.nan)          # avoid /0
    denom = ss_b + ss_w + ms_w
    num   = ss_b - df_b * ms_w

    bad = (k <= 1) | (n <= k) | (df_w == 0) | (denom == 0)
    return jnp.where(bad, jnp.nan, num / denom)



# --------------------------------------------------------------------------- #
# Generic permutation engine
# --------------------------------------------------------------------------- #

def _perm_test(all_vals: jnp.ndarray,
               codes: np.ndarray,
               stat_fn: Callable[[List[jnp.ndarray]], float],
               observed: float,
               n_perm: int) -> float:
    rng = np.random.default_rng()
    exceed = 0
    for _ in tqdm(range(n_perm), desc="Permuting", disable=n_perm < 200):
        shuffled = rng.permutation(codes)
        groups = _split_by_codes(all_vals, shuffled)
        s = stat_fn(groups)
        exceed += (s >= observed)
    return exceed / n_perm


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #

def _bootstrap(groups: List[jnp.ndarray],
               eff_fn: Callable[[List[jnp.ndarray]], float],
               n_boot: int,
               seed: int | None = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float32)
    for i in tqdm(range(n_boot), desc="Bootstrapping", disable=n_boot < 200):
        resamp = [jnp.asarray(rng.choice(g, size=g.size, replace=True)) if g.size else g
                  for g in groups]
        out[i] = float(eff_fn(resamp))
    ci = (np.percentile(out, 2.5), np.percentile(out, 97.5))
    return out, ci


# --------------------------------------------------------------------------- #
# High-level tests
# --------------------------------------------------------------------------- #

def _anova_linear(groups: List[jnp.ndarray]) -> Tuple[float, float]:
    ss_b, ss_w, n, k = _ss_linear(groups)
    if k <= 1 or n <= k:
        return jnp.nan, jnp.nan
    df_b, df_w = k - 1, n - k
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    F = jnp.where(ms_w == 0, jnp.inf, ms_b / ms_w)
    p = 1 - scipy_f.cdf(float(F), df_b, df_w) if jnp.isfinite(F) else 0.0
    return float(F), p


def _anova_circular(groups: List[jnp.ndarray],
                    circmean=gj.circmean,
                    rvl=_rvl) -> Tuple[float, float]:
    ss_b = _ss_circular_between(groups, circmean, rvl)
    ss_w = _ss_circular_within(groups, circmean)
    k = len(groups)
    n = int(jnp.sum(jnp.array([g.size for g in groups])))
    df_b, df_w = k - 1, n - k
    F = (df_w * ss_b) / (df_b * ss_w) if (df_b > 0 and ss_w > 0) else jnp.nan
    p = 1 - scipy_f.cdf(float(F), df_b, df_w) if jnp.isfinite(F) else jnp.nan
    return float(F), p


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #

def compare_groups(df: pd.DataFrame,
                   group_col: str,
                   value_col: str,
                   *,
                   method: str = "circular",       # "circular" | "linear"
                   use_permutations: bool = False,
                   n_iterations: int = 1_000,
                   bootstrap_ci: bool = False,
                   bootstrap_seed: int | None = None
                   ) -> StatisticalResult:
    """
    Fast one-way ANOVA (linear or circular) with optional permutation p-value
    and bootstrap CI for ω².
    """
    if method not in {"circular", "linear"}:
        raise ValueError("method must be 'circular' or 'linear'")

    vals_np  = df[value_col].to_numpy(dtype=np.float32)
    codes_np = _encode_groups(df[group_col].to_numpy())
    vals_jax = jnp.asarray(vals_np)

    groups = _split_by_codes(vals_jax, codes_np)

    if method == "linear":
        F_obs, p_asym = _anova_linear(groups)
        ss_b, ss_w, n, k = _ss_linear(groups)
        omega2 = float(_omega2(ss_b, ss_w, n, k))
        stat_name = "One-way ANOVA (linear)"
        stat_fn   = lambda g: _anova_linear(g)[0]
    else:  # circular
        F_obs, p_asym = _anova_circular(groups)
        ss_b = _ss_circular_between(groups)
        ss_w = _ss_circular_within(groups)
        n    = int(jnp.sum(jnp.array([g.size for g in groups])))
        k    = len(groups)
        omega2 = float(_omega2(ss_b, ss_w, n, k))
        stat_name = "Watson–Williams"
        stat_fn   = lambda g: _ss_circular_between(g)

    # optional permutation
    p_perm = None
    if use_permutations:
        p_perm = _perm_test(vals_jax, codes_np, stat_fn,
                            observed=F_obs if method == "linear" else ss_b,
                            n_perm=n_iterations)

    # optional bootstrap
    boots = ci = None
    if bootstrap_ci:
        boots, ci = _bootstrap(groups,
                               eff_fn=lambda g: _omega2(
                                   *_ss_linear(g) if method == "linear"
                                   else (_ss_circular_between(g),
                                         _ss_circular_within(g),
                                         int(jnp.sum(jnp.array([x.size for x in g]))),
                                         len(g))),
                               n_boot=n_iterations,
                               seed=bootstrap_seed)

    return StatisticalResult(test_statistic=F_obs,
                             p_value=p_perm if use_permutations else p_asym,
                             effect_size=omega2,
                             confidence_interval=ci,
                             bootstraps=boots,
                             method=f"{stat_name} + ω²")

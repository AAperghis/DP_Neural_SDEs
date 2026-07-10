"""Convergence analysis: how statistics stabilise with simulation time and sample count."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from thesis.statistics.ensemble_stats import (
    MomentResult,
    PSDResult,
    ensemble_moments,
    ensemble_psd,
)
from thesis.statistics.extreme_values import EVResult, pot_extreme_values


# ---------------------------------------------------------------------------
# Lightweight container for bootstrap CI bounds on maxima
# ---------------------------------------------------------------------------


@dataclass
class MaximaBounds:
    """Bootstrap CI bound on maxima summary statistics.

    Attributes
    ----------
    max_mean : (F,) — mean of per-realisation maxima.
    max_std  : (F,) — std of per-realisation maxima.
    """

    max_mean: np.ndarray
    max_std: np.ndarray


# ---------------------------------------------------------------------------
# Convergence with simulation time
# ---------------------------------------------------------------------------


@dataclass
class TimeConvergenceResult:
    """Statistics recomputed at increasing window durations.

    Attributes
    ----------
    windows : list[float]
        Window durations T_w in seconds.
    moments : list[MomentResult]
        Moment results per window.
    maxima : list[EVResult]
        Extreme value results per window.
    psd : list[PSDResult]
        PSD results per window.
    """

    windows: list[float]
    moments: list[MomentResult]
    maxima: list[EVResult]
    psd: list[PSDResult]


def convergence_time(
    data: np.ndarray,
    time: np.ndarray,
    dt: float,
    windows: Sequence[float] | None = None,
) -> TimeConvergenceResult:
    """Recompute ensemble statistics for increasing trajectory durations.

    Parameters
    ----------
    data : (N, T, F)
    time : (T,)
    dt : float
    windows : sequence of float, optional
        Duration values in seconds.  Defaults to
        [600, 1200, 1800, 3600, 5400, 7200, 10800].

    Returns
    -------
    TimeConvergenceResult
    """
    if windows is None:
        max_t = float(time[-1])
        windows = [float(w) for w in [600, 1200, 1800, 3600, 5400, 7200, 10800] if w <= max_t]
        if max_t not in windows:
            windows.append(max_t)

    moments_list: list[MomentResult] = []
    maxima_list: list[EVResult] = []
    psd_list: list[PSDResult] = []

    for tw in windows:
        idx_end = np.searchsorted(time, tw, side="right")
        sub = data[:, :idx_end, :]
        moments_list.append(ensemble_moments(sub))
        maxima_list.append(pot_extreme_values(sub, dt))
        psd_list.append(ensemble_psd(sub, dt))

    return TimeConvergenceResult(
        windows=list(windows),
        moments=moments_list,
        maxima=maxima_list,
        psd=psd_list,
    )


# ---------------------------------------------------------------------------
# Convergence with number of ensemble members
# ---------------------------------------------------------------------------


@dataclass
class SampleConvergenceResult:
    """Statistics recomputed at increasing ensemble sizes with bootstrap CIs.

    Attributes
    ----------
    sample_sizes : list[int]
    moments : list[MomentResult]
        Point estimate at each sample size.
    maxima : list[EVResult]
    psd : list[PSDResult]
    moments_ci_low : list[MomentResult]
        Lower CI bound (bootstrap).
    moments_ci_high : list[MomentResult]
        Upper CI bound (bootstrap).
    maxima_ci_low : list[MaximaBounds]
    maxima_ci_high : list[MaximaBounds]
    """

    sample_sizes: list[int]
    moments: list[MomentResult]
    maxima: list[EVResult]
    psd: list[PSDResult]
    moments_ci_low: list[MomentResult]
    moments_ci_high: list[MomentResult]
    maxima_ci_low: list[MaximaBounds]
    maxima_ci_high: list[MaximaBounds]


def convergence_samples(
    data: np.ndarray,
    dt: float,
    sample_sizes: Sequence[int] | None = None,
    n_bootstrap: int = 100,
    seed: int = 0,
    ci: float = 0.95,
) -> SampleConvergenceResult:
    """Recompute ensemble statistics for increasing ensemble sizes.

    At each size, bootstrap resampling is used to estimate confidence intervals.

    Parameters
    ----------
    data : (N, T, F)
    dt : float
    sample_sizes : sequence of int, optional
        Defaults to [5, 10, 20, 50] (capped at N).
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.
    seed : int
        RNG seed for reproducibility.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    SampleConvergenceResult
    """
    N = data.shape[0]
    if sample_sizes is None:
        sample_sizes = [s for s in [5, 10, 20, 50] if s <= N]
        if N not in sample_sizes:
            sample_sizes.append(N)

    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    moments_list: list[MomentResult] = []
    maxima_list: list[EVResult] = []
    psd_list: list[PSDResult] = []
    moments_lo: list[MomentResult] = []
    moments_hi: list[MomentResult] = []
    maxima_lo: list[MaximaBounds] = []
    maxima_hi: list[MaximaBounds] = []

    for n_sub in sample_sizes:
        # Point estimate: first n_sub members
        sub = data[:n_sub]
        moments_list.append(ensemble_moments(sub))
        maxima_list.append(pot_extreme_values(sub, dt))
        psd_list.append(ensemble_psd(sub, dt))

        # Bootstrap CIs on scalar moments and maxima stats
        boot_mean = []
        boot_var = []
        boot_max_mean = []
        boot_max_std = []

        for _ in range(n_bootstrap):
            idx = rng.choice(N, size=n_sub, replace=True)
            b = data[idx]
            bm = ensemble_moments(b)
            boot_mean.append(bm.mean_scalar)
            boot_var.append(bm.var_scalar)
            # Simple maxima summary (no full POT fit in bootstrap loop)
            b_maxima = np.max(b, axis=1)  # (n_sub, F)
            boot_max_mean.append(b_maxima.mean(axis=0))
            boot_max_std.append(b_maxima.std(axis=0, ddof=1))

        def _ci_bounds(samples):
            arr = np.stack(samples, axis=0)  # (n_bootstrap, F)
            lo = np.quantile(arr, alpha, axis=0)
            hi = np.quantile(arr, 1 - alpha, axis=0)
            return lo, hi

        mean_lo, mean_hi = _ci_bounds(boot_mean)
        var_lo, var_hi = _ci_bounds(boot_var)
        max_mean_lo, max_mean_hi = _ci_bounds(boot_max_mean)
        max_std_lo, max_std_hi = _ci_bounds(boot_max_std)

        # Pack into result objects (only scalar fields filled for CI bounds)
        ref_m = moments_list[-1]
        moments_lo.append(
            MomentResult(
                mean_t=ref_m.mean_t,
                var_t=ref_m.var_t,
                skew_t=ref_m.skew_t,
                kurt_t=ref_m.kurt_t,
                mean_scalar=mean_lo,
                var_scalar=var_lo,
                skew_scalar=ref_m.skew_scalar,
                kurt_scalar=ref_m.kurt_scalar,
            )
        )
        moments_hi.append(
            MomentResult(
                mean_t=ref_m.mean_t,
                var_t=ref_m.var_t,
                skew_t=ref_m.skew_t,
                kurt_t=ref_m.kurt_t,
                mean_scalar=mean_hi,
                var_scalar=var_hi,
                skew_scalar=ref_m.skew_scalar,
                kurt_scalar=ref_m.kurt_scalar,
            )
        )

        maxima_lo.append(MaximaBounds(max_mean=max_mean_lo, max_std=max_std_lo))
        maxima_hi.append(MaximaBounds(max_mean=max_mean_hi, max_std=max_std_hi))

    return SampleConvergenceResult(
        sample_sizes=list(sample_sizes),
        moments=moments_list,
        maxima=maxima_list,
        psd=psd_list,
        moments_ci_low=moments_lo,
        moments_ci_high=moments_hi,
        maxima_ci_low=maxima_lo,
        maxima_ci_high=maxima_hi,
    )

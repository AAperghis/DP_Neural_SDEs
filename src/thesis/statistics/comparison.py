"""Two-ensemble comparison metrics.

All functions take two (N, T, F) arrays (or pre-computed results)
and return per-feature comparison statistics.
"""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np
from scipy import stats as sp_stats

from thesis.statistics.ensemble_stats import (
    ensemble_moments,
    ensemble_psd,
    ensemble_cross_correlation,
    cross_correlation_error,
)
from thesis.statistics.extreme_values import pot_extreme_values


@dataclass
class ComparisonResult:
    """Per-feature comparison between two ensembles A and B.

    Attributes
    ----------
    feature_names : list[str]
    moment_rel_error : dict[str, np.ndarray]
        Relative error for each moment (mean, var, skew, kurt), shape (F,).
    ks_statistic : np.ndarray, (F,)
        KS test statistic on maxima distributions.
    ks_pvalue : np.ndarray, (F,)
        KS test p-value.
    spectral_l1 : np.ndarray, (F,)
        Relative L1 distance between ensemble-mean PSDs.
    wasserstein : np.ndarray, (F,)
        Wasserstein-1 distance on time-averaged marginals.
    mmd : np.ndarray, (F,)
        Maximum Mean Discrepancy with RBF kernel.
    cross_corr_error : float
        Mean absolute difference between off-diagonal entries of the
        ensemble-averaged cross-correlation matrices.
    """

    feature_names: list[str]
    moment_rel_error: dict[str, np.ndarray]
    ks_statistic: np.ndarray
    ks_pvalue: np.ndarray
    spectral_l1: np.ndarray
    wasserstein: np.ndarray
    mmd: np.ndarray
    cross_corr_error: float


# ---------------------------------------------------------------------------
# Helper: MMD with RBF kernel
# ---------------------------------------------------------------------------


def _rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased estimate of Maximum Mean Discrepancy^2 with Gaussian (RBF) kernel between 1-D sample vectors."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    if gamma is None:
        combined = np.concatenate([x, y])
        gamma = 1.0 / (2.0 * max(np.var(combined), 1e-12))

    def k(a, b):
        return np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)

    n, m = len(x), len(y)
    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)

    # Unbiased estimator: exclude diagonal for same-sample terms
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (
        Kxx.sum() / max(n * (n - 1), 1)
        + Kyy.sum() / max(m * (m - 1), 1)
        - 2.0 * Kxy.mean()
    )
    return float(max(mmd2, 0.0))  # clamp numerical negatives


# ---------------------------------------------------------------------------
# Moment comparison
# ---------------------------------------------------------------------------


def _safe_rel_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative error |a-b|/|b|, with safe handling of near-zero b."""
    denom = np.abs(b)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return np.abs(a - b) / denom


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------


def compare_ensembles(
    data_a: np.ndarray,
    data_b: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
) -> ComparisonResult:
    """Compare two ensembles across all metric families.

    Parameters
    ----------
    data_a, data_b : (N_a, T, F), (N_b, T, F)
        Must share the same T and F dimensions.
    dt : float
        Sampling interval in seconds.
    feature_names : list[str], optional
        Names for reporting.

    Returns
    -------
    ComparisonResult
    """
    _, T_a, F = data_a.shape
    _, T_b, _ = data_b.shape
    T = min(T_a, T_b)
    data_a = data_a[:, :T, :]
    data_b = data_b[:, :T, :]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    # --- Moments ---
    mom_a = ensemble_moments(data_a)
    mom_b = ensemble_moments(data_b)

    moment_rel = {
        "mean": _safe_rel_error(mom_a.mean_scalar, mom_b.mean_scalar),
        "var": _safe_rel_error(mom_a.var_scalar, mom_b.var_scalar),
        "skew": _safe_rel_error(mom_a.skew_scalar, mom_b.skew_scalar),
        "kurt": _safe_rel_error(mom_a.kurt_scalar, mom_b.kurt_scalar),
    }

    # --- Maxima KS test (skip features where POT fit failed) ---
    max_a = pot_extreme_values(data_a, dt)
    max_b = pot_extreme_values(data_b, dt)

    ks_stat = np.full(F, np.nan)
    ks_pval = np.full(F, np.nan)
    for f in range(F):
        col_a = max_a.observed_maxima[:, f]
        col_b = max_b.observed_maxima[:, f]
        if np.any(np.isnan(col_a)) or np.any(np.isnan(col_b)):
            continue
        stat, pval = sp_stats.ks_2samp(col_a, col_b)
        ks_stat[f] = stat
        ks_pval[f] = pval

    # --- Spectral L1 ---
    psd_a = ensemble_psd(data_a, dt)
    psd_b = ensemble_psd(data_b, dt)

    K = min(psd_a.psd_mean.shape[0], psd_b.psd_mean.shape[0])
    df = psd_b.freqs[1] - psd_b.freqs[0] if len(psd_b.freqs) > 1 else 1.0
    l1_num = np.trapezoid(
        np.abs(psd_a.psd_mean[:K] - psd_b.psd_mean[:K]), dx=df, axis=0
    )
    l1_den = np.trapezoid(psd_b.psd_mean[:K], dx=df, axis=0)
    l1_den = np.where(l1_den < 1e-12, 1.0, l1_den)
    spectral_l1 = l1_num / l1_den

    # --- Wasserstein-1 on time-averaged marginals ---
    wass = np.empty(F)
    for f in range(F):
        # Flatten time and ensemble → marginal samples
        flat_a = data_a[:, :, f].ravel()
        flat_b = data_b[:, :, f].ravel()
        wass[f] = sp_stats.wasserstein_distance(flat_a, flat_b)

    # --- MMD ---
    mmd_vals = np.empty(F)
    for f in range(F):
        flat_a = data_a[:, :, f].ravel()
        flat_b = data_b[:, :, f].ravel()
        # Subsample for computational tractability if ensembles are large
        max_samples = 5000
        if len(flat_a) > max_samples:
            rng = np.random.default_rng(42)
            flat_a = rng.choice(flat_a, max_samples, replace=False)
            flat_b = rng.choice(flat_b, max_samples, replace=False)
        mmd_vals[f] = _rbf_mmd(flat_a, flat_b)

    # --- Cross-correlation structure ---
    xcorr_a = ensemble_cross_correlation(data_a)
    xcorr_b = ensemble_cross_correlation(data_b)
    xcorr_err = cross_correlation_error(xcorr_a.corr_mean, xcorr_b.corr_mean)

    return ComparisonResult(
        feature_names=feature_names,
        moment_rel_error=moment_rel,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        spectral_l1=spectral_l1,
        wasserstein=wass,
        mmd=mmd_vals,
        cross_corr_error=xcorr_err,
    )

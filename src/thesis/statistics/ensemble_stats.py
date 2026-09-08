"""Core ensemble statistics: moments, maxima/extremes, and spectral distributions.

All functions operate on plain numpy arrays of shape (N, T, F).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


import numpy as np
from scipy import signal, stats

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 1. Statistical moments
# ---------------------------------------------------------------------------


@dataclass
class MomentResult:
    """Results from ensemble moment computation.

    Attributes:
        mean_t: Ensemble mean at each timestep, shape ``(T, F)``.
        var_t: Ensemble variance at each timestep, shape ``(T, F)``.
        skew_t: Ensemble skewness at each timestep, shape ``(T, F)``.
        kurt_t: Ensemble excess kurtosis at each timestep, shape ``(T, F)``.
        mean_scalar: Time-averaged ensemble mean (stationary summary), shape
            ``(F,)``.
        var_scalar: Time-averaged ensemble variance, shape ``(F,)``.
        skew_scalar: Time-averaged ensemble skewness, shape ``(F,)``.
        kurt_scalar: Time-averaged ensemble kurtosis, shape ``(F,)``.
    """

    mean_t: np.ndarray
    var_t: np.ndarray
    skew_t: np.ndarray
    kurt_t: np.ndarray
    mean_scalar: np.ndarray
    var_scalar: np.ndarray
    skew_scalar: np.ndarray
    kurt_scalar: np.ndarray


def ensemble_moments(
    data: np.ndarray,
) -> MomentResult:
    """Compute time-resolved and scalar ensemble moments.

    Args:
        data: Ensemble of time-series, shape ``(N, T, F)``.

    Returns:
        The computed moments.
    """
    data = np.asarray(data, dtype=np.float64)

    # Replace non-finite values to prevent overflow in scipy stats
    if not np.all(np.isfinite(data)):
        data = np.where(np.isfinite(data), data, 0.0)

    # Time-resolved: statistics across ensemble axis (axis=0)
    mean_t = np.mean(data, axis=0)  # (T, F)
    var_t = np.var(data, axis=0, ddof=1)  # (T, F)
    skew_t = stats.skew(data, axis=0)  # (T, F)
    kurt_t = stats.kurtosis(data, axis=0)  # (T, F)  excess kurtosis

    # Stationary scalar: average the time-resolved profiles over time
    mean_scalar = np.abs(np.mean(mean_t, axis=0))  # (F,)
    var_scalar = np.abs(np.mean(var_t, axis=0))
    skew_scalar = np.abs(np.mean(skew_t, axis=0))
    kurt_scalar = np.abs(np.mean(kurt_t, axis=0))

    return MomentResult(
        mean_t=mean_t,
        var_t=var_t,
        skew_t=skew_t,
        kurt_t=kurt_t,
        mean_scalar=mean_scalar,
        var_scalar=var_scalar,
        skew_scalar=skew_scalar,
        kurt_scalar=kurt_scalar,
    )


# ---------------------------------------------------------------------------
# 2. Extreme value statistics — use pot_extreme_values from extreme_values.py
# ---------------------------------------------------------------------------


def footprint_radius(data: np.ndarray, ix_eta_x: int, ix_eta_y: int) -> np.ndarray:
    """Per-sample maximum excursion radius sqrt(eta_x^2 + eta_y^2).

    Args:
        data: Ensemble of time-series, shape ``(N, T, F)``.
        ix_eta_x: Feature index for eta_x.
        ix_eta_y: Feature index for eta_y.

    Returns:
        Max radii, shape ``(N,)``.
    """
    r = np.sqrt(data[:, :, ix_eta_x] ** 2 + data[:, :, ix_eta_y] ** 2)  # (N, T)
    return np.max(r, axis=1)  # (N,)


# ---------------------------------------------------------------------------
# 3. Cross-correlation between features
# ---------------------------------------------------------------------------


@dataclass
class CrossCorrelationResult:
    """Pearson correlation structure between features, averaged over ensemble.

    ``corr_mean[i, j]`` is the average Pearson correlation between feature *i*
    and feature *j*, computed per realisation then averaged across the ensemble.
    A perfect model reproduces the full (F, F) matrix of the reference data.

    Use :func:`cross_correlation_error` to get a single scalar distance
    (Frobenius norm of the difference) between two correlation matrices.

    Attributes:
        corr_mean: Ensemble-mean correlation matrix, shape ``(F, F)``.
        corr_std: Ensemble std of correlation coefficients, shape ``(F, F)``.
    """

    corr_mean: np.ndarray
    corr_std: np.ndarray


def ensemble_cross_correlation(
    data: np.ndarray,
) -> CrossCorrelationResult:
    """Compute the ensemble-averaged Pearson cross-correlation matrix.

    For each realisation the (F, F) correlation matrix is computed over
    the time axis, then the matrices are averaged across realisations.

    Args:
        data: Ensemble of time-series, shape ``(N, T, F)``. N = realisations,
            T = time steps, F = features.

    Returns:
        The ensemble-averaged correlation structure.
    """
    data = np.asarray(data, dtype=np.float64)
    N, T, F = data.shape

    # Stack per-realisation correlation matrices: (N, F, F)
    corrs = np.empty((N, F, F), dtype=np.float64)
    for n in range(N):
        corrs[n] = np.corrcoef(data[n].T)  # (F, F)

    return CrossCorrelationResult(
        corr_mean=np.mean(corrs, axis=0),
        corr_std=np.std(corrs, axis=0, ddof=1),
    )


def cross_correlation_error(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    """Frobenius-norm distance between two (F, F) correlation matrices.

    This gives a single scalar summarising how different the cross-feature
    coupling is between two ensembles.  A value of 0 means identical
    correlation structure.

    The Frobenius norm is normalised by the number of unique off-diagonal
    pairs so the result is interpretable as a *mean absolute entry error*.

    Args:
        corr_a: First ``(F, F)`` correlation matrix.
        corr_b: Second ``(F, F)`` correlation matrix.

    Returns:
        Normalised Frobenius distance.
    """
    diff = corr_a - corr_b
    F = diff.shape[0]
    # Use upper triangle (excluding diagonal) to avoid double-counting
    # and ignore the trivial corr(i,i) = 1 entries.
    triu_idx = np.triu_indices(F, k=1)
    return float(np.mean(np.abs(diff[triu_idx])))


# ---------------------------------------------------------------------------
# 4. Spectral distributions (PSD)
# ---------------------------------------------------------------------------


@dataclass
class PSDResult:
    """Results from power spectral density analysis.

    Attributes:
        freqs: Frequency vector in Hz, shape ``(K,)``.
        psd_mean: Ensemble-averaged PSD, shape ``(K, F)``.
        psd_std: Ensemble std of PSD, shape ``(K, F)``.
        m0: Zeroth spectral moment (mean square), shape ``(F,)``.
        m2: Second spectral moment, shape ``(F,)``.
        m4: Fourth spectral moment, shape ``(F,)``.
        significant: Significant value = 4*sqrt(m0), shape ``(F,)``.
        Tz: Mean zero-crossing period = sqrt(m0/m2), shape ``(F,)``.
        bandwidth: Spectral bandwidth ε = sqrt(1 - m2^2/(m0*m4)), shape
            ``(F,)``.
        f_peak: Peak frequency per feature, shape ``(F,)``.
    """

    freqs: np.ndarray
    psd_mean: np.ndarray
    psd_std: np.ndarray
    m0: np.ndarray
    m2: np.ndarray
    m4: np.ndarray
    significant: np.ndarray
    Tz: np.ndarray
    bandwidth: np.ndarray
    f_peak: np.ndarray


def ensemble_psd(
    data: np.ndarray,
    dt: float,
    n_per_seg: int | None = None,
    n_overlap: int | None = None,
) -> PSDResult:
    """Compute ensemble-averaged PSD via Welch's method.

    Args:
        data: Ensemble of time-series, shape ``(N, T, F)``.
        dt: Sampling interval in seconds.
        n_per_seg: Segment length for Welch.  Default: ``T // 8``.
        n_overlap: Overlap.  Default: ``n_per_seg // 2``.

    Returns:
        The computed spectral distribution.
    """
    data = np.asarray(data, dtype=np.float64)
    N, T, F = data.shape
    fs = 1.0 / dt
    if n_per_seg is None:
        n_per_seg = min(T, max(256, T // 8))
    if n_overlap is None:
        n_overlap = n_per_seg // 2

    # Compute PSD for every (sample, feature)
    freqs, psd_0 = signal.welch(
        data[0, :, 0], fs=fs, nperseg=n_per_seg, noverlap=n_overlap
    )
    K = len(freqs)
    all_psd = np.empty((N, K, F), dtype=np.float64)

    for n in range(N):
        for f in range(F):
            _, pxx = signal.welch(
                data[n, :, f], fs=fs, nperseg=n_per_seg, noverlap=n_overlap
            )
            all_psd[n, :, f] = pxx

    psd_mean = np.mean(all_psd, axis=0)  # (K, F)
    psd_std = np.std(all_psd, axis=0, ddof=1)

    # Spectral moments via trapezoidal integration
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    m0 = np.trapezoid(psd_mean, dx=df, axis=0)  # (F,)
    m2 = np.trapezoid(psd_mean * freqs[:, None] ** 2, dx=df, axis=0)
    m4 = np.trapezoid(psd_mean * freqs[:, None] ** 4, dx=df, axis=0)

    significant = 4.0 * np.sqrt(np.maximum(m0, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        Tz = np.where(m2 > 0, np.sqrt(m0 / m2), np.inf)
        bandwidth = np.where(
            (m0 > 0) & (m4 > 0),
            np.sqrt(np.clip(1.0 - m2**2 / (m0 * m4), 0, 1)),
            0.0,
        )

    # Peak frequency per feature
    f_peak = freqs[np.argmax(psd_mean, axis=0)]  # (F,)

    return PSDResult(
        freqs=freqs,
        psd_mean=psd_mean,
        psd_std=psd_std,
        m0=m0,
        m2=m2,
        m4=m4,
        significant=significant,
        Tz=Tz,
        bandwidth=bandwidth,
        f_peak=f_peak,
    )

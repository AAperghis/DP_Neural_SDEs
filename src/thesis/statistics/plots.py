"""Unified ensemble-statistics plotting functions.

Single source of truth for moments, PSD, and maxima distribution plots.
Works on the cluster (no dependency on ``visualisation``).

All functions accept a list of result objects (one per dataset) so they
work for both single-dataset inspection and multi-dataset comparison.
Pass ``feature_indices`` to select a subset of features for plotting.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from thesis.shared.data_structures import FEATURE_REGISTRY
from thesis.statistics.ensemble_stats import (
    MomentResult,
    PSDResult,
    CrossCorrelationResult,
)
from thesis.statistics.extreme_values import EVResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_features(
    n_features: int,
    feature_names: list[str] | None,
    feature_indices: Sequence[int] | None,
) -> tuple[list[int], list[str]]:
    """Return (indices, display_names) honouring user selection."""
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    if feature_indices is None:
        feature_indices = list(range(n_features))

    display_names = [feature_names[i] for i in feature_indices]
    return list(feature_indices), display_names


def _feature_label(name: str) -> str:
    info = FEATURE_REGISTRY.get(name)
    return info.label if info else name


def _feature_unit(name: str) -> str:
    info = FEATURE_REGISTRY.get(name)
    return info.unit if info else ""


# ---------------------------------------------------------------------------
# 1. Moments bar chart
# ---------------------------------------------------------------------------


def plot_moments(
    mom_results: list[MomentResult],
    feature_names: list[str] | None = None,
    labels: list[str] | None = None,
    feature_indices: Sequence[int] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Grouped bar chart comparing scalar moments across datasets.

    Args:
        mom_results: One entry per dataset.
        feature_names: Full feature name list (length F). Used for display
            labels and unit look-up via ``FEATURE_REGISTRY``.
        labels: Legend label per dataset.  Defaults to "Dataset 0", …
        feature_indices: Indices of features to plot.  Defaults to all.

    Returns:
        (fig, axes)
    """
    F_total = len(mom_results[0].mean_scalar)
    idx, names = _resolve_features(F_total, feature_names, feature_indices)
    n_ds = len(mom_results)
    if labels is None:
        labels = [f"Dataset {i}" for i in range(n_ds)]

    display = [_feature_label(n) for n in names]

    moment_keys = [
        ("Mean", lambda m: m.mean_scalar),
        ("Variance", lambda m: m.var_scalar),
        ("Skewness", lambda m: m.skew_scalar),
        ("Excess kurtosis", lambda m: m.kurt_scalar),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.ravel()
    bar_width = 0.8 / n_ds
    x = np.arange(len(idx))

    for ax, (title, accessor) in zip(axes_flat, moment_keys):
        all_vals: list[np.ndarray] = []
        for j, mom in enumerate(mom_results):
            vals = accessor(mom)[idx]
            all_vals.append(vals)
            offset = (j - (n_ds - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width, label=labels[j], color=f"C{j}")

        # Safeguard: adaptive symlog threshold so small bars stay visible
        all_abs = np.abs(np.concatenate(all_vals))
        nonzero = all_abs[all_abs > 0]
        linthresh = float(np.percentile(nonzero, 5)) if len(nonzero) > 0 else 1e-3
        linthresh = max(linthresh, 1e-6)

        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.legend(fontsize=6)

    fig.tight_layout()
    return fig, axes_flat


# ---------------------------------------------------------------------------
# 2. PSD (log-log spectra)
# ---------------------------------------------------------------------------


def plot_psd(
    psd_results: list[PSDResult],
    feature_names: list[str] | None = None,
    labels: list[str] | None = None,
    feature_indices: Sequence[int] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Log-log PSD comparison per feature across datasets.

    Args:
        psd_results: One entry per dataset.
        feature_names, labels, feature_indices: See :func:`plot_moments`.

    Returns:
        (fig, axes)
    """
    F_total = psd_results[0].psd_mean.shape[1]
    idx, names = _resolve_features(F_total, feature_names, feature_indices)
    n_ds = len(psd_results)
    n_feat = len(idx)
    if labels is None:
        labels = [f"Dataset {i}" for i in range(n_ds)]

    ncols = min(4, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for plot_i, (fi, name) in enumerate(zip(idx, names)):
        ax = axes[plot_i]
        for j, psd in enumerate(psd_results):
            color = f"C{j}"
            f_mask = psd.freqs > 0
            mean_psd = psd.psd_mean[f_mask, fi]
            std_psd = psd.psd_std[f_mask, fi]
            f = psd.freqs[f_mask]

            ax.loglog(f, mean_psd, color=color, lw=0.8, label=labels[j])
            ax.fill_between(
                f,
                np.maximum(mean_psd - std_psd, 1e-20),
                mean_psd + std_psd,
                alpha=0.25,
                color=color,
            )
            ax.axvline(psd.f_peak[fi], color=color, ls=":", lw=0.7, alpha=0.7)

        # Safeguard: cap y-range to at most 8 decades
        ymin, ymax = ax.get_ylim()
        if ymax / max(ymin, 1e-30) > 1e8:
            ax.set_ylim(ymax / 1e8, ymax)

        label_text = _feature_label(name)
        unit = _feature_unit(name)
        ax.set_title(label_text, fontsize=8)
        ax.set_xlabel("Frequency [Hz]")
        if plot_i % ncols == 0:
            ax.set_ylabel(f"PSD [{unit}$^2$/Hz]" if unit else "PSD")
        ax.legend(fontsize=6)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Power Spectral Density (Welch)", y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Maxima distributions (histograms + POT fit)
# ---------------------------------------------------------------------------


def plot_maxima(
    max_results: list[EVResult],
    feature_names: list[str] | None = None,
    labels: list[str] | None = None,
    feature_indices: Sequence[int] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Overlaid histograms of per-sample maxima with POT compound-max PDF overlay.

    Args:
        max_results: One entry per dataset.
        feature_names, labels, feature_indices: See :func:`plot_moments`.

    Returns:
        (fig, axes)
    """
    F_total = len(max_results[0])
    idx, names = _resolve_features(F_total, feature_names, feature_indices)
    n_ds = len(max_results)
    n_feat = len(idx)
    if labels is None:
        labels = [f"Dataset {i}" for i in range(n_ds)]

    ncols = min(4, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for plot_i, (fi, name) in enumerate(zip(idx, names)):
        ax = axes[plot_i]
        unit = _feature_unit(name)

        # Safeguard: robust x-limits from all datasets
        all_samples = np.concatenate([mx.observed_maxima[:, fi] for mx in max_results])
        lo = float(np.percentile(all_samples, 0.5))
        hi = float(np.percentile(all_samples, 99.5))
        margin = (hi - lo) * 0.15
        if margin < 1e-12:
            margin = max(abs(hi) * 0.1, 1.0)
        xlim = (lo - margin, hi + margin)

        any_fallback = False

        for j, mx in enumerate(max_results):
            color = f"C{j}"
            samples = mx.observed_maxima[:, fi]
            pot = mx[fi]
            is_fallback = pot.peak_rate == 0.0

            if is_fallback:
                any_fallback = True

            ax.hist(
                samples,
                bins=min(max(int(np.sqrt(len(samples))), 10), 50),
                density=True,
                alpha=0.35,
                color=color,
                edgecolor="white",
                lw=0.5,
                label=labels[j] + (" *" if is_fallback else ""),
                range=xlim,
            )

            # POT compound-max PDF overlay (skip for fallback fits)
            if not is_fallback:
                x_fit = np.linspace(xlim[0], xlim[1], 200)
                pdf = pot.pdf(x_fit)
                valid = np.isfinite(pdf) & (pdf > 0)
                if np.any(valid):
                    ax.plot(x_fit[valid], pdf[valid], color=color, lw=1.2, ls="--")

            # MPM line (only if within visible range; skip for fallback)
            if not is_fallback and xlim[0] <= mx.mpm[fi] <= xlim[1]:
                ax.axvline(mx.mpm[fi], color=color, ls=":", lw=0.9)

        ax.set_xlim(xlim)
        title = _feature_label(name)
        if any_fallback:
            title += "  (* no fit)"
            ax.patch.set_facecolor("#fff3cd")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(f"[{unit}]" if unit else "")
        if plot_i == 0:
            ax.legend(fontsize=6)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("3-Hour Maxima Distributions", y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 4. Cross-correlation heatmaps
# ---------------------------------------------------------------------------


def plot_cross_correlation(
    xcorr_results: list[CrossCorrelationResult],
    feature_names: list[str] | None = None,
    labels: list[str] | None = None,
    show_difference: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side heatmaps of the cross-correlation matrices.

    When two datasets are given and *show_difference* is True, a third
    panel shows the element-wise difference (dataset 0 − dataset 1) so
    you can immediately see which feature pairs the model gets wrong.

    Args:
        xcorr_results: One entry per dataset (typically [model, reference]).
        feature_names: Display labels for the axes.  Defaults to f0, f1, …
        labels: Legend / title label per dataset.
        show_difference: If True and exactly two datasets are given, add a
            difference panel.

    Returns:
        (fig, axes)
    """
    n_ds = len(xcorr_results)
    F = xcorr_results[0].corr_mean.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]
    if labels is None:
        labels = [f"Dataset {i}" for i in range(n_ds)]

    display = [_feature_label(n) for n in feature_names]

    # Decide number of panels
    n_panels = n_ds + (1 if show_difference and n_ds == 2 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), squeeze=False)
    axes = axes.ravel()

    # Plot each dataset's correlation matrix
    for i, (xcorr, label) in enumerate(zip(xcorr_results, labels)):
        ax = axes[i]
        im = ax.imshow(
            xcorr.corr_mean,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            aspect="equal",
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(F))
        ax.set_yticks(range(F))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(display, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Difference panel (dataset 0 minus dataset 1)
    if show_difference and n_ds == 2:
        ax = axes[n_ds]
        diff = xcorr_results[0].corr_mean - xcorr_results[1].corr_mean
        vmax = max(0.1, float(np.abs(diff).max()))  # ensure visible scale
        im = ax.imshow(
            diff,
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
            aspect="equal",
        )
        ax.set_title(f"Difference ({labels[0]} − {labels[1]})", fontsize=10)
        ax.set_xticks(range(F))
        ax.set_yticks(range(F))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(display, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-Correlation Between Features", y=1.02)
    fig.tight_layout()
    return fig, axes

"""Online statistics checks for evaluating generative models during training.

Generates an ensemble of model samples, computes distributional statistics,
and compares them against a reference dataset ensemble. All metrics are
returned as a flat dict suitable for MLflow logging.

Metrics reported (per feature group and as aggregates):
- Moment relative errors (mean, variance, skewness, kurtosis)
- Extreme value: MPM relative error, Gumbel parameter error
- Spectral: zero-crossing period Tz error, peak frequency error, spectral L1
- Distributional: Wasserstein-1 distance, KS statistic, MMD
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from thesis.shared.data_structures import FEATURE_REGISTRY
from thesis.statistics.ensemble_stats import (
    MomentResult,
    PSDResult,
    ensemble_moments,
    ensemble_psd,
    ensemble_cross_correlation,
    cross_correlation_error,
)
from thesis.statistics.extreme_values import EVResult, pot_extreme_values


@dataclass
class TrainingStatsResult:
    """Container for training-time statistics comparison.

    Attributes
    ----------
    metrics : dict[str, float]
        Flat metric dict ready for MLflow logging.
    """

    metrics: dict[str, float]
    mom: dict[str, MomentResult]
    psd: dict[str, PSDResult]
    max: dict[str, EVResult]


# Feature group indices for the standard 12-feature layout.
_GROUPS = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "rpm": slice(6, 12),
}


def _safe_rel_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.abs(b)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return np.abs(a - b) / denom


def _group_mean(vals: np.ndarray, group_slice: slice) -> float:
    return float(np.mean(vals[group_slice]))


def compute_training_stats(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    prefix: str = "stats",
    ref_cache: StatsRefCache | None = None,
) -> TrainingStatsResult:
    """Compare model samples against reference data across all metric families.

    Parameters
    ----------
    model_data : (N_model, T, F)
        Ensemble of model-generated trajectories (physical units).
    ref_data : (N_ref, T, F)
        Ensemble of reference (dataset) trajectories (physical units).
    dt : float
        Sampling interval in seconds.
    feature_names : list[str], optional
        Feature names for reporting. Defaults to f0, f1, ...
    prefix : str
        Prefix for all metric keys (e.g. "prior_stats" or "post_stats").
    ref_cache : StatsRefCache, optional
        Precomputed reference statistics. If provided, ``ref_data`` is only
        used for per-feature distributional tests (Wasserstein, KS).

    Returns
    -------
    TrainingStatsResult
    """
    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    metrics: dict[str, float] = {}

    # --- 1. Moment comparison ---
    mom_model = ensemble_moments(model_data)
    mom_ref = ref_cache.moments if ref_cache is not None else ensemble_moments(ref_data)
    moments = dict(model=mom_model, ref=mom_ref)

    for name, model_val, ref_val in [
        ("mean", moments["model"].mean_scalar, moments["ref"].mean_scalar),
        ("var", moments["model"].var_scalar, moments["ref"].var_scalar),
        ("skew", moments["model"].skew_scalar, moments["ref"].skew_scalar),
        ("kurt", moments["model"].kurt_scalar, moments["ref"].kurt_scalar),
    ]:
        rel_err = _safe_rel_error(model_val, ref_val)
        metrics[f"{prefix}/moment_{name}_rel_err"] = float(np.mean(rel_err))
        for gname, gslice in _GROUPS.items():
            if gslice.stop <= F:
                metrics[f"{prefix}/{gname}/moment_{name}_rel_err"] = _group_mean(
                    rel_err, gslice
                )

    # --- 2. Extreme value comparison ---
    max_model = pot_extreme_values(model_data, dt)
    max_ref = (
        ref_cache.maxima if ref_cache is not None else pot_extreme_values(ref_data, dt)
    )
    maxima = dict(model=max_model, ref=max_ref)

    mpm_rel = _safe_rel_error(maxima["model"].mpm, maxima["ref"].mpm)
    metrics[f"{prefix}/mpm_rel_err"] = float(np.nanmean(mpm_rel))

    max_mean_rel = _safe_rel_error(maxima["model"].max_mean, maxima["ref"].max_mean)
    metrics[f"{prefix}/max_mean_rel_err"] = float(np.nanmean(max_mean_rel))

    for gname, gslice in _GROUPS.items():
        if gslice.stop <= F:
            metrics[f"{prefix}/{gname}/mpm_rel_err"] = float(
                np.nanmean(mpm_rel[gslice])
            )
            metrics[f"{prefix}/{gname}/max_mean_rel_err"] = float(
                np.nanmean(max_mean_rel[gslice])
            )

    # --- 3. Spectral / zero-crossing comparison ---
    psd_model = ensemble_psd(model_data, dt)
    psd_ref = ref_cache.psd if ref_cache is not None else ensemble_psd(ref_data, dt)
    psd = dict(model=psd_model, ref=psd_ref)

    tz_rel = _safe_rel_error(psd["model"].Tz, psd["ref"].Tz)
    # Mask inf values (features with no zero crossings)
    tz_finite = np.isfinite(tz_rel)
    metrics[f"{prefix}/Tz_rel_err"] = float(
        np.mean(tz_rel[tz_finite]) if np.any(tz_finite) else np.nan
    )

    fp_rel = _safe_rel_error(psd["model"].f_peak, psd["ref"].f_peak)
    fp_finite = np.isfinite(fp_rel)
    metrics[f"{prefix}/f_peak_rel_err"] = float(
        np.mean(fp_rel[fp_finite]) if np.any(fp_finite) else np.nan
    )

    sig_rel = _safe_rel_error(psd["model"].significant, psd["ref"].significant)
    metrics[f"{prefix}/significant_rel_err"] = float(np.mean(sig_rel))

    bw_err = np.abs(psd["model"].bandwidth - psd["ref"].bandwidth)
    metrics[f"{prefix}/bandwidth_abs_err"] = float(np.mean(bw_err))

    # Spectral L1
    K = min(psd["model"].psd_mean.shape[0], psd["ref"].psd_mean.shape[0])
    df = psd["ref"].freqs[1] - psd["ref"].freqs[0] if len(psd["ref"].freqs) > 1 else 1.0
    l1_num = np.trapezoid(
        np.abs(psd["model"].psd_mean[:K] - psd["ref"].psd_mean[:K]), dx=df, axis=0
    )
    l1_den = np.trapezoid(psd["ref"].psd_mean[:K], dx=df, axis=0)
    l1_den = np.where(l1_den < 1e-12, 1.0, l1_den)
    spectral_l1 = l1_num / l1_den
    metrics[f"{prefix}/spectral_l1"] = float(np.mean(spectral_l1))

    for gname, gslice in _GROUPS.items():
        if gslice.stop <= F:
            tz_g = tz_rel[gslice]
            tz_g_fin = np.isfinite(tz_g)
            metrics[f"{prefix}/{gname}/Tz_rel_err"] = float(
                np.mean(tz_g[tz_g_fin]) if np.any(tz_g_fin) else np.nan
            )
            metrics[f"{prefix}/{gname}/spectral_l1"] = _group_mean(spectral_l1, gslice)
            metrics[f"{prefix}/{gname}/significant_rel_err"] = _group_mean(
                sig_rel, gslice
            )

    # --- 4. Distributional metrics (per-feature, then aggregated) ---
    from scipy import stats as sp_stats

    # KS test on maxima distributions (skip features with NaN)
    ks_stats = np.full(F, np.nan)
    ks_pvals = np.full(F, np.nan)
    for f in range(F):
        m_max = maxima["model"].observed_maxima[:, f]
        r_max = maxima["ref"].observed_maxima[:, f]
        if np.any(np.isnan(m_max)) or np.any(np.isnan(r_max)):
            continue
        stat, pval = sp_stats.ks_2samp(m_max, r_max)
        ks_stats[f] = stat
        ks_pvals[f] = pval

    metrics[f"{prefix}/ks_stat_maxima"] = float(np.nanmean(ks_stats))
    metrics[f"{prefix}/ks_pvalue_maxima"] = float(np.nanmean(ks_pvals))

    # Wasserstein-1 on time-averaged marginals
    wass = np.empty(F)
    for f in range(F):
        flat_m = model_data[:, :, f].ravel()
        flat_r = ref_data[:, :, f].ravel()
        wass[f] = sp_stats.wasserstein_distance(flat_m, flat_r)

    metrics[f"{prefix}/wasserstein"] = float(np.mean(wass))

    for gname, gslice in _GROUPS.items():
        if gslice.stop <= F:
            metrics[f"{prefix}/{gname}/ks_stat_maxima"] = _group_mean(ks_stats, gslice)
            metrics[f"{prefix}/{gname}/wasserstein"] = _group_mean(wass, gslice)

    # --- 5. Log absolute model stats for monitoring ---
    metrics[f"{prefix}/model_mean_abs"] = float(
        np.mean(np.abs(moments["model"].mean_scalar))
    )
    metrics[f"{prefix}/model_var"] = float(np.mean(moments["model"].var_scalar))
    metrics[f"{prefix}/model_Tz_mean"] = float(
        np.mean(psd["model"].Tz[np.isfinite(psd["model"].Tz)])
        if np.any(np.isfinite(psd["model"].Tz))
        else np.nan
    )
    metrics[f"{prefix}/model_mpm_mean"] = float(np.nanmean(np.abs(maxima["model"].mpm)))

    # --- 6. Cross-correlation structure ---
    xcorr_model = ensemble_cross_correlation(model_data)
    xcorr_ref_mean = (
        ref_cache.xcorr_mean
        if ref_cache is not None
        else ensemble_cross_correlation(ref_data).corr_mean
    )
    metrics[f"{prefix}/cross_corr_err"] = cross_correlation_error(
        xcorr_model.corr_mean, xcorr_ref_mean
    )

    # Filter out NaN values (MLflow doesn't handle them well)
    metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}

    return TrainingStatsResult(metrics=metrics, psd=psd, mom=moments, max=maxima)


# ---------------------------------------------------------------------------
# Compact sweep ranking scores (η-focused)
# ---------------------------------------------------------------------------


@dataclass
class SweepRefCache:
    """Precomputed reference-side quantities for ``sweep_score``.

    Create once via ``precompute_sweep_ref`` and pass to every
    ``sweep_score`` call to avoid redundant work.
    """

    r_eta_2d: np.ndarray  # (M, T*3) flattened η
    dYY_mean: float  # mean self-distance for energy
    sigs_y: np.ndarray  # (M, D) truncated signatures
    corr_r: np.ndarray  # (n_eta,) Pearson(σ_η, Hs) per η channel


def precompute_sweep_ref(
    ref_data: np.ndarray,
    ref_wave: np.ndarray | None = None,
    eta_idx: slice = slice(0, 3),
    sig_depth: int = 3,
) -> SweepRefCache:
    """Precompute all reference-side quantities for ``sweep_score``."""
    from scipy.spatial.distance import cdist

    r_eta = ref_data[:, :, eta_idx]
    r_eta_2d = r_eta.reshape(r_eta.shape[0], -1).astype(np.float64)
    dYY = cdist(r_eta_2d, r_eta_2d, metric="euclidean")
    dYY_mean = float(dYY.mean())

    sigs_y = np.stack(
        [_truncated_signature(r_eta[i], sig_depth) for i in range(r_eta.shape[0])]
    )

    corr_r = np.zeros(r_eta.shape[2])
    if ref_wave is not None:
        hs_r = ref_wave[:, 0]
        r_sigma = np.std(r_eta, axis=1)
        for c in range(r_eta.shape[2]):
            s, h = np.std(r_sigma[:, c]), np.std(hs_r)
            if s < 1e-12 or h < 1e-12:
                corr_r[c] = 0.0
            else:
                corr_r[c] = np.corrcoef(r_sigma[:, c], hs_r)[0, 1]

    return SweepRefCache(
        r_eta_2d=r_eta_2d,
        dYY_mean=dYY_mean,
        sigs_y=sigs_y,
        corr_r=corr_r,
    )


@dataclass
class StatsRefCache:
    """Precomputed reference-side quantities for ``compute_training_stats``.

    Create once via ``precompute_stats_ref``.
    """

    moments: MomentResult
    maxima: EVResult
    psd: PSDResult
    xcorr_mean: np.ndarray  # (F, F) cross-correlation matrix
    ref_data: np.ndarray  # raw data kept for per-feature Wasserstein / KS


def precompute_stats_ref(
    ref_data: np.ndarray,
    dt: float,
) -> StatsRefCache:
    """Precompute all reference-side quantities for ``compute_training_stats``."""
    return StatsRefCache(
        moments=ensemble_moments(ref_data),
        maxima=pot_extreme_values(ref_data, dt),
        psd=ensemble_psd(ref_data, dt),
        xcorr_mean=ensemble_cross_correlation(ref_data).corr_mean,
        ref_data=ref_data,
    )


def _truncated_signature(path: np.ndarray, depth: int = 3) -> np.ndarray:
    """Compute the truncated signature of a path via iterated integrals.

    Parameters
    ----------
    path : (T, d)
        Single multivariate time series.
    depth : int
        Truncation depth (1–4 recommended).

    Returns
    -------
    sig : (D,)  where D = d + d² + … + d^depth
    """
    T, d = path.shape
    increments = np.diff(path, axis=0)  # (T-1, d)

    # Level 1: ∫ dX  →  shape (d,)
    levels = [increments.sum(axis=0)]

    # Higher levels via iterated integrals
    prev = np.cumsum(increments, axis=0)  # running integral, (T-1, d)
    for k in range(2, depth + 1):
        # prev has shape (T-1, d^(k-1))
        # new level: ∫ prev_{t-} ⊗ dX_t  →  shape (d^(k-1) * d,)
        # = Σ_t prev[t-1] ⊗ dX[t]   (left-point Riemann)
        prev_shifted = np.vstack([np.zeros((1, prev.shape[1])), prev[:-1]])
        # outer product at each time step then sum
        outer = prev_shifted[:, :, None] * increments[:, None, :]  # (T-1, d^(k-1), d)
        level_k = outer.sum(axis=0).ravel()
        levels.append(level_k)
        # Update prev for next depth: cumulative sum of outer products
        prev = np.cumsum(outer.reshape(T - 1, -1), axis=0)

    return np.concatenate(levels)


def sweep_score(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    dt: float,
    eta_idx: slice = slice(0, 3),
    model_wave: np.ndarray | None = None,
    ref_wave: np.ndarray | None = None,
    sig_depth: int = 3,
    ref_cache: SweepRefCache | None = None,
) -> dict[str, float]:
    """Compact sweep comparison scores for η channels.

    Returns a dict with:
      - ``energy``     : Energy distance on η trajectories (lower = better).
      - ``sig_mmd``    : Signature MMD on η trajectories (lower = better).
      - ``conditioning``: |Δ corr(σ_η, Hs)| model vs ref (lower = better,
                          only if wave arrays supplied).

    Parameters
    ----------
    model_data, ref_data : (N, T, F)
        Ensembles in physical units.
    dt : float
        Sampling interval (seconds), unused here but kept for API compat.
    eta_idx : slice
        Which channels are η (default 0:3).
    model_wave, ref_wave : (N, 3) optional
        Sea-state [Hs, Tp, Dir] per sample for conditioning score.
    sig_depth : int
        Truncation depth for signatures (default 3).
    ref_cache : SweepRefCache, optional
        Precomputed reference quantities from ``precompute_sweep_ref``.
    """
    from scipy.spatial.distance import cdist

    m_eta = model_data[:, :, eta_idx]  # (N, T, 3)

    scores: dict[str, float] = {}

    # --- 1. Energy distance on full η trajectories ---
    X2d = m_eta.reshape(m_eta.shape[0], -1).astype(np.float64)
    dXX = cdist(X2d, X2d, metric="euclidean")

    if ref_cache is not None:
        dXY = cdist(X2d, ref_cache.r_eta_2d, metric="euclidean")
        scores["energy"] = float(2.0 * dXY.mean() - dXX.mean() - ref_cache.dYY_mean)
    else:
        r_eta = ref_data[:, :, eta_idx]
        Y2d = r_eta.reshape(r_eta.shape[0], -1).astype(np.float64)
        dXY = cdist(X2d, Y2d, metric="euclidean")
        dYY = cdist(Y2d, Y2d, metric="euclidean")
        scores["energy"] = float(2.0 * dXY.mean() - dXX.mean() - dYY.mean())

    # --- 2. Signature MMD ---
    sigs_x = np.stack(
        [_truncated_signature(m_eta[i], sig_depth) for i in range(m_eta.shape[0])]
    )
    sigs_y = (
        ref_cache.sigs_y
        if ref_cache is not None
        else np.stack(
            [
                _truncated_signature(ref_data[:, :, eta_idx][i], sig_depth)
                for i in range(ref_data.shape[0])
            ]
        )
    )

    dists_all = cdist(sigs_x, sigs_y, metric="sqeuclidean")
    bandwidth = float(np.median(dists_all))
    if bandwidth < 1e-12:
        bandwidth = 1.0

    def _rbf_mean(A, B, bw):
        d2 = cdist(A, B, metric="sqeuclidean")
        return float(np.mean(np.exp(-d2 / (2.0 * bw))))

    kxx = _rbf_mean(sigs_x, sigs_x, bandwidth)
    kyy = _rbf_mean(sigs_y, sigs_y, bandwidth)
    kxy = _rbf_mean(sigs_x, sigs_y, bandwidth)
    mmd2 = kxx + kyy - 2.0 * kxy
    scores["sig_mmd"] = float(np.sqrt(max(mmd2, 0.0)))

    # --- 3. Conditioning (optional) ---
    if model_wave is not None:
        hs_m = model_wave[:, 0]
        m_sigma = np.std(m_eta, axis=1)  # (N, 3)

        def _corr_sigma_hs(sigma, hs):
            corrs = np.empty(sigma.shape[1])
            for c in range(sigma.shape[1]):
                s, h = np.std(sigma[:, c]), np.std(hs)
                if s < 1e-12 or h < 1e-12:
                    corrs[c] = 0.0
                else:
                    corrs[c] = np.corrcoef(sigma[:, c], hs)[0, 1]
            return corrs

        corr_m = _corr_sigma_hs(m_sigma, hs_m)

        if ref_cache is not None:
            corr_r = ref_cache.corr_r
        elif ref_wave is not None:
            r_eta = ref_data[:, :, eta_idx]
            r_sigma = np.std(r_eta, axis=1)
            corr_r = _corr_sigma_hs(r_sigma, ref_wave[:, 0])
        else:
            corr_r = None

        if corr_r is not None:
            scores["conditioning"] = float(np.mean(np.abs(corr_m - corr_r)))

    return scores


# ---------------------------------------------------------------------------
# Derived physical quantities (lightweight, training-time)
# ---------------------------------------------------------------------------


def compute_derived_quantities(
    data: np.ndarray,
    physics_config,
) -> dict[str, np.ndarray]:
    """Compute operationally relevant derived quantities from state trajectories.

    Designed for cheap training-time evaluation — pure numpy, no fitting.

    Parameters
    ----------
    data : (N, T, F)
        Trajectories in **physical units**.
    physics_config : FullOrderPhysicsConfig
        Provides thruster geometry and index mapping.

    Returns
    -------
    dict mapping quantity name → (N, T) array.
    """
    eta_ix = physics_config.eta_idx
    n_ix = list(physics_config.n_idx)
    alpha_ix = list(physics_config.alpha_idx)
    n_tunnel = physics_config.n_tunnel
    K_diag = np.diag(np.asarray(physics_config.K_thr))  # (n_thr,)
    np.asarray(physics_config.l_x)
    np.asarray(physics_config.l_y)

    N_s, T_s = data.shape[:2]

    # 1. Excursion radius  √(x² + y²)
    excursion = np.sqrt(data[:, :, eta_ix[0]] ** 2 + data[:, :, eta_ix[1]] ** 2)

    # 2. Heading
    heading = data[:, :, eta_ix[2]]

    # 3. Total thrust force magnitude  ‖F_surge, F_sway‖
    n = data[:, :, n_ix]  # (N, T, n_thr)
    alpha = data[:, :, alpha_ix]  # (N, T, n_azi)

    tunnel_angles = np.full((N_s, T_s, n_tunnel), np.pi / 2)
    angles = np.concatenate([tunnel_angles, alpha], axis=-1)  # (N, T, n_thr)

    cos_a, sin_a = np.cos(angles), np.sin(angles)
    f_thr = K_diag * np.abs(n) * n  # (N, T, n_thr)

    F_surge = np.sum(cos_a * f_thr, axis=-1)
    F_sway = np.sum(sin_a * f_thr, axis=-1)
    thrust_force = np.sqrt(F_surge**2 + F_sway**2)

    # 4. Generator load  ∝ Σ|nᵢ|³
    gen_load = np.sum(np.abs(n) ** 3, axis=-1)

    return {
        "excursion": excursion,
        "heading": heading,
        "thrust_force": thrust_force,
        "generator_load": gen_load,
    }


def plot_cross_correlation(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    feature_names: list[str] | None = None,
    step: int = 0,
) -> "plt.Figure":
    """Side-by-side feature cross-correlation heatmaps (model vs reference).

    Shows the ensemble-averaged Pearson correlation matrix for each
    dataset and their element-wise difference.
    """
    import matplotlib

    matplotlib.use("Agg")

    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    def group_lines(ax):
        # Add lines to separate groups of features (e.g., states vs inputs)
        group_indices = [
            3,
            6,
            12,
        ]  # Indices where groups change (after each state/input pair)
        for idx in group_indices:
            ax.axhline(idx - 0.5, color="black", linestyle="--", linewidth=1)
            ax.axvline(idx - 0.5, color="black", linestyle="--", linewidth=1)

    xcorr_model = ensemble_cross_correlation(model_data)
    xcorr_ref = ensemble_cross_correlation(ref_data)
    diff = xcorr_model.corr_mean - xcorr_ref.corr_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, mat, title, lims in [
        (axes[0], xcorr_model.corr_mean, "Model", (-1, 1)),
        (axes[1], xcorr_ref.corr_mean, "Reference", (-1, 1)),
        (axes[2], diff, "Difference", (-0.5, 0.5)),
    ]:
        im = ax.imshow(mat, vmin=lims[0], vmax=lims[1], cmap="RdBu_r", aspect="equal")

        ax.set_xticks(range(F))
        ax.set_yticks(range(F))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(feature_names, fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
        group_lines(ax)
        if F <= 16:
            for i in range(F):
                for j in range(F):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="w" if abs(mat[i, j]) > 0.6 else "k",
                    )

    fig.suptitle(f"Feature Cross-Correlation \u2014 step {step}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-feature sea-state-dependent plots
# ---------------------------------------------------------------------------


def plot_mpm_vs_hs(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    model_wave: np.ndarray,
    ref_wave: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    step: int = 0,
) -> "plt.Figure":
    """MPM with bootstrap error bounds vs Hs, per feature.

    Each subplot shows one feature's MPM (computed per-sample as the observed
    max) scattered against Hs for both model and reference.
    """
    import matplotlib

    matplotlib.use("Agg")

    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    # Per-sample observed max per feature: (N, F)
    m_max = np.max(np.abs(model_data), axis=1)
    r_max = np.max(np.abs(ref_data), axis=1)

    ncols = min(4, F)
    nrows = int(np.ceil(F / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for fi in range(F):
        ax = axes[fi]
        ax.scatter(
            ref_wave[:, 0],
            r_max[:, fi],
            c="grey",
            s=15,
            alpha=0.5,
            edgecolor="none",
            label="Ref",
        )
        ax.scatter(
            model_wave[:, 0],
            m_max[:, fi],
            c="C0",
            s=25,
            edgecolor="k",
            linewidth=0.3,
            label="Model",
            zorder=3,
        )
        ax.set_xlabel("Hs [m]")
        ax.set_ylabel("Max |value|")
        ax.set_title(feature_names[fi], fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(F, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Per-Feature Max vs Hs \u2014 step {step}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_feature_wave_correlation(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    model_wave: np.ndarray,
    ref_wave: np.ndarray,
    feature_names: list[str] | None = None,
    step: int = 0,
) -> "plt.Figure":
    """Pearson correlation between per-feature std and Hs/Tp/Dir.

    Two side-by-side heatmaps (F x 3): model and reference.
    """
    import matplotlib

    matplotlib.use("Agg")

    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    m_std = np.std(model_data, axis=1)  # (N, F)
    r_std = np.std(ref_data, axis=1)

    m_corr = _pearson_summary(m_std, model_wave)  # (F, 3)
    r_corr = _pearson_summary(r_std, ref_wave)

    fig, axes = plt.subplots(1, 2, figsize=(10, max(4, 0.35 * F)))

    for ax, mat, title in [
        (axes[0], m_corr, "Model"),
        (axes[1], r_corr, "Reference"),
    ]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(_WAVE_NAMES, fontsize=9)
        ax.set_yticks(range(F))
        ax.set_yticklabels(feature_names, fontsize=7)
        for ii in range(F):
            for jj in range(3):
                ax.text(
                    jj,
                    ii,
                    f"{mat[ii, jj]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="w" if abs(mat[ii, jj]) > 0.6 else "k",
                )
        ax.set_title(f"Pearson(\u03c3, wave) \u2014 {title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Feature\u2013Wave Correlation \u2014 step {step}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_moments_vs_hs(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    model_wave: np.ndarray,
    ref_wave: np.ndarray,
    feature_names: list[str] | None = None,
    n_hs_bins: int = 4,
    step: int = 0,
) -> "plt.Figure":
    """Mean, variance, skewness and kurtosis vs Hs, per feature.

    4 rows (one per moment) x F columns. Reference shown as binned profile
    with error bars; model as individual scatter points.
    """
    import matplotlib

    matplotlib.use("Agg")

    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    # Per-sample moments: each (N, F)
    def _sample_moments(data):
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        skew = stats.skew(data, axis=1)
        kurt = stats.kurtosis(data, axis=1)
        return mean, var, skew, kurt

    m_moments = _sample_moments(model_data)
    r_moments = _sample_moments(ref_data)
    moment_names = ["Mean", "Variance", "Skewness", "Kurtosis"]

    # Hs binning for reference
    all_hs = np.concatenate([model_wave[:, 0], ref_wave[:, 0]])
    hs_edges = np.linspace(all_hs.min(), all_hs.max(), n_hs_bins + 1)
    hs_centres = 0.5 * (hs_edges[:-1] + hs_edges[1:])
    ref_bin = np.clip(np.digitize(ref_wave[:, 0], hs_edges) - 1, 0, n_hs_bins - 1)

    ncols = min(F, 6)
    nrows_per_moment = int(np.ceil(F / ncols))
    total_rows = 4 * nrows_per_moment
    fig, axes = plt.subplots(
        total_rows,
        ncols,
        figsize=(3 * ncols, 2.5 * total_rows),
        squeeze=False,
    )

    for mi, (m_stat, r_stat, mname) in enumerate(
        zip(m_moments, r_moments, moment_names)
    ):
        for fi in range(F):
            row = mi * nrows_per_moment + fi // ncols
            col = fi % ncols
            ax = axes[row, col]

            # Reference binned profile
            bin_mean = np.full(n_hs_bins, np.nan)
            bin_std = np.full(n_hs_bins, np.nan)
            for b in range(n_hs_bins):
                mask = ref_bin == b
                if mask.sum() < 2:
                    continue
                vals = r_stat[mask, fi]
                bin_mean[b] = vals.mean()
                bin_std[b] = vals.std()
            valid = np.isfinite(bin_mean)
            if valid.any():
                ax.errorbar(
                    hs_centres[valid],
                    bin_mean[valid],
                    yerr=bin_std[valid],
                    fmt="o-",
                    capsize=2,
                    color="grey",
                    markersize=4,
                    label="Ref",
                )

            # Model scatter
            ax.scatter(
                model_wave[:, 0],
                m_stat[:, fi],
                c="C0",
                s=20,
                edgecolor="k",
                linewidth=0.3,
                zorder=3,
                label="Model",
            )
            ax.set_title(f"{mname} \u2014 {feature_names[fi]}", fontsize=7)
            ax.grid(True, alpha=0.3)
            if fi == 0:
                ax.legend(fontsize=5)

        # Hide unused subplots in this moment block
        for fi in range(F, nrows_per_moment * ncols):
            row = mi * nrows_per_moment + fi // ncols
            col = fi % ncols
            axes[row, col].set_visible(False)

    fig.suptitle(f"Moments vs Hs \u2014 step {step}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_psd_comparison(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    step: int = 0,
) -> "plt.Figure":
    """Per-sample PSD overlay (model vs reference) per feature.

    Each sample's Welch PSD is drawn individually to avoid ensemble-averaging
    across different sea states.  Model curves in blue, reference in grey.
    """
    import matplotlib

    matplotlib.use("Agg")
    from scipy import signal

    N_m, T_m, F = model_data.shape
    N_r, T_r, _ = ref_data.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    fs = 1.0 / dt
    nperseg_m = min(T_m, max(256, T_m // 8))
    nperseg_r = min(T_r, max(256, T_r // 8))

    ncols = min(4, F)
    nrows = int(np.ceil(F / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for fi in range(F):
        ax = axes[fi]
        # Reference samples
        for n in range(N_r):
            freqs, pxx = signal.welch(
                ref_data[n, :, fi],
                fs=fs,
                nperseg=nperseg_r,
            )
            f_mask = freqs > 0
            ax.loglog(
                freqs[f_mask],
                pxx[f_mask],
                color="grey",
                alpha=0.3,
                lw=0.4,
                label="Ref" if n == 0 else None,
            )
        # Model samples
        for n in range(N_m):
            freqs, pxx = signal.welch(
                model_data[n, :, fi],
                fs=fs,
                nperseg=nperseg_m,
            )
            f_mask = freqs > 0
            ax.loglog(
                freqs[f_mask],
                pxx[f_mask],
                color="C0",
                alpha=0.6,
                lw=0.6,
                label="Model" if n == 0 else None,
            )

        ax.set_title(feature_names[fi], fontsize=8)
        ax.set_xlabel("Freq [Hz]")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, which="both")

    for j in range(F, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"PSD Comparison \u2014 step {step}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sea-state-dependent evaluation (Hs-binned)
# ---------------------------------------------------------------------------

_WAVE_NAMES = ["Hs", "Tp", "Dir"]


def _build_feature_groups(
    feature_names: list[str],
) -> dict[str, list[int]]:
    """Map FeatureGroup name → column indices for *feature_names*."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        info = FEATURE_REGISTRY.get(name)
        g = info.group.value if info else "other"
        groups.setdefault(g, []).append(i)
    return groups


def _per_sample_summary(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap per-sample statistics across the time axis.

    Returns (std, amax, kurt) each of shape (N, F).
    """
    std = np.std(data, axis=1)
    amax = np.max(np.abs(data), axis=1)
    kurt = stats.kurtosis(data, axis=1)  # excess kurtosis
    return std, amax, kurt


def _pearson_summary(
    summary: np.ndarray,
    wave: np.ndarray,
) -> np.ndarray:
    """(F_summary, 3) Pearson correlation of summary stats vs wave params."""
    F = summary.shape[1]
    corr = np.full((F, 3), np.nan)
    if summary.shape[0] < 4:
        return corr
    for fi in range(F):
        for wi in range(3):
            c = np.corrcoef(summary[:, fi], wave[:, wi])[0, 1]
            corr[fi, wi] = c if np.isfinite(c) else 0.0
    return corr


def _group_corr(
    corr: np.ndarray,
    groups: dict[str, list[int]],
) -> np.ndarray:
    """Average per-feature correlations into groups.

    Returns (n_groups, 3) with row order matching sorted group names.
    """
    names = sorted(groups.keys())
    out = np.zeros((len(names), 3))
    for gi, gn in enumerate(names):
        out[gi] = np.nanmean(corr[groups[gn]], axis=0)
    return out


def plot_sea_state_comparison(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    model_wave: np.ndarray,
    ref_wave: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    physics_config=None,
    n_hs_bins: int = 4,
    step: int = 0,
) -> "plt.Figure":
    """Compact 2×3 figure comparing model vs reference across sea states.

    Top row: derived quantity max vs Hs (excursion, thrust, gen load).
    Bottom row: Pearson correlation heatmaps (model | ref) + variance vs Hs.
    """
    import matplotlib

    matplotlib.use("Agg")

    F = model_data.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]
    groups = _build_feature_groups(feature_names)
    gnames = sorted(groups.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # ---- Row 0: derived quantities max vs Hs ----
    dq_names = ["excursion", "thrust_force", "generator_load"]
    dq_labels = ["Excursion max [m]", "Thrust force max [N]", "Gen. load max [RPM³]"]
    if physics_config is not None:
        m_dq = compute_derived_quantities(model_data, physics_config)
        r_dq = compute_derived_quantities(ref_data, physics_config)
        for col, (dqn, dql) in enumerate(zip(dq_names, dq_labels)):
            ax = axes[0, col]
            r_dmax = np.max(np.abs(r_dq[dqn]), axis=1)
            m_dmax = np.max(np.abs(m_dq[dqn]), axis=1)
            ax.scatter(
                ref_wave[:, 0],
                r_dmax,
                c="grey",
                s=15,
                alpha=0.5,
                edgecolor="none",
                label="Ref",
            )
            ax.scatter(
                model_wave[:, 0],
                m_dmax,
                c="C0",
                s=30,
                edgecolor="k",
                linewidth=0.3,
                label="Model",
                zorder=3,
            )
            ax.set_xlabel("Hs [m]")
            ax.set_ylabel(dql)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    else:
        for col in range(3):
            axes[0, col].set_visible(False)

    # ---- Row 1, cols 0-1: grouped Pearson heatmaps ----
    m_std, _, _ = _per_sample_summary(model_data)
    r_std, _, _ = _per_sample_summary(ref_data)
    m_corr_full = _pearson_summary(m_std, model_wave)
    r_corr_full = _pearson_summary(r_std, ref_wave)

    # Group-averaged + derived rows
    row_labels = list(gnames)
    m_rows = _group_corr(m_corr_full, groups)
    r_rows = _group_corr(r_corr_full, groups)
    if physics_config is not None:
        for dqn in dq_names:
            m_dstd = np.std(m_dq[dqn], axis=1)
            r_dstd = np.std(r_dq[dqn], axis=1)
            mc = np.array(
                [
                    np.corrcoef(m_dstd, model_wave[:, wi])[0, 1]
                    if model_data.shape[0] >= 4
                    else 0.0
                    for wi in range(3)
                ]
            )
            rc = np.array(
                [
                    np.corrcoef(r_dstd, ref_wave[:, wi])[0, 1]
                    if ref_data.shape[0] >= 4
                    else 0.0
                    for wi in range(3)
                ]
            )
            mc = np.where(np.isfinite(mc), mc, 0.0)
            rc = np.where(np.isfinite(rc), rc, 0.0)
            m_rows = np.vstack([m_rows, mc[None]])
            r_rows = np.vstack([r_rows, rc[None]])
            row_labels.append(dqn.replace("_", " ").title())

    for col, (mat, title) in enumerate([(m_rows, "Model"), (r_rows, "Reference")]):
        ax = axes[1, col]
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(_WAVE_NAMES, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                ax.text(
                    jj,
                    ii,
                    f"{mat[ii, jj]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="w" if abs(mat[ii, jj]) > 0.6 else "k",
                )
        ax.set_title(f"Pearson(σ, wave) — {title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)

    # ---- Row 1, col 2: variance vs Hs (binned ref + model scatter) ----
    ax = axes[1, 2]
    all_hs = np.concatenate([model_wave[:, 0], ref_wave[:, 0]])
    hs_edges = np.linspace(all_hs.min(), all_hs.max(), n_hs_bins + 1)
    hs_centres = 0.5 * (hs_edges[:-1] + hs_edges[1:])
    ref_bin = np.clip(np.digitize(ref_wave[:, 0], hs_edges) - 1, 0, n_hs_bins - 1)

    for ci, (gn, gidx) in enumerate(sorted(groups.items())):
        # Reference binned profile
        bin_var = np.full(n_hs_bins, np.nan)
        bin_var_std = np.full(n_hs_bins, np.nan)
        for b in range(n_hs_bins):
            mask = ref_bin == b
            if mask.sum() < 2:
                continue
            per_sample_var = np.mean(np.var(ref_data[mask][:, :, gidx], axis=1), axis=1)
            bin_var[b] = per_sample_var.mean()
            bin_var_std[b] = per_sample_var.std()
        valid = np.isfinite(bin_var)
        if valid.any():
            ax.errorbar(
                hs_centres[valid],
                bin_var[valid],
                yerr=bin_var_std[valid],
                fmt="o-",
                capsize=2,
                color=f"C{ci}",
                label=f"{gn} (ref)",
                markersize=4,
            )
        # Model scatter (per-sample group-averaged variance)
        m_var = np.mean(np.var(model_data[:, :, gidx], axis=1), axis=1)
        ax.scatter(model_wave[:, 0], m_var, marker="x", color=f"C{ci}", s=30, zorder=3)
    ax.set_xlabel("Hs [m]")
    ax.set_ylabel("Variance (group avg)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Variance vs Hs", fontsize=9)

    fig.suptitle(f"Sea-State Evaluation — step {step}", fontsize=11)
    fig.tight_layout()
    return fig


# Re-export canonical plotting functions from thesis.statistics.plots
from thesis.statistics.plots import (  # noqa: E402, F401
    plot_moments as ensemble_moments_plot,
    plot_psd as ensemble_psd_plot,
    plot_maxima as ensemble_maxima_plot,
)


def plot_training_stats(
    result: TrainingStatsResult, feature_names: list[str] | None = None
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Plot training statistics for visual inspection.

    Parameters
    ----------
    result : TrainingStatsResult
        The result of compute_training_stats containing metrics and raw stats.
    feature_names : list[str], optional
        Feature names for labeling. Defaults to f0, f1, ...
    """
    fig_moments, _ = ensemble_moments_plot(
        [result.mom["model"], result.mom["ref"]],
        feature_names,
        labels=["Model", "Reference"],
    )
    fig_psd, _ = ensemble_psd_plot(
        [result.psd["model"], result.psd["ref"]],
        feature_names,
        labels=["Model", "Reference"],
    )
    fig_maxima, _ = ensemble_maxima_plot(
        [result.max["model"], result.max["ref"]],
        feature_names,
        labels=["Model", "Reference"],
    )
    return fig_moments, fig_psd, fig_maxima

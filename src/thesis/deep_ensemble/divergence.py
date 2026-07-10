"""Trajectory feasibility detection for deep-ensemble samples.

Pure, reusable detection logic decoupled from data loading and caching. Operates on
the radial station-keeping excursion ``r = sqrt(eta_x^2 + eta_y^2)`` of a batch of
trajectories and classifies each as feasible or unfeasible. See
:func:`detect_unfeasible` and ``docs/guides/feasibility_detection.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional progress bar
    from tqdm import trange
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    def trange(n, **_kwargs):
        return range(n)


def radial_excursion(samples: np.ndarray) -> np.ndarray:
    """Radial station-keeping excursion ``r = sqrt(eta_x^2 + eta_y^2)``.

    Args:
        samples: array of shape ``(..., T, F)`` with North/East in features 0/1.
    Returns:
        array of shape ``(..., T)`` -- distance from the setpoint at each step.
    """
    return np.sqrt(samples[..., 0] ** 2 + samples[..., 1] ** 2)


# ---------------------------------------------------------------------------
# Initialisation-bias (warm-up) removal + non-convergence detection
# ---------------------------------------------------------------------------


@dataclass
class FeasibilityResult:
    """Per-run feasibility classification and initialisation-transient cut point.

    All arrays have shape ``(n,)`` for ``n`` input series.

    Attributes:
        feasible: run reaches and maintains an *admissible* steady state (keep it).
        warmup: initialisation-transient cut index in *original* (un-batched)
            samples; downstream statistics should use ``series[warmup:]``.
        no_steady_state: MSER-5 found no usable steady-state segment (its warm-up
            estimate is pinned to the end of the admissible search region).
        unit_root: ADF could not reject a unit root on the *retained* tail (a slow
            random-walk / explosive run that MSER-5's mean criterion alone misses).
        level_outlier: the run settles to a stationary but inadmissible level (a
            blow-up followed by a stable offset), flagged as an upper outlier of the
            ensemble's steady-state means.
        exceeds_limit: the run *does* settle, but only after a warm-up longer than
            the global truncation limit ``trunc_limit``; keeping it would require
            cutting more than the shared limit, so it is flagged/discarded rather
            than biasing the common cut towards the slowest runs. All-``False``
            when ``trunc_limit`` is ``None``.
        p_adf: ADF p-value on the retained tail (NaN if the tail was not tested).
        steady_mean: mean of the retained (post-warm-up) series.
        mod_zscore: one-sided modified Z-score of ``steady_mean`` across the settled
            runs (NaN when an explicit ``level_limit`` was used instead).
        batch_size: MSER-5 batch size used.
        alpha: ADF significance level used.
        trunc_limit: global warm-up truncation limit in *original* samples used for
            ``exceeds_limit`` (``None`` when the check was disabled).
    """

    feasible: np.ndarray
    warmup: np.ndarray
    no_steady_state: np.ndarray
    unit_root: np.ndarray
    level_outlier: np.ndarray
    exceeds_limit: np.ndarray
    p_adf: np.ndarray
    steady_mean: np.ndarray
    mod_zscore: np.ndarray
    batch_size: int
    alpha: float
    trunc_limit: int | None


def _mser5_batched(
    r: np.ndarray,
    batch_size: int,
    search_frac: float,
    plateau_tol: float = 0.05,
    clip_quantile: float | None = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised MSER-5 warm-up truncation for a batch of series ``r`` (n, T).

    The excursion is winsorised at ``clip_quantile`` and the warm-up is taken as the
    onset of the MSER plateau (earliest cut within ``plateau_tol`` of the global
    minimum), both to curb MSER-5's over-truncation of heavy-tailed station-keeping
    series. See ``docs/guides/feasibility_detection.md``.

    Returns ``(warmup, failed)`` of shape ``(n,)``: the cut index in *original*
    samples and a flag that the global minimiser is pinned to the search boundary
    (no steady state found).
    """
    r = np.asarray(r, dtype=np.float64)
    n, T = r.shape
    m = T // batch_size
    if m < 4:  # too few batches to locate a transient reliably
        return np.zeros(n, dtype=int), np.ones(n, dtype=bool)

    x = r
    if clip_quantile is not None:  # winsorise the heavy upper tail per series
        cap = np.quantile(r, clip_quantile, axis=1, keepdims=True)
        x = np.minimum(r, cap)

    yb = x[:, : m * batch_size].reshape(n, m, batch_size).mean(axis=2)  # (n, m)

    # Suffix (reverse-cumulative) statistics: column d summarises the retained
    # tail y[:, d:], so MSER for every truncation d is evaluated in one pass.
    rs = np.cumsum(yb[:, ::-1], axis=1)[:, ::-1]            # sum_{i>=d} y_i
    rq = np.cumsum((yb ** 2)[:, ::-1], axis=1)[:, ::-1]     # sum_{i>=d} y_i^2
    k = np.arange(m, 0, -1, dtype=np.float64)[None, :]      # retained count m-d
    ss = rq - rs ** 2 / k                                   # retained sum of squares
    mser = ss / k ** 2                                      # squared SE of trunc. mean

    dmax = max(2, int(m * search_frac))                    # admissible truncation region
    reg = mser[:, :dmax]
    g_argmin = np.argmin(reg, axis=1)                      # global (noisy) minimiser
    failed = g_argmin >= dmax - 1                          # pinned at search boundary
    # Plateau onset: earliest truncation within plateau_tol of the global minimum.
    thresh = reg.min(axis=1) * (1.0 + plateau_tol)
    d_star = (reg <= thresh[:, None]).argmax(axis=1)       # first True per row
    warmup = (d_star * batch_size).astype(int)
    return warmup, failed


def detect_unfeasible(
    r: np.ndarray,
    *,
    batch_size: int = 5,
    search_frac: float = 0.5,
    alpha: float = 0.05,
    min_tail_batches: int = 20,
    adf_max_points: int = 500,
    level_limit: float | None = None,
    mod_z_thresh: float = 3.5,
    trunc_limit: int | None = None,
    plateau_tol: float = 0.05,
    clip_quantile: float | None = 0.95,
    progress: bool = True,
) -> FeasibilityResult:
    """Classify each excursion series as feasible or unfeasible.

    A run is feasible iff MSER-5 finds a steady state, ADF rejects a unit root on the
    retained tail, its settled level is admissible, and (when ``trunc_limit`` is set)
    it settles within that limit; otherwise it is attributed to exactly one reason in
    the priority order no-steady-state > unit-root > level-outlier > exceeds-limit.
    See ``docs/guides/feasibility_detection.md`` for the method.

    Args:
        r: series of shape ``(n, T)`` (e.g. the radial excursion).
        batch_size: MSER-5 batch size (5 is the canonical choice).
        search_frac: fraction of the series over which the warm-up cut may fall; a
            minimiser pinned at this limit means "no steady state".
        alpha: ADF significance level (unit root when ``p_adf > alpha``).
        min_tail_batches: minimum retained tail length for a usable steady-state
            estimate; shorter tails are treated as non-converging.
        adf_max_points: retained tail is decimated to at most this many points before
            the ADF test.
        level_limit: explicit upper bound on the admissible settled level; if ``None``
            the level-outlier check is data-driven via the modified Z-score.
        mod_z_thresh: modified Z-score cut-off for the data-driven level check.
        trunc_limit: global warm-up truncation limit in *original* samples; settling
            runs slower than this are flagged ``exceeds_limit``. ``None`` disables it.
        plateau_tol: MSER plateau tolerance for the warm-up cut (see MSER helper).
        clip_quantile: per-series winsorisation quantile before the MSER search.
        progress: show a progress bar over the per-run ADF tests.

    Returns:
        :class:`FeasibilityResult`.
    """
    from statsmodels.tsa.stattools import adfuller
    import warnings

    r = np.asarray(r, dtype=np.float64)
    n, _ = r.shape

    warmup, failed = _mser5_batched(
        r, batch_size, search_frac, plateau_tol=plateau_tol, clip_quantile=clip_quantile
    )

    p_adf = np.full(n, np.nan)
    steady_mean = np.full(n, np.nan)
    unit_root = np.zeros(n, dtype=bool)
    no_steady_state = failed.copy()

    iterator = trange(n) if progress else range(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in iterator:
            seg = r[i, warmup[i]:]
            if seg.size:
                steady_mean[i] = seg.mean()
            if failed[i]:
                continue
            stride = max(batch_size, seg.size // adf_max_points)  # cap ADF length
            tail = seg[::stride]                    # decorrelated retained tail
            if tail.size < min_tail_batches:
                no_steady_state[i] = True
                continue
            p = adfuller(tail, autolag="AIC")[1]
            p_adf[i] = p
            unit_root[i] = p > alpha

    # Level-outlier check on the runs that did settle to a stationary level.
    settled = ~no_steady_state & ~unit_root
    mod_zscore = np.full(n, np.nan)
    level_outlier = np.zeros(n, dtype=bool)
    if level_limit is not None:
        level_outlier = settled & (steady_mean > float(level_limit))
    else:
        ref = steady_mean[settled]
        if ref.size >= 3:
            med = np.median(ref)
            mad = np.median(np.abs(ref - med))
            if mad > 0:
                mod_zscore = 0.6745 * (steady_mean - med) / mad
                level_outlier = settled & (mod_zscore > mod_z_thresh)

    # Runs that settle only after a warm-up longer than the global truncation limit.
    exceeds_limit = np.zeros(n, dtype=bool)
    if trunc_limit is not None:
        exceeds_limit = warmup > int(trunc_limit)

    feasible = ~(no_steady_state | unit_root | level_outlier | exceeds_limit)

    return FeasibilityResult(
        feasible=feasible,
        warmup=warmup,
        no_steady_state=no_steady_state,
        unit_root=unit_root,
        level_outlier=level_outlier,
        exceeds_limit=exceeds_limit,
        p_adf=p_adf,
        steady_mean=steady_mean,
        mod_zscore=mod_zscore,
        batch_size=int(batch_size),
        alpha=float(alpha),
        trunc_limit=None if trunc_limit is None else int(trunc_limit),
    )


def mser5_truncation(
    series: np.ndarray, *, batch_size: int = 5, search_frac: float = 0.5
) -> tuple[int, bool]:
    """MSER-5 warm-up cut for a single series. See :func:`detect_unfeasible`.

    Returns ``(warmup, failed)``: the initialisation-transient cut index in
    original samples and whether the rule found no steady state (non-converging).
    """
    warmup, failed = _mser5_batched(
        np.asarray(series)[None, :], batch_size, search_frac
    )
    return int(warmup[0]), bool(failed[0])

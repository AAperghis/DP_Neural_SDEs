"""POT-based extreme value analysis pipeline.

Full pipeline: ACF → declustering lag → threshold → GPD fit → compound-Poisson
3h-max distribution → MPM.

All public functions operate on plain numpy arrays of shape ``(N, T, F)``.
"""

from __future__ import annotations

import logging
import os
import time as _time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import jit, vmap
from scipy import stats
from tqdm import trange


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TEMPORARY profiling.  Enabled by setting EV_PROFILE=1.  Remove when done.
# ---------------------------------------------------------------------------
_EV_PROFILE = os.environ.get("EV_PROFILE", "0") == "1"
_EV_TIMES: dict[str, float] = defaultdict(float)
_EV_COUNTS: dict[str, int] = defaultdict(int)


@contextmanager
def _ev_timer(label: str):
    if not _EV_PROFILE:
        yield
        return
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        _EV_TIMES[label] += _time.perf_counter() - t0
        _EV_COUNTS[label] += 1


def _ev_profile_reset() -> None:
    _EV_TIMES.clear()
    _EV_COUNTS.clear()


def _ev_profile_report() -> None:
    if not _EV_PROFILE or not _EV_TIMES:
        return
    total = sum(_EV_TIMES.values())
    width = max(len(k) for k in _EV_TIMES)
    print("\n=== EV POT profiling (EV_PROFILE=1) ===")
    for label, secs in sorted(_EV_TIMES.items(), key=lambda kv: -kv[1]):
        n = _EV_COUNTS[label]
        print(
            f"  {label:<{width}}  {secs:8.3f}s  "
            f"{100 * secs / total:5.1f}%  ({n} calls, {secs / n * 1e3:.2f} ms/call)"
        )
    print(f"  {'TOTAL':<{width}}  {total:8.3f}s\n")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class POTResult:
    """Result of the POT extreme-value pipeline for a single feature.

    Plotting helpers:
        Use :meth:`pdf` and :meth:`cdf` to evaluate the compound-Poisson max
        distribution on an arbitrary grid.  A convenience :meth:`pdf_grid`
        returns a ready-made (x, pdf) pair for quick plotting.

    Attributes:
        shape: GPD shape parameter (ξ).
        scale: GPD scale parameter (σ).
        threshold: Exceedance threshold *u*.
        peak_rate: Mean number of independent peaks per target duration.
        decluster_lag: Minimum separation between independent peaks (seconds).
        correlation_time: ACF first zero-crossing (seconds).
        n_peaks: Total number of declustered peaks used for the fit.
        mpm: Most probable maximum (mode of the max-distribution PDF).
        peaks: Raw peak values (1-D).
        observed_maxima: Per-realisation observed maxima (1-D, length N).
        ks_statistic: KS statistic of the compound-max CDF vs observed maxima.
        ks_pvalue: KS p-value.
        mpm_ci_low: Lower 95 % bootstrap CI bound on MPM.
        mpm_ci_high: Upper 95 % bootstrap CI bound on MPM.
    """

    shape: float
    scale: float
    threshold: float
    peak_rate: float
    decluster_lag: float
    correlation_time: float
    n_peaks: int
    mpm: float
    peaks: np.ndarray
    observed_maxima: np.ndarray
    ks_statistic: float
    ks_pvalue: float
    mpm_ci_low: float = 0.0
    mpm_ci_high: float = 0.0

    # -- evaluation helpers ------------------------------------------------

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compound-Poisson max CDF: F(x) = exp(-λ S̄(x))."""
        return _compound_max_cdf(
            np.asarray(x, dtype=float),
            self.shape,
            self.scale,
            self.threshold,
            self.peak_rate,
        )

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-PDF of the compound-Poisson max distribution."""
        return _compound_max_logpdf(
            np.asarray(x, dtype=float),
            self.shape,
            self.scale,
            self.threshold,
            self.peak_rate,
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF of the compound-Poisson 3h-max distribution."""
        return np.exp(self.logpdf(x))

    def pdf_grid(
        self, n: int = 500, pad: float = 0.15
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, pdf)`` on a grid spanning the observed maxima range.

        Args:
            n: Number of grid points.
            pad: Fractional padding beyond the observed maxima range.
        """
        lo = (
            self.observed_maxima.min() * (1 - pad)
            if self.observed_maxima.min() > 0
            else self.observed_maxima.min() - pad * abs(self.observed_maxima.min())
        )
        hi = self.observed_maxima.max() * (1 + pad)
        x = np.linspace(lo, hi, n)
        return x, self.pdf(x)


@dataclass
class EVResult:
    """Container for extreme-value results across all features.

    Provides per-feature access via ``result[f]`` and convenience array
    properties (``observed_maxima``, ``max_mean``, ``max_std``, ``mpm``)
    for downstream compatibility.

    In **per-instance** mode (``per_instance=True`` in
    :func:`pot_extreme_values`), each ensemble member is fitted
    independently.  Array properties then return ``(N, F)`` instead of
    ``(F,)``, and individual results are accessible via ``result[n, f]``.

    Features where the GPD fit failed receive a fallback ``POTResult``
    with ``peak_rate=0`` and ``mpm`` set to the mean of observed maxima.

    Attributes:
        features: Dict mapping feature index → :class:`POTResult`.
        n_features: Total number of features.
    """

    features: dict[int, POTResult]
    n_features: int = field(default=0)
    _instances: list[dict[int, POTResult]] = field(default_factory=list, repr=False)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            n, f = key
            return self._instances[n][f]
        return self.features[key]

    def __len__(self) -> int:
        if self._instances:
            return len(self._instances)
        return len(self.features)

    def __iter__(self):
        return iter(self.features.values())

    # -- Array-style convenience properties --------------------------------

    def __post_init__(self):
        if self.n_features == 0:
            self.n_features = max(self.features.keys(), default=-1) + 1

    @property
    def is_per_instance(self) -> bool:
        """True when per-instance results are stored."""
        return len(self._instances) > 0

    @property
    def observed_maxima(self) -> np.ndarray:
        """(N, F) array of per-realisation observed maxima."""
        if self._instances:
            return np.array(
                [
                    [d[f].observed_maxima.item() for f in sorted(d)]
                    for d in self._instances
                ]
            )
        arrs = [self.features[f].observed_maxima for f in sorted(self.features)]
        return np.column_stack(arrs)

    @property
    def max_mean(self) -> np.ndarray:
        """(F,) mean of observed maxima per feature."""
        return self.observed_maxima.mean(axis=0)

    @property
    def max_std(self) -> np.ndarray:
        """(F,) std of observed maxima per feature."""
        return self.observed_maxima.std(axis=0, ddof=1)

    @property
    def mpm(self) -> np.ndarray:
        """(F,) or (N, F) array of most-probable-maximum values."""
        if self._instances:
            return np.array([[d[f].mpm for f in sorted(d)] for d in self._instances])
        return np.array([r.mpm for _, r in sorted(self.features.items())])

    @property
    def mpm_ci_low(self) -> np.ndarray:
        """(F,) or (N, F) lower 95 % bootstrap CI bound on MPM."""
        if self._instances:
            return np.array(
                [[d[f].mpm_ci_low for f in sorted(d)] for d in self._instances]
            )
        return np.array([r.mpm_ci_low for _, r in sorted(self.features.items())])

    @property
    def mpm_ci_high(self) -> np.ndarray:
        """(F,) or (N, F) upper 95 % bootstrap CI bound on MPM."""
        if self._instances:
            return np.array(
                [[d[f].mpm_ci_high for f in sorted(d)] for d in self._instances]
            )
        return np.array([r.mpm_ci_high for _, r in sorted(self.features.items())])


# ---------------------------------------------------------------------------
# Internal math
# ---------------------------------------------------------------------------


def _compound_max_cdf(
    x: np.ndarray, xi: float, sigma: float, th: float, lam: float
) -> np.ndarray:
    z = (x - th) / sigma
    if abs(xi) < 1e-8:
        S = np.exp(-z)
    else:
        arg = 1 + xi * z
        S = np.where(arg > 0, arg ** (-1 / xi), 0.0)
    return np.exp(-lam * S)


def _compound_max_logpdf(
    x: np.ndarray, xi: float, sigma: float, th: float, lam: float
) -> np.ndarray:
    z = (x - th) / sigma
    if abs(xi) < 1e-8:
        S = np.exp(-z)
        log_s_deriv = -z
    else:
        arg = 1 + xi * z
        valid = arg > 0
        S = np.where(valid, arg ** (-1 / xi), 0.0)
        log_s_deriv = np.where(
            valid,
            -(1 / xi + 1) * np.log(np.maximum(arg, 1e-300)),
            -np.inf,
        )
    log_F = -lam * S
    return log_F + np.log(lam / sigma) + log_s_deriv


def _mpm_grid(xi: float, sigma: float, th: float, lam: float) -> float:
    """Mode of the compound-Poisson max PDF via grid search."""
    if xi < -1e-8:
        upper = th + sigma / abs(xi) - 1e-8
    else:
        upper = th + 15 * sigma
    x = np.linspace(th + 1e-8, upper, 2000)
    return float(x[np.argmax(_compound_max_logpdf(x, xi, sigma, th, lam))])


# ---------------------------------------------------------------------------
# JAX-accelerated batched GPD fit + MPM
# ---------------------------------------------------------------------------
# The GPD maximum-likelihood fit, the MPM grid search and the bootstrap are the
# only expensive, repeated parts of the pipeline.  Once the (variable-length)
# peaks have been extracted they are pure numeric kernels, so we run them for
# *all* features / models / bootstrap replicates at once through a single
# vmapped, jitted kernel instead of a Python loop.  The irregular work
# (declustering) stays in NumPy — it is cheap and does not vectorise cleanly.


def _bucket_len(n: int, minimum: int = 16) -> int:
    """Round ``n`` up to the next power of two (floored at ``minimum``).

    Padding every excess vector in a batch to one of a few power-of-two lengths
    bounds the number of distinct array shapes JAX sees, so the batched kernel
    compiles a handful of times across a whole run rather than once per feature.
    """
    if n <= minimum:
        return minimum
    return 1 << (n - 1).bit_length()


def _mom_init(excesses: np.ndarray) -> np.ndarray:
    """Method-of-moments initial guess ``[xi, log sigma]`` for the GPD fit.

    For a GPD with mean ``m`` and variance ``v``: ``m^2/v = 1 - 2*xi`` and
    ``sigma = m * (1 - xi)``, hence ``xi = (1 - m^2/v) / 2``.
    """
    exc = np.asarray(excesses, dtype=np.float64)
    m1 = float(exc.mean())
    m2 = float(exc.var())
    if m2 > 0:
        ratio = m1 * m1 / m2
        xi0 = 0.5 * (1.0 - ratio)
        sigma0 = max(0.5 * m1 * (ratio + 1.0), 1e-6)
    else:
        xi0, sigma0 = 0.0, max(m1, 1e-6)
    return np.array([xi0, np.log(sigma0)], dtype=np.float64)


@jit
def _compound_max_logpdf_jax(
    x: jnp.ndarray, xi: float, sigma: float, th: float, lam: float
) -> jnp.ndarray:
    """JAX twin of :func:`_compound_max_logpdf` (used by the MPM grid)."""
    z = (x - th) / sigma
    arg = 1.0 + xi * z
    safe_arg = jnp.where(arg > 1e-10, arg, 1e-10)
    S = jnp.where(jnp.abs(xi) < 1e-8, jnp.exp(-z), safe_arg ** (-1.0 / xi))
    log_s_deriv = jnp.where(
        jnp.abs(xi) < 1e-8, -z, -(1.0 / xi + 1.0) * jnp.log(safe_arg)
    )
    log_F = -lam * S
    return jnp.where(arg > 0, log_F + jnp.log(lam / sigma) + log_s_deriv, -jnp.inf)


def _mpm_grid_jax(xi, sigma, th, lam):
    """Mode of the compound-Poisson max PDF via a 2000-point grid (JAX)."""
    upper = jnp.where(xi < -1e-8, th + sigma / jnp.abs(xi) - 1e-8, th + 15.0 * sigma)
    x = jnp.linspace(th + 1e-8, upper, 2000)
    return x[jnp.argmax(_compound_max_logpdf_jax(x, xi, sigma, th, lam))]


def _gpd_nll_masked(params, excesses, mask):
    """GPD negative log-likelihood with a boolean mask over padded entries.

    Outside the GPD support (``1 + xi*z <= 0`` for some real excess) the
    log-likelihood is undefined.  We clamp the log term and add a smooth
    quadratic penalty in the constraint violation, which gives the optimiser a
    restoring gradient back into the feasible region.  A *constant* penalty
    would instead leave a flat, zero-gradient plateau that traps the line search
    at a garbage point (the failure mode that collapsed ``sigma`` to ~1e-10 and
    ``xi`` to large negative values for some features).

    The penalty is identically zero at and around the MLE (where every
    ``arg`` sits well above the ``1e-10`` floor), so it does not shift the fit.
    """
    xi, log_sigma = params[0], params[1]
    sigma = jnp.exp(log_sigma)
    n = mask.sum()
    z = excesses / sigma
    arg = 1.0 + xi * z
    safe_arg = jnp.where(arg > 1e-10, arg, 1e-10)
    nll_exp = n * log_sigma + jnp.sum(jnp.where(mask, z, 0.0))
    nll_gpd = n * log_sigma + (1.0 / xi + 1.0) * jnp.sum(
        jnp.where(mask, jnp.log(safe_arg), 0.0)
    )
    nll = jnp.where(jnp.abs(xi) < 1e-6, nll_exp, nll_gpd)
    violation = jnp.where(mask, jnp.maximum(1e-10 - arg, 0.0), 0.0)
    penalty = 1e6 * jnp.sum(violation**2)
    return nll + penalty


def _nll_fn(params, args):
    """Optimistix objective wrapper; ``args = (excesses, mask)``."""
    excesses, mask = args
    return _gpd_nll_masked(params, excesses, mask)


# Optimistix's backtracking line search rejects steps that overshoot into the
# unfeasible region, unlike the Wolfe search in the (now unmaintained)
# ``jax.scipy.optimize.minimize`` BFGS, which used to settle on garbage points.
_GPD_SOLVER = optx.BFGS(rtol=1e-6, atol=1e-6)


@jit
def _fit_mpm_batched_jax(exc, mask, x0s, th, lam):
    """Batched masked-GPD MLE (BFGS) + MPM grid over the whole batch.

    Args:
        exc, mask: Padded excess vectors and validity masks, shape (M, L).
        x0s: Per-item initial ``[xi, log sigma]``, shape (M, 2).
        th, lam: Per-item threshold and Poisson rate, shape (M,) (traced, so
            distinct values reuse the same compiled program).

    Returns ``(xi, sigma, mpm)`` each of shape ``(M,)``.
    """

    def _one(e, m, x0, t, lam_i):
        sol = optx.minimise(
            _nll_fn, _GPD_SOLVER, x0, args=(e, m), max_steps=256, throw=False
        )
        xi = sol.value[0]
        sigma = jnp.exp(sol.value[1])
        return xi, sigma, _mpm_grid_jax(xi, sigma, t, lam_i)

    return vmap(_one, in_axes=(0, 0, 0, 0, 0))(exc, mask, x0s, th, lam)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def acf_zero_crossing(
    data_2d: np.ndarray, dt: float, max_lag: int | None = None
) -> float:
    """Mean-ACF first zero-crossing time (seconds) for one feature.

    Args:
        data_2d: Time series per realisation, shape (N, T).
        dt: Sampling interval in seconds.
        max_lag: Maximum lag in samples (default T // 2).

    Returns:
        First zero-crossing of the mean ACF, in seconds.
    """
    N, T = data_2d.shape
    if max_lag is None:
        max_lag = min(500, T // 2)
    n_fft = 1 << int(np.ceil(np.log2(2 * T - 1)))
    acf_mean = np.zeros(max_lag)
    for r in range(N):
        sig = data_2d[r] - data_2d[r].mean()
        fft_sig = np.fft.rfft(sig, n=n_fft)
        acf = np.fft.irfft(fft_sig * np.conj(fft_sig), n=n_fft)[:max_lag]
        acf /= acf[0]
        acf_mean += acf
    acf_mean /= N
    zero_idx = np.where(acf_mean[1:] <= 0)[0]
    if len(zero_idx) > 0:
        return float((zero_idx[0] + 1) * dt)
    return float(max_lag * dt)


def decluster_peaks(
    signal: np.ndarray, time: np.ndarray, th: float, dl: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract declustered peaks above *th* with minimum separation *dl* (s).

    Only exceedance samples can start or update a cluster maximum, so the scan
    iterates over the exceedance indices directly rather than every sample (a
    large saving when the threshold is high and exceedances are sparse). Two
    consecutive exceedances stay in the same cluster unless a below-threshold
    sample strictly between them lies at least *dl* seconds after the cluster's
    running peak time — equivalent to the original sample-by-sample rule.
    """
    above_idx = np.flatnonzero(signal > th)
    if above_idx.size == 0:
        return np.array([]), np.array([])
    peak_values: list[float] = []
    peak_times: list[float] = []
    m = above_idx.size
    k = 0
    while k < m:
        i = above_idx[k]
        best_val, best_t = signal[i], time[i]
        prev = i
        k2 = k + 1
        while k2 < m:
            q = above_idx[k2]
            # Below-threshold samples strictly between prev and q (the largest
            # gap is at q-1) end the cluster if one lies >= dl after the peak.
            if q - 1 > prev and time[q - 1] - best_t >= dl:
                break
            if signal[q] > best_val:
                best_val, best_t = signal[q], time[q]
            prev = q
            k2 += 1
        peak_values.append(best_val)
        peak_times.append(best_t)
        k = k2
    return np.array(peak_values), np.array(peak_times)


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def _prepare_feature(
    feat,
    time,
    dt,
    *,
    threshold_quantile,
    dl_factor,
    min_peaks,
    target_duration,
    n_bootstrap,
    lower_tail,
    rng,
):
    """NumPy-side preparation for one feature.

    Performs tail detection, the ACF correlation time, thresholding,
    declustering and bootstrap *resampling* — but not the GPD fit.  The fit,
    MPM and bootstrap-MPM are deferred so that every feature/model can be solved
    together in a single batched JAX call (see :func:`_solve_batched`).
    """
    N, T = feat.shape
    realisation_duration = T * dt

    # 0. Detect tail direction
    if lower_tail is None:
        p01, p99 = np.percentile(feat, [1, 99])
        upper_mean = np.abs(feat[feat >= p99].mean())
        lower_mean = np.abs(feat[feat <= p01].mean())
        negated = lower_mean > upper_mean
    else:
        negated = lower_tail
    if negated:
        feat = -feat

    # 1-3. Correlation time, declustering lag, threshold
    with _ev_timer("prep: acf_zero_crossing"):
        tau_0 = acf_zero_crossing(feat, dt)
    dl = dl_factor * tau_0
    th = float(np.percentile(feat.ravel(), 100 * threshold_quantile))

    # 4. Declustered peaks + observed maxima.  Decluster each realisation once
    #    and reuse the per-realisation peaks for the bootstrap below (which
    #    resamples whole realisations), avoiding a second declustering pass.
    with _ev_timer("prep: decluster_peaks"):
        peaks_per_real = [decluster_peaks(feat[n], time, th, dl)[0] for n in range(N)]
    nonempty = [p for p in peaks_per_real if len(p) > 0]
    peaks = np.concatenate(nonempty) if nonempty else np.array([])
    maxima = feat.max(axis=1)  # (N,)

    prep = dict(
        negated=negated,
        tau_0=tau_0,
        dl=dl,
        th=th,
        peaks=peaks,
        maxima=maxima,
        n_peaks=len(peaks),
        fallback=len(peaks) < min_peaks,
    )
    if prep["fallback"]:
        return prep

    excesses = peaks - th
    prep["excess"] = excesses
    prep["x0"] = _mom_init(excesses)
    prep["lam"] = (len(peaks) / N) * (target_duration / realisation_duration)

    if n_bootstrap > 0:
        with _ev_timer("prep: bootstrap_resamples"):
            prep.update(
                _gen_bootstrap_resamples(
                    feat, time, dt, th, dl, peaks, N, T,
                    target_duration, realisation_duration, n_bootstrap, min_peaks, rng,
                    peaks_per_real=peaks_per_real,
                )
            )
    return prep


def _gen_bootstrap_resamples(
    feat, time, dt, th, dl, peaks, N, T,
    target_duration, realisation_duration, n_bootstrap, min_peaks, rng,
    peaks_per_real=None,
):
    """Generate per-replicate bootstrap *excess* vectors.

    Reproduces the reference RNG draw order exactly: full-realisation resampling
    for ``N > 1`` (declustering is row-independent, so resampled peaks equal the
    precomputed per-realisation peaks concatenated by index) and a block
    bootstrap for ``N == 1``.  The GPD fits are deferred to the batched solve.
    """
    if N > 1:
        if peaks_per_real is None:
            peaks_per_real = [
                decluster_peaks(feat[n], time, th, dl)[0] for n in range(N)
            ]
    else:
        block_len = max(int(2 * dl / dt), 1)
        n_blocks = T // block_len

    boot_exc: list[np.ndarray] = []
    boot_lam = np.zeros(n_bootstrap)
    boot_valid = np.zeros(n_bootstrap, dtype=bool)

    for b in range(n_bootstrap):
        try:
            if N > 1 and peaks_per_real is not None:
                idx = rng.choice(N, size=N, replace=True)
                bp = np.concatenate([peaks_per_real[i] for i in idx])
                blam = (len(bp) / N) * (target_duration / realisation_duration)
            elif n_blocks < 2:
                # Too few blocks — fall back to resampling peaks
                n_bp = max(rng.poisson(len(peaks)), 1)
                bp = rng.choice(peaks, size=n_bp, replace=True)
                blam = (len(bp) / 1) * (target_duration / realisation_duration)
            else:
                bidx = rng.choice(n_blocks, size=n_blocks, replace=True)
                boot_series = np.concatenate(
                    [feat[0, i * block_len : (i + 1) * block_len] for i in bidx]
                )
                boot_time = np.arange(len(boot_series)) * dt
                bp, _ = decluster_peaks(boot_series, boot_time, th, dl)
                blam = (len(bp) / 1) * (target_duration / (len(boot_series) * dt))
            if len(bp) < min_peaks:
                boot_exc.append(np.empty(0))
                boot_lam[b] = blam
                continue
            boot_exc.append(np.asarray(bp, dtype=np.float64) - th)
            boot_lam[b] = blam
            boot_valid[b] = True
        except Exception:
            boot_exc.append(np.empty(0))

    # Pad ragged replicate excesses to (B, Lf) + mask, with per-replicate inits.
    Lf = max(max((len(e) for e in boot_exc), default=0), 1)
    bexc = np.zeros((n_bootstrap, Lf))
    bmask = np.zeros((n_bootstrap, Lf), dtype=bool)
    bx0 = np.zeros((n_bootstrap, 2))
    for b, e in enumerate(boot_exc):
        ne = len(e)
        if ne:
            bexc[b, :ne] = e
            bmask[b, :ne] = True
            bx0[b] = _mom_init(e)
    return dict(
        boot_exc=bexc, boot_mask=bmask, boot_x0=bx0,
        boot_lam=boot_lam, boot_valid=boot_valid,
    )


def _solve_batched(preps):
    """Fill ``xi``/``sigma``/``mpm`` and bootstrap CIs on every non-fallback
    prep using two batched JAX kernels: one for the point estimates and one for
    all bootstrap replicates across all features/models at once.
    """
    valid = [p for p in preps if not p["fallback"]]
    if not valid:
        return

    # --- point estimates: one batched fit over all features/models ---
    L = _bucket_len(max(len(p["excess"]) for p in valid))
    M = len(valid)
    exc = np.zeros((M, L))
    mask = np.zeros((M, L), dtype=bool)
    x0s = np.zeros((M, 2))
    th = np.zeros(M)
    lam = np.zeros(M)
    for i, p in enumerate(valid):
        ne = len(p["excess"])
        exc[i, :ne] = p["excess"]
        mask[i, :ne] = True
        x0s[i] = p["x0"]
        th[i] = p["th"]
        lam[i] = p["lam"]
    xi_a, sig_a, mpm_a = (
        np.asarray(a)
        for a in _fit_mpm_batched_jax(
            jnp.asarray(exc), jnp.asarray(mask), jnp.asarray(x0s),
            jnp.asarray(th), jnp.asarray(lam),
        )
    )
    for i, p in enumerate(valid):
        xi, sigma, mpm = float(xi_a[i]), float(sig_a[i]), float(mpm_a[i])
        if not (np.isfinite(xi) and np.isfinite(sigma) and sigma > 0):
            xi, _, sigma = stats.genpareto.fit(p["excess"], floc=0)
            mpm = _mpm_grid(xi, sigma, p["th"], p["lam"])
        p["xi"], p["sigma"], p["mpm"] = xi, sigma, mpm

    # --- bootstrap: one batched fit over all (feature × replicate) ---
    boot_feats = [p for p in valid if "boot_exc" in p]
    if not boot_feats:
        return
    Lb = _bucket_len(max(p["boot_exc"].shape[1] for p in boot_feats))
    chunks = []
    for p in boot_feats:
        B, lf = p["boot_exc"].shape
        e = np.zeros((B, Lb))
        e[:, :lf] = p["boot_exc"]
        m = np.zeros((B, Lb), dtype=bool)
        m[:, :lf] = p["boot_mask"]
        chunks.append(
            (e, m, p["boot_x0"], np.full(B, p["th"]), p["boot_lam"], p["boot_valid"], B)
        )
    _, _, mpm_boot = _fit_mpm_batched_jax(
        jnp.asarray(np.concatenate([c[0] for c in chunks])),
        jnp.asarray(np.concatenate([c[1] for c in chunks])),
        jnp.asarray(np.concatenate([c[2] for c in chunks])),
        jnp.asarray(np.concatenate([c[3] for c in chunks])),
        jnp.asarray(np.concatenate([c[4] for c in chunks])),
    )
    valid_flags = np.concatenate([c[5] for c in chunks])
    mpm_boot = np.asarray(mpm_boot)
    mpm_boot = np.where(np.isfinite(mpm_boot) & valid_flags, mpm_boot, np.nan)
    off = 0
    for p, c in zip(boot_feats, chunks):
        B = c[6]
        bm = mpm_boot[off : off + B]
        off += B
        p["n_boot_failed"] = int(np.isnan(bm).sum())
        good = bm[~np.isnan(bm)]
        if len(good) >= 2:
            p["ci_low"] = float(np.percentile(good, 2.5))
            p["ci_high"] = float(np.percentile(good, 97.5))
        else:
            p["ci_low"], p["ci_high"] = p["mpm"], p["mpm"]


def _build_potresult(p) -> POTResult:
    """Assemble a :class:`POTResult` from a solved prep dict (with negation)."""
    maxima = p["maxima"]
    if p["fallback"]:
        # No valid GPD fit — summarise via observed maxima.
        pot = POTResult(
            shape=0.0,
            scale=1.0,
            threshold=p["th"],
            peak_rate=0.0,
            decluster_lag=p["dl"],
            correlation_time=p["tau_0"],
            n_peaks=p["n_peaks"],
            mpm=float(np.mean(maxima)),
            peaks=p["peaks"] if len(p["peaks"]) > 0 else np.array([p["th"]]),
            observed_maxima=maxima,
            ks_statistic=1.0,
            ks_pvalue=0.0,
        )
        if p["negated"]:
            pot = POTResult(
                shape=0.0,
                scale=1.0,
                threshold=-pot.threshold,
                peak_rate=0.0,
                decluster_lag=p["dl"],
                correlation_time=p["tau_0"],
                n_peaks=pot.n_peaks,
                mpm=-pot.mpm,
                peaks=-pot.peaks,
                observed_maxima=-pot.observed_maxima,
                ks_statistic=1.0,
                ks_pvalue=0.0,
            )
        return pot

    xi, sigma, th, lam = p["xi"], p["sigma"], p["th"], p["lam"]

    # KS test of the compound-max CDF against observed per-realisation maxima.
    ks_stat, ks_p = stats.kstest(
        maxima,
        lambda x, _xi=xi, _s=sigma, _th=th, _l=lam: _compound_max_cdf(
            np.asarray(x), _xi, _s, _th, _l
        ),
    )

    pot = POTResult(
        shape=xi,
        scale=sigma,
        threshold=th,
        peak_rate=lam,
        decluster_lag=p["dl"],
        correlation_time=p["tau_0"],
        n_peaks=p["n_peaks"],
        mpm=p["mpm"],
        peaks=p["peaks"],
        observed_maxima=maxima,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        mpm_ci_low=p.get("ci_low", p["mpm"]),
        mpm_ci_high=p.get("ci_high", p["mpm"]),
    )

    # Negate results back for lower-tail features
    if p["negated"]:
        pot = POTResult(
            shape=pot.shape,
            scale=pot.scale,
            threshold=-pot.threshold,
            peak_rate=pot.peak_rate,
            decluster_lag=pot.decluster_lag,
            correlation_time=pot.correlation_time,
            n_peaks=pot.n_peaks,
            mpm=-pot.mpm,
            peaks=-pot.peaks,
            observed_maxima=-pot.observed_maxima,
            ks_statistic=pot.ks_statistic,
            ks_pvalue=pot.ks_pvalue,
            mpm_ci_low=-pot.mpm_ci_high,  # swap after negation
            mpm_ci_high=-pot.mpm_ci_low,
        )
    return pot


def _run_pot(
    items,
    dt,
    *,
    threshold_quantile,
    dl_factor,
    min_peaks,
    target_duration,
    n_bootstrap,
    verbose,
    warn_on_bootstrap_failures,
    lower_tail,
):
    """Core driver: prepare every ``(item, feature)`` in NumPy, then solve them
    all with two batched JAX calls.  Returns one :class:`EVResult` per item.
    """
    preps = []
    _ev_profile_reset()
    item_iter = (
        trange(len(items), disable=not verbose, desc="POT items")
        if len(items) > 1
        else range(len(items))
    )
    for it in item_iter:
        data = np.asarray(items[it], dtype=np.float64)
        _, T, F = data.shape
        time = np.arange(T) * dt
        td = T * dt if target_duration is None else target_duration
        for f in range(F):
            with _ev_timer("prep: total (_prepare_feature)"):
                p = _prepare_feature(
                    data[:, :, f],
                    time,
                    dt,
                    threshold_quantile=threshold_quantile,
                    dl_factor=dl_factor,
                    min_peaks=min_peaks,
                    target_duration=td,
                    n_bootstrap=n_bootstrap,
                    lower_tail=lower_tail,
                    rng=np.random.default_rng(42 + f),
                )
            p["item"], p["feat"] = it, f
            preps.append(p)

    with _ev_timer("solve: _solve_batched (JAX)"):
        _solve_batched(preps)

    _ev_profile_report()

    if warn_on_bootstrap_failures:
        for p in preps:
            n_failed = p.get("n_boot_failed", 0)
            if n_failed:
                _log.warning(
                    "Feature %d: %d / %d bootstrap replicates failed.",
                    p["feat"],
                    n_failed,
                    n_bootstrap,
                )

    results = []
    for it in range(len(items)):
        F = np.asarray(items[it]).shape[2]
        feats = {p["feat"]: _build_potresult(p) for p in preps if p["item"] == it}
        results.append(EVResult(features=feats, n_features=F))
    return results


def pot_extreme_values(
    data: np.ndarray,
    dt: float,
    *,
    threshold_quantile: float = 0.95,
    dl_factor: float = 2.0,
    min_peaks: int = 5,
    target_duration: float | None = None,
    n_bootstrap: int = 200,
    per_instance: bool = False,
    verbose: bool = False,
    warn_on_bootstrap_failures: bool = True,
    lower_tail: bool | None = None,
) -> EVResult:
    """Run the full POT pipeline on an ``(N, T, F)`` ensemble.

    Steps:
        1. Estimate correlation time τ₀ per feature (ACF first zero-crossing).
        2. Set declustering lag ``dl = dl_factor × τ₀``.
        3. Compute threshold as the *threshold_quantile* of the pooled data.
        4. Decluster peaks, fit GPD to excesses.
        5. Build compound-Poisson 3h-max distribution, compute MPM.
        6. KS test of compound-max CDF against observed per-realisation maxima.

    All features are declustered in NumPy and then the GPD fits, MPM grids and
    bootstraps are solved together in two batched JAX kernels (rather than a
    Python loop over features).

    Args:
        data: Ensemble time series, shape (N, T, F) (already absolute-valued
            for features where you want the positive exceedance).
        dt: Sampling interval in seconds.
        threshold_quantile: Quantile for automatic threshold (default 0.95).
        dl_factor: Multiplier on τ₀ for the declustering lag (default 2).
        min_peaks: Minimum peaks required for a valid GPD fit (default 5).
        target_duration: Duration in seconds for the max distribution.
            Default: full realisation length ``T × dt``.
        n_bootstrap: Number of bootstrap resamples for MPM CI (default 200).
        per_instance: If True, fit each ensemble member independently
            and return an :class:`EVResult` with ``(N, F)`` MPM values.
        lower_tail: If ``None`` (default), automatically detect per feature
            whether extremes are in the lower tail by comparing tail spread
            below vs above the median.  If ``True``/``False``, force
            lower/upper tail for all features.

    Returns:
        EVResult — per-feature :class:`POTResult` objects.
    """
    data = np.asarray(data, dtype=np.float64)
    N, _, F = data.shape

    if per_instance:
        instances = _run_pot(
            [data[n : n + 1] for n in range(N)],
            dt,
            threshold_quantile=threshold_quantile,
            dl_factor=dl_factor,
            min_peaks=min_peaks,
            target_duration=target_duration,
            n_bootstrap=n_bootstrap,
            verbose=verbose,
            warn_on_bootstrap_failures=warn_on_bootstrap_failures,
            lower_tail=lower_tail,
        )
        return EVResult(
            features=instances[0].features,
            n_features=F,
            _instances=[r.features for r in instances],
        )

    return _run_pot(
        [data],
        dt,
        threshold_quantile=threshold_quantile,
        dl_factor=dl_factor,
        min_peaks=min_peaks,
        target_duration=target_duration,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
        warn_on_bootstrap_failures=warn_on_bootstrap_failures,
        lower_tail=lower_tail,
    )[0]

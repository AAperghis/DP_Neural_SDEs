"""
Verification of Newman's slowly-varying drift force implementation.

Verifies:
  1. Mean:     time-average of Newman output matches the analytical mean
               drift force (Fossen 2021, Eq. 8.87).
  2. Spectrum: PSD of Newman time series matches Pinkster's slow-drift
               force spectrum (Pinkster 1980, Eq. 4.59).
  3. Variance: integral of the Pinkster spectrum matches the empirical
               variance of the time series.

Produces figures for the report:
  1. Drift coefficients T_i(w) overlaid with JONSWAP spectrum
  2. Newman drift force time series with running mean → analytical mean
  3. Convergence of time-averaged mean vs simulation length
  4. PSD comparison: Welch estimate vs Pinkster slow-drift spectrum
  5. Heading sweep: mean and std across 0–330°

References:
  - Pinkster, J.A. (1980). Low frequency second order wave exciting
    forces on floating structures. PhD thesis, TU Delft.
  - Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and
    Motion Control. 2nd ed., Wiley. Chapter 8.

Usage:
    uv run python -m thesis.full_order_dp.verify_drift [--Hs 2.5] [--Tp 8.0]
        [--beta 0] [--T 20000] [--save-dir tmp/test_plots]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from thesis.full_order_dp.environment import (
    WaveDriftCoefficients,
    jonswap_spectrum,
    newman_drift_force,
)


def _interp_coeffs(coeffs, coeff_2d, omega_eval, heading_eval):
    interp = RegularGridInterpolator(
        (coeffs.omega.ravel(), coeffs.headings.ravel()),
        coeff_2d,
        method="linear",
        bounds_error=False,
        fill_value=None,  # nearest extrapolation beyond grid
    )
    return interp(np.column_stack([omega_eval, heading_eval]))


def analytical_mean_drift(
    coeffs: WaveDriftCoefficients,
    wave_omega: np.ndarray,
    wave_amplitudes: np.ndarray,
    beta_wave: float,
) -> np.ndarray:
    r"""Analytical mean drift force (Fossen 2021, Eq. 8.87).

    .. math::
        \bar{F}_i = \sum_k a_k^2 \, T_i(\omega_k, \beta)
    """
    heading_eval = np.full_like(wave_omega, beta_wave % (2 * np.pi))
    a2 = wave_amplitudes**2
    T = np.stack(
        [
            _interp_coeffs(coeffs, coeffs.surge, wave_omega, heading_eval),
            _interp_coeffs(coeffs, coeffs.sway, wave_omega, heading_eval),
            _interp_coeffs(coeffs, coeffs.yaw, wave_omega, heading_eval),
        ]
    )
    return (a2[np.newaxis, :] * T).sum(axis=-1)


def pinkster_slow_drift_spectrum(
    coeffs: WaveDriftCoefficients,
    wave_omega: np.ndarray,
    S: np.ndarray,
    d_omega: float,
    beta_wave: float,
    mu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Slow-drift force spectrum (Pinkster 1980) with arithmetic-mean Newman.

    Under Newman's (1974) arithmetic-mean approximation the off-diagonal
    QTF is :math:`T^-_{ij} \approx \tfrac{1}{2}(Q_i + Q_j)`, so
    :math:`|T^-_{ij}|^2 = \tfrac{1}{4}(Q_i + Q_j)^2`.

    The one-sided slow-drift force spectrum (Pinkster 1980, Eq. 4.52)
    evaluated with this approximation gives:

    .. math::
        S_F(\mu) = 2 \int_0^\infty
            S(\omega)\,S(\omega + \mu)\,
            \bigl(Q(\omega) + Q(\omega+\mu)\bigr)^2\;d\omega

    where :math:`\mu \geq 0` is the difference frequency and
    :math:`Q(\omega) = T^-_{ii}(\omega, \beta)` is the diagonal QTF
    (drift coefficient).

    Parameters
    ----------
    coeffs : WaveDriftCoefficients
        Drift force transfer functions.
    wave_omega : NDArray
        Frequencies at which the wave spectrum is evaluated (rad/s).
    S : NDArray
        Wave spectral density at ``wave_omega`` (m²·s/rad).
    d_omega : float
        Frequency spacing used in the discretisation (rad/s).
    beta_wave : float
        Wave heading (rad).
    mu : NDArray or None
        Difference-frequency grid (rad/s).  If None, a default grid
        from 0 to half the Nyquist is used.

    Returns
    -------
    mu : NDArray
        Difference-frequency grid (rad/s).
    S_F : NDArray
        Slow-drift force spectral density, shape ``(3, len(mu))``,
        for surge, sway, yaw.
    """
    heading_eval = np.full_like(wave_omega, beta_wave % (2 * np.pi))
    Q_all = np.stack(
        [
            _interp_coeffs(coeffs, coeffs.surge, wave_omega, heading_eval),
            _interp_coeffs(coeffs, coeffs.sway, wave_omega, heading_eval),
            _interp_coeffs(coeffs, coeffs.yaw, wave_omega, heading_eval),
        ]
    )  # (3, n_freq)

    if mu is None:
        mu_max = (wave_omega[-1] - wave_omega[0]) / 2
        mu = np.linspace(0, mu_max, 500)

    S_F = np.zeros((3, len(mu)))
    for k, mu_k in enumerate(mu):
        omega_hi = wave_omega + mu_k
        valid = omega_hi <= wave_omega[-1]
        if not np.any(valid):
            continue
        w = wave_omega[valid]
        S_lo = S[valid]
        S_hi = np.interp(w + mu_k, wave_omega, S, left=0.0, right=0.0)
        for dof in range(3):
            Q_lo = Q_all[dof, valid]
            Q_hi = np.interp(w + mu_k, wave_omega, Q_all[dof], left=0.0, right=0.0)
            S_F[dof, k] = 2.0 * np.trapezoid(S_lo * S_hi * (Q_lo + Q_hi) ** 2, w)

    return mu, S_F


def pinkster_variance(
    coeffs: WaveDriftCoefficients,
    wave_omega: np.ndarray,
    S: np.ndarray,
    d_omega: float,
    beta_wave: float,
) -> np.ndarray:
    r"""Variance of Newman's slow-drift force from Pinkster's spectrum.

    .. math::
        \sigma_F^2 = \int_0^\infty S_F(\mu)\,d\mu

    This is the standard result from second-order stochastic wave
    theory (Pinkster 1980, §4.3).
    """
    mu, S_F = pinkster_slow_drift_spectrum(coeffs, wave_omega, S, d_omega, beta_wave)
    return np.trapezoid(S_F, mu, axis=-1)


# ── Wave realisation (same as OSV.init_wave_realisation) ─────────────────


def make_wave_realisation(
    Hs,
    Tp,
    n_freq=200,
    omega_range=(0.2, 2.5),
    seed=42,
    gamma=3.3,
    spacing="equal_frequency",
):
    rng = np.random.default_rng(seed)

    if spacing == "equal_energy":
        # OrcaFlex equal-energy spacing: each component carries the same energy.
        # Build a fine CDF of the spectrum energy and invert it.
        n_fine = 10000
        w_fine = np.linspace(omega_range[0], omega_range[1], n_fine)
        S_fine = jonswap_spectrum(w_fine, Hs, Tp, gamma=gamma)
        cdf = np.cumsum(S_fine)
        cdf = cdf / cdf[-1]
        # Target quantiles: midpoints of n_freq equal-probability bins
        quantiles = (np.arange(n_freq) + 0.5) / n_freq
        wave_omega = np.interp(quantiles, cdf, w_fine)
        S = jonswap_spectrum(wave_omega, Hs, Tp, gamma=gamma)
        # Each component has equal energy = m0 / n_freq
        w_fine[1] - w_fine[0]
        m0 = np.trapezoid(S_fine, w_fine)
        wave_amplitudes = np.sqrt(2.0 * m0 / n_freq) * np.ones(n_freq)
        d_omega = (omega_range[1] - omega_range[0]) / n_freq  # nominal
    else:
        d_omega = (omega_range[1] - omega_range[0]) / n_freq
        wave_omega = np.linspace(
            omega_range[0], omega_range[1], n_freq, endpoint=False
        ) + rng.uniform(0, d_omega, n_freq)
        S = jonswap_spectrum(wave_omega, Hs, Tp, gamma=gamma)
        wave_amplitudes = np.sqrt(2.0 * S * d_omega)

    wave_phases = rng.uniform(0, 2 * np.pi, n_freq)
    return wave_omega, wave_amplitudes, wave_phases, S, d_omega


def _empirical_skewness(x: np.ndarray) -> np.ndarray:
    """Skewness per column"""
    n = x.shape[0]
    m = x.mean(axis=0)
    m3 = ((x - m) ** 3).mean(axis=0)
    m2 = ((x - m) ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = np.where(m2 > 0, m3 / m2**1.5, 0.0)
    # Bias correction (Fisher)
    if n > 2:
        g1 *= np.sqrt(n * (n - 1)) / (n - 2)
    return g1


def verify_orcaflex_analytical() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]
]:
    """Verify against both the analytical mean and OrcaFlex reference for the same sea state.
    Return:
    - t: Time vector (n,)
    - F_orcaflex: Time series of drift force from OrcaFlex reference (N / Nm) over time (n, 3)
    - F_newman: Time series of drift force from Newman calculation (N / Nm) over time (n, 3)
    - statistics: Dictionary containing mean and std for analytical, OrcaFlex, and Newman results.
    """
    Hs = 3.0
    Tp = 7.0
    beta_deg = 22.5
    beta_wave = np.deg2rad(beta_deg)
    gamma = 1.0

    orcaflex_coeffs = WaveDriftCoefficients.from_npz(
        Path(__file__).parent / "orcaflex_drift_coefficients.npz"
    )
    ref_csv = Path(__file__).parent / "ref_wave_drift.csv"

    ref = np.loadtxt(ref_csv, delimiter=",", skiprows=1)
    t_ref = ref[:, 0]
    F_orcaflex = ref[:, 1:]  # (n, 3) — surge, sway, yaw in kN / kNm

    beta_wave = np.deg2rad(22.5)  # OrcaFlex reference heading

    # Build wave realisation matching OrcaFlex settings:
    # 228 components, equal-energy spacing, freq range 0.5–10 × fp, gamma=1
    wp = 2 * np.pi / Tp
    omega_lo = 0.5 * wp
    omega_hi = 10.0 * wp
    n_freq = 228
    wave_omega, wave_amplitudes, wave_phases, S, d_omega = make_wave_realisation(
        Hs,
        Tp,
        n_freq=n_freq,
        omega_range=(omega_lo, omega_hi),
        gamma=gamma,
        spacing="equal_energy",
    )

    # Compute Newman time series on the reference time grid
    t_pos = t_ref[t_ref >= 0]
    n_steps = len(t_pos)
    F_newman = np.zeros((n_steps, 3))

    print(f"  Computing Newman drift force ({n_steps} steps)...")
    for j, t in enumerate(t_pos):
        F_newman[j] = newman_drift_force(
            orcaflex_coeffs,
            wave_omega,
            wave_amplitudes,
            wave_phases,
            beta_wave,
            t,
            U=None,
        )

    F_newman_kN = F_newman * 1e-3  # N → kN

    # Reference on same time range (t >= 0)
    mask_pos = t_ref >= 0
    F_orcaflex_pos = F_orcaflex[mask_pos]

    F_mean_analytical = analytical_mean_drift(
        orcaflex_coeffs, wave_omega, wave_amplitudes, beta_wave
    )
    F_var_pinkster = pinkster_variance(
        orcaflex_coeffs, wave_omega, S, d_omega, beta_wave
    )

    # Statistics calculations
    statistics = {
        "mean": {
            "analytical": F_mean_analytical,
            "orcaflex": F_orcaflex_pos.mean(axis=0),
            "newman": F_newman_kN.mean(axis=0),
        },
        "std": {
            "analytical": np.sqrt(F_var_pinkster),
            "orcaflex": F_orcaflex_pos.std(axis=0),
            "newman": F_newman_kN.std(axis=0),
        },
        "skewness": {
            "orcaflex": _empirical_skewness(F_orcaflex_pos),
            "newman": _empirical_skewness(F_newman_kN),
        },
    }

    return t_pos, F_orcaflex_pos, F_newman_kN, statistics

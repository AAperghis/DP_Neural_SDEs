"""
Environmental loads: ocean current and wave drift forces.

Ocean current model from Fossen (2021, Ch. 8).
Wave spectrum models translated from the MSS toolbox ``waveSpectrum.m``.
Mean wave drift forces via numerical integration of drift force
coefficients over the wave energy spectrum (Fossen 2021, Eq. 8.87):

    F_drift_i = 2 * integral{ S(w) * T_i(w, beta) dw }

where S(w) is the wave energy spectrum and T_i(w, beta) is the
drift force transfer function for DOF *i* and wave heading *beta*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from thesis.full_order_dp.gnc import smtrx


# ---------------------------------------------------------------------------
# Ocean current
# ---------------------------------------------------------------------------


def ocean_current(
    Vc: float,
    betaVc: float,
    psi: float,
    nu_ang: NDArray,
) -> tuple[NDArray, NDArray]:
    """Irrotational ocean current in the body frame.

    Parameters
    ----------
    Vc : float
        Current speed (m/s).
    betaVc : float
        Current direction in NED (rad).
    psi : float
        Vessel heading (rad).
    nu_ang : NDArray
        Angular body velocities ``[p, q, r]``.

    Returns
    -------
    nu_c : NDArray
        6-DOF current velocity in body frame.
    nu_c_dot : NDArray
        Time derivative of ``nu_c`` (due to vessel rotation in current).
    """
    v_c = np.array(
        [
            Vc * np.cos(betaVc - psi),
            Vc * np.sin(betaVc - psi),
            0.0,
        ]
    )
    nu_c = np.concatenate([v_c, np.zeros(3)])
    nu_c_dot = np.concatenate([-smtrx(nu_ang) @ v_c, np.zeros(3)])
    return nu_c, nu_c_dot


# ---------------------------------------------------------------------------
# Wave spectra (translated from MSS waveSpectrum.m)
# ---------------------------------------------------------------------------


def jonswap_spectrum(
    omega: NDArray,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
) -> NDArray:
    """JONSWAP wave energy spectrum.

    Parameters
    ----------
    omega : NDArray
        Wave frequencies (rad/s).
    Hs : float
        Significant wave height (m).
    Tp : float
        Peak period (s).
    gamma : float
        Peak enhancement factor (default 3.3).

    Returns
    -------
    S : NDArray
        Spectral density values (m^2 s / rad).
    """
    wp = 2 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)

    # Build the PM base spectrum with unit Hs, then apply JONSWAP peak
    # enhancement and re-scale to match the requested Hs exactly.
    # This avoids the Goda (1999) A_gamma approximation which is only
    # accurate for gamma ≈ 2-7 and breaks down at gamma = 1.
    alpha_unit = wp**4 / 5.0  # PM alpha for Hs=1
    S_pm = (alpha_unit / omega**5) * np.exp(-1.25 * (wp / omega) ** 4)
    G = gamma ** np.exp(-0.5 * ((omega - wp) / (sigma * wp)) ** 2)
    S_raw = S_pm * G

    # Re-scale so that m0 = (Hs/4)^2
    d_omega = np.gradient(omega)
    m0_raw = np.sum(S_raw * d_omega)
    m0_target = (Hs / 4.0) ** 2
    return S_raw * (m0_target / m0_raw)


def pierson_moskowitz_spectrum(omega: NDArray, Hs: float, Tp: float) -> NDArray:
    """Pierson-Moskowitz (ITTC modified) wave spectrum."""
    wp = 2 * np.pi / Tp
    A = (5 / 16) * Hs**2 * wp**4
    B = 1.25 * wp**4
    return A / omega**5 * np.exp(-B / omega**4)


def cos_spreading(
    theta: NDArray,
    theta0: float,
    s: float = 2.0,
) -> NDArray:
    r"""Cosine-s directional spreading function.

    .. math::
        D(\theta) = \frac{\Gamma(\frac{s}{2}+1)^2}{\Gamma(\frac{s}{2}+\frac{1}{2})\,\sqrt{\pi}}
        \cos^{s}\!\left(\theta - \theta_0\right)

    Normalised so that :math:`\int_{-\pi}^{\pi} D(\theta)\,d\theta = 1`.

    From DNV-RP-C205
    3.5.8.6 Due consideration should be taken to modelling of the directional distribution since it may have a
    significant effect on wave parameters and consequently on the estimated wave loads. For sea states (Hs,Tp)
    there may be a large variation in the wave spreading parameter n.

    3.5.8.7 Typical values for extratropical wind generated sea states are n=2 to n=10. Low wind sea states
    have in general low n, while high wind sea states have higher values of n. Swell sea states should be assumed
    to be long-crested or n>10.

    3.5.8.8 Without a more detailed documentation on wave spreading for wind generated
    sea states, the most conservative value in the range 2 to 10 should be selected. When estimating extreme wave
    loads, n should not be taken lower than 10. For fatigue assessment, where low and moderate sea states are governing
    the fatigue accumulation, n should be taken as the most unfavourable value between 2 and 6.

    Parameters
    ----------
    theta : NDArray
        Directional bins (rad).
    theta0 : float
        Mean wave direction (rad).
    s : float
        Spreading parameter.  ``s=1`` gives broad spreading,
        ``s→∞`` approaches long-crested.

    Returns
    -------
    D : NDArray
        Spreading weights (1/rad), same shape as *theta*.
    """
    from scipy.special import gamma as gammafn

    C = gammafn(s / 2 + 1) / (gammafn(s / 2 + 0.5) * np.sqrt(np.pi))
    delta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    valid = np.abs(delta) <= np.pi / 2
    return np.where(valid, C * np.cos(delta) ** s, 0.0)


# ---------------------------------------------------------------------------
# Wave drift forces
# ---------------------------------------------------------------------------


@dataclass
class WaveDriftCoefficients:
    """Drift-force transfer functions T_i(omega, beta) for the OSV.

    The coefficients represent non-dimensional mean drift force amplitudes
    per unit wave amplitude squared, for each DOF (surge, sway, yaw)
    at discrete frequencies and headings.

    Each coefficient is defined as: (freq, heading) -> T_i, where T_i is the
    drift force coefficient for DOF *i*.


    Attributes
    ----------
    omega : NDArray
        Discrete frequencies at which coefficients are defined (rad/s).
    headings : NDArray
        Corresponding wave headings (rad) for the coefficients.
    surge : NDArray
        Drift force coefficient in surge (N/m^2) at each frequency and heading.
    sway : NDArray
        Drift force coefficient in sway (N/m^2) at each frequency and heading.
    yaw : NDArray
        Drift moment coefficient in yaw (Nm/m^2) at each frequency and heading.
    """

    headings: NDArray = field(default_factory=lambda: np.array([]))
    omega: NDArray = field(default_factory=lambda: np.array([]))
    surge: NDArray = field(default_factory=lambda: np.array([]))
    sway: NDArray = field(default_factory=lambda: np.array([]))
    yaw: NDArray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_npz(cls, path: str | Path) -> WaveDriftCoefficients:
        """Load drift coefficients from a ``.npz`` file.

        Expects keys ``drift_coefficients`` (6, n_freq, n_heading),
        ``frequencies``, and ``headings``.
        """
        from pathlib import Path as _Path

        data = np.load(_Path(path))
        coeffs = data["drift_coefficients"]
        headings = data["headings"]
        # Convert headings to radians if stored in degrees
        if headings.max() > 2 * np.pi:
            headings = np.deg2rad(headings)
        return cls(
            omega=data["frequencies"],
            headings=headings,
            surge=coeffs[0],
            sway=coeffs[1],
            yaw=coeffs[5],
        )


@dataclass
class ForceRAO:
    """First-order wave force RAOs (Response Amplitude Operators).

    Stores amplitude and phase per DOF at discrete frequencies and headings.
    Shape of each array: ``(n_freq, n_heading)``.

    Attributes
    ----------
    omega : NDArray
        Discrete frequencies (rad/s).
    headings : NDArray
        Headings (rad).
    amplitude : NDArray
        RAO amplitudes, shape ``(6, n_freq, n_heading)``.
    phase : NDArray
        RAO phases (rad), shape ``(6, n_freq, n_heading)``.
    """

    omega: NDArray = field(default_factory=lambda: np.array([]))
    headings: NDArray = field(default_factory=lambda: np.array([]))
    amplitude: NDArray = field(default_factory=lambda: np.array([]))
    phase: NDArray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_npz(cls, path: str | Path) -> ForceRAO:
        """Load force RAOs from a ``.npz`` file."""
        data = np.load(Path(path))
        headings = data["headings"]
        if headings.max() > 2 * np.pi:
            headings = np.deg2rad(headings)
        return cls(
            omega=data["frequencies"],
            headings=headings,
            amplitude=data["amplitude"],
            phase=data["phase"],
        )


def first_order_wave_force(
    rao: ForceRAO,
    wave_omega: NDArray,
    wave_amplitudes: NDArray,
    wave_phases: NDArray,
    wave_heading: float,
    t: float,
) -> NDArray:
    """First-order wave excitation force from RAOs.

    Computes:
        F_i(t) = sum_k  a_k * |H_i(omega_k, beta)| * cos(omega_k*t + phi_k + angle(H_i))

    Parameters
    ----------
    rao : ForceRAO
        Force RAO data.
    wave_omega : NDArray
        Wave component frequencies (rad/s), shape ``(n_freq,)``.
    wave_amplitudes : NDArray
        Wave component amplitudes (m), shape ``(n_freq,)``.
    wave_phases : NDArray
        Wave component random phases (rad), shape ``(n_freq,)``.
    wave_heading : float
        Wave heading (rad).
    t : float
        Current time (s).

    Returns
    -------
    F : NDArray
        First-order wave force for all 6 DOFs, shape ``(6,)``.
    """
    from scipy.interpolate import RegularGridInterpolator

    omega_grid = rao.omega.ravel()
    heading_grid = rao.headings.ravel()
    heading_eval = np.full_like(wave_omega, wave_heading % (2 * np.pi))

    F = np.zeros(6)
    for dof in range(6):
        interp_amp = RegularGridInterpolator(
            (omega_grid, heading_grid),
            rao.amplitude[dof],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        interp_phase = RegularGridInterpolator(
            (omega_grid, heading_grid),
            rao.phase[dof],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.column_stack([wave_omega, heading_eval])
        H_amp = interp_amp(pts)
        H_phase = interp_phase(pts)

        F[dof] = np.sum(
            wave_amplitudes * H_amp * np.cos(wave_omega * t + wave_phases + H_phase)
        )
    return F


def _encounter_correction_deep(
    omega: NDArray,
    wave_heading: float | NDArray,
    U: NDArray,
    g: float = 9.81,
) -> tuple[NDArray, NDArray, NDArray]:
    """Deep-water encounter-frequency correction (Aranha/OrcaFlex/Molin).

    Computes modified frequency, heading, and amplitude scaling factor
    for wave drift damping in deep water (Eqs. 7-9 from the OrcaFlex
    formulation).

    Parameters
    ----------
    omega : NDArray
        Absolute wave frequencies (rad/s), shape ``(n,)``.
    wave_heading : float or NDArray
        Absolute wave heading(s) (rad).  Scalar for long-crested seas,
        or shape ``(n,)`` for short-crested seas.
        Absolute wave heading (rad).
    U : NDArray
        Low-frequency vessel velocity minus current velocity at the QTF
        origin, shape ``(2,)`` — ``[u_surge, u_sway]`` in NED aligned
        with the wave direction frame.
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    A_e : NDArray
        Aranha scaling factor, shape ``(n_freq,)``.
    omega_e : NDArray
        Encounter frequency (rad/s), shape ``(n_freq,)``.
    beta_e : NDArray
        Encounter heading (rad), shape ``(n_freq,)``.
    """
    U_L = U[0] * np.cos(wave_heading) + U[1] * np.sin(wave_heading)
    U_T = -U[0] * np.sin(wave_heading) + U[1] * np.cos(wave_heading)

    A_e = 1.0 - 4.0 * omega / g * U_L
    omega_e = omega - omega**2 / g * U_L
    beta_e = wave_heading + 2.0 * omega / g * U_T

    return A_e, omega_e, beta_e


def newman_drift_force(
    coeffs: WaveDriftCoefficients,
    wave_omega: NDArray,
    wave_amplitudes: NDArray,
    wave_phases: NDArray,
    wave_heading: float,
    t: float,
    U: NDArray | None = None,
    spreading_s: float | None = None,
    spreading_dirs: NDArray | None = None,
) -> NDArray:
    """Slowly-varying drift force via factorized Newman's approximation.

    Exploits the geometric-mean form to avoid constructing the full QTF.
    When all diagonal entries for a DOF share the same sign, the double
    sum factorizes into a single weighted sum squared:

        F_sv(t) = sgn * |sum_i  a_i * sqrt(|Q_ii|) * exp(j*omega_i*t + phi_i)|^2

    This is O(N) per timestep instead of O(N^2) with a full QTF.

    When ``U`` is provided, wave drift damping is included for surge and
    sway by evaluating the diagonal QTF at encounter frequencies and
    headings with Aranha scaling (deep-water, OrcaFlex/Molin formulation).
    Yaw always uses unmodified QTF values.

    When ``spreading_s`` is provided, short-crested seas are modelled
    using a cos-2s spreading function.  Wave components are distributed
    over ``spreading_dirs`` (default: 36 bins covering ±π around the
    mean heading).  Each (frequency, direction) pair gets its own
    amplitude ``a_{ik} = a_i * sqrt(D(θ_k) * Δθ)`` and random phase.

    Parameters
    ----------
    coeffs : WaveDriftCoefficients
        Drift force coefficients with discrete frequencies and headings.
    wave_omega : NDArray
        Wave component frequencies (rad/s), shape ``(n_freq,)``.
    wave_amplitudes : NDArray
        Wave component amplitudes, shape ``(n_freq,)``.
        These are the long-crested amplitudes (from the 1-D spectrum).
    wave_phases : NDArray
        Wave component random phases, shape ``(n_freq,)``.
        For short-crested seas, unique phases per (freq, dir) are
        generated deterministically from these seeds.
    wave_heading : float
        Mean wave heading (rad).
    t : float
        Current time (s).
    U : NDArray or None
        Low-frequency vessel velocity minus current velocity,
        shape ``(2,)`` — ``[u_surge, u_sway]`` in the NED frame.
        If None, wave drift damping is not applied.
    spreading_s : float or None
        Spreading parameter *s* for cos-2s function.  If None,
        long-crested seas are used (no spreading).
    spreading_dirs : NDArray or None
        Directional bins (rad) for the spreading discretisation.
        Default (if None and spreading_s is set): 36 bins over
        [wave_heading − π, wave_heading + π).

    Returns
    -------
    F_drift : NDArray
        Slowly-varying drift force for surge, sway, yaw — shape ``(3,)``.
    """
    from scipy.interpolate import RegularGridInterpolator

    omega_grid = coeffs.omega.ravel()
    heading_grid = coeffs.headings.ravel()

    def _interp_coeffs(coeff_2d, omega_eval, heading_eval):
        interp = RegularGridInterpolator(
            (omega_grid, heading_grid),
            coeff_2d,
            method="linear",
            bounds_error=False,
            fill_value=None,  # nearest extrapolation beyond grid
        )
        pts = np.column_stack([omega_eval, heading_eval])
        return interp(pts)

    # --- Expand to (freq, dir) grid if spreading is requested ----------
    if spreading_s is not None:
        n_freq = len(wave_omega)
        if spreading_dirs is None:
            n_dir = 36
            spreading_dirs = np.linspace(
                wave_heading - np.pi,
                wave_heading + np.pi,
                n_dir,
                endpoint=False,
            )
        else:
            n_dir = len(spreading_dirs)

        d_theta = 2 * np.pi / n_dir
        D = cos_spreading(spreading_dirs, wave_heading, spreading_s)

        # Wrap directions to [0, 2π) so they match the heading grid
        spreading_dirs_wrapped = spreading_dirs % (2 * np.pi)

        # (n_freq, n_dir) grids
        omega_2d = np.repeat(wave_omega, n_dir)
        dir_2d = np.tile(spreading_dirs_wrapped, n_freq)
        amp_2d = np.repeat(wave_amplitudes, n_dir) * np.sqrt(
            np.tile(D * d_theta, n_freq)
        )
        # Independent random phases per (freq, dir) — seeded from
        # the per-frequency phases to stay deterministic for a given seed.
        rng = np.random.default_rng(seed=int(np.abs(wave_phases.sum()) * 1e6) % (2**31))
        phases_2d = rng.uniform(0, 2 * np.pi, n_freq * n_dir)

        wave_omega_flat = omega_2d
        wave_amplitudes_flat = amp_2d
        wave_phases_flat = phases_2d
        headings_flat = dir_2d
    else:
        wave_omega_flat = wave_omega
        wave_amplitudes_flat = wave_amplitudes
        wave_phases_flat = wave_phases
        headings_flat = np.full_like(wave_omega, wave_heading)

    # --- Encounter correction for surge & sway -------------------------
    if U is not None:
        A_e, omega_e, beta_e = _encounter_correction_deep(
            wave_omega_flat, headings_flat, U
        )
        omega_e = np.maximum(omega_e, 1e-6)
    else:
        A_e = np.ones_like(wave_omega_flat)
        omega_e = wave_omega_flat
        beta_e = headings_flat

    Q_surge = A_e * _interp_coeffs(coeffs.surge, omega_e, beta_e)
    Q_sway = A_e * _interp_coeffs(coeffs.sway, omega_e, beta_e)
    # Yaw always uses unmodified QTF
    Q_yaw = _interp_coeffs(coeffs.yaw, wave_omega_flat, headings_flat)

    Q_diag = np.stack([Q_surge, Q_sway, Q_yaw])  # (3, n_components)

    # Newman's arithmetic-mean approximation (Newman 1974):
    #   T^-_{ij} ≈ ½(Q_i + Q_j)
    #
    # The double sum factorizes into two O(N) complex sums:
    #   F_sv(t) = Re[W(t) · Z*(t)]
    # where
    #   Z(t) = Σ_k  a_k · exp(j·θ_k)          (wave phasor sum)
    #   W(t) = Σ_k  Q_k · a_k · exp(j·θ_k)    (QTF-weighted phasor sum)
    #
    # This preserves the correct mean  <F> = Σ_k a_k² Q_k
    # and the full slow-drift variance
    phasor = wave_amplitudes_flat * np.exp(
        1j * (wave_omega_flat * t + wave_phases_flat)
    )  # (n_components,)

    Z = phasor.sum()  # complex scalar
    W = (Q_diag * phasor[np.newaxis, :]).sum(axis=-1)  # (3,) complex

    F_drift = np.real(W * np.conj(Z))
    return F_drift

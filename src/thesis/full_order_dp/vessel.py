"""
Offshore Supply Vessel (OSV) model — 6-DOF equations of motion.

Translated from ``osv.m`` / ``SIMosv.m`` (Thor I. Fossen, 2024)
in the MSS toolbox.  The equations follow Fossen (2021, Eqs. 6.111-6.116):

    eta_dot = J(eta) * nu
    nu_dot  = nu_c_dot + Minv * (tau_thr + tau_drag + tau_crossflow
              + tau_wave_drift - (CRB + CA + D) * nu_r - G * eta)

References:
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
    Motion Control. 2nd Edition, Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from thesis.full_order_dp.gnc import eulerang
from thesis.full_order_dp.matrices import gmtrx, rbody, m2c, dmtrx
from thesis.full_order_dp.hydrodynamics import force_surge_damping, cross_flow_drag
from thesis.full_order_dp.thruster import thruster_config
from thesis.full_order_dp.environment import (
    ocean_current,
    newman_drift_force,
    first_order_wave_force,
    jonswap_spectrum,
    WaveDriftCoefficients,
    ForceRAO,
)


@dataclass
class OSVParams:
    """Geometric, inertial, and hydrodynamic parameters for the OSV."""

    L: float = 83.0
    B: float = 18.0
    T: float = 5.0
    rho: float = 1025.0
    Cb: float = 0.65
    S: float = 0.0

    K_max: NDArray = field(
        default_factory=lambda: np.array([300e3, 300e3, 420e3, 655e3])
    )
    n_max: NDArray = field(
        default_factory=lambda: np.array([140.0, 140.0, 150.0, 200.0])
    )
    K_thr: NDArray = field(default_factory=lambda: np.eye(4))
    l_x: list[float] = field(default_factory=lambda: [37.0, 35.0, -41.5, -41.5])
    l_y: list[float] = field(default_factory=lambda: [0.0, 0.0, 7.0, -7.0])

    thrust_max: float = 0.0
    U_max: float = 7.7

    M: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    Minv: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    D: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    G: NDArray = field(default_factory=lambda: np.zeros((6, 6)))

    m: float = 0.0
    nabla: float = 0.0
    r_bg: NDArray = field(default_factory=lambda: np.array([-4.5, 0.0, -1.2]))

    R44: float = 0.0
    R55: float = 0.0
    R66: float = 0.0

    T1: float = 100.0
    T2: float = 100.0
    T6: float = 1.0
    zeta4: float = 0.15
    zeta5: float = 0.3

    MA: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    MRB: NDArray = field(default_factory=lambda: np.zeros((6, 6)))


class OSV:
    """6-DOF Offshore Supply Vessel dynamics.

    Manages parameter initialisation and evaluates the equations of motion.
    Wave drift forces are included when ``Hs > 0``.
    """

    def __init__(
        self,
        drift_coeffs: WaveDriftCoefficients | None = None,
        force_rao: ForceRAO | None = None,
        spreading_s: float | None = None,
    ) -> None:
        self.params = self._init_params()
        self.drift_coeffs = drift_coeffs
        self.force_rao = force_rao
        self.spreading_s = spreading_s

        # Wave realisation — populated by init_wave_realisation()
        self.wave_omega: NDArray = np.array([])
        self.wave_amplitudes: NDArray = np.array([])
        self.wave_phases: NDArray = np.array([])

        # LF velocity filter state (NED surge/sway)
        self.nu_LF: NDArray = np.zeros(2)

        # Last computed forces (for logging)
        self.tau_wave: NDArray = np.zeros(6)
        self.tau_wave1: NDArray = np.zeros(6)
        self.tau_thr: NDArray = np.zeros(6)

    def init_wave_realisation(
        self,
        Hs: float,
        Tp: float,
        n_freq: int = 200,
        omega_range: tuple[float, float] = (0.2, 2.5),
        seed: int = 0,
    ) -> None:
        """Pre-generate the wave component amplitudes and random phases.

        Frequencies are randomly perturbed within each bin to avoid the
        artificial periodicity that arises from equal spacing
        (repeat period = 2π/Δω).

        Must be called once before the simulation loop when ``Hs > 0``.
        """
        rng = np.random.default_rng(seed)
        d_omega = (omega_range[1] - omega_range[0]) / n_freq
        # Jitter each frequency uniformly within its bin
        self.wave_omega = np.linspace(
            omega_range[0], omega_range[1], n_freq, endpoint=False
        ) + rng.uniform(0, d_omega, n_freq)
        S = jonswap_spectrum(self.wave_omega, Hs, Tp)
        self.wave_amplitudes = np.sqrt(2.0 * S * d_omega)
        self.wave_phases = rng.uniform(0, 2 * np.pi, n_freq)

    @staticmethod
    def _init_params() -> OSVParams:
        v = OSVParams()
        v.S = v.L * v.B + 2 * v.T * v.B
        v.K_thr = np.diag(v.K_max / v.n_max**2)
        v.l_x = [37.0, 35.0, -v.L / 2, -v.L / 2]
        v.nabla = v.Cb * v.L * v.B * v.T
        v.m = v.rho * v.nabla
        v.thrust_max = v.K_max[2] + v.K_max[3]

        r_bg = v.r_bg
        Cw = 0.8
        Awp = Cw * v.B * v.L
        KB = (1 / 3) * (5 * v.T / 2 - v.nabla / Awp)
        r_bb = np.array([-4.5, 0.0, v.T - KB])
        BG = r_bb[2] - r_bg[2]
        k_ms = (6 * Cw**3) / ((1 + Cw) * (1 + 2 * Cw))
        I_T = k_ms * (v.B**3 * v.L) / 12
        I_L = 0.7 * (v.L**3 * v.B) / 12
        BM_T = I_T / v.nabla
        BM_L = I_L / v.nabla
        GM_T = BM_T - BG
        GM_L = BM_L - BG
        LCF = -0.5

        v.G = gmtrx(v.nabla, Awp, GM_T, GM_L, LCF, np.zeros(3))

        v.R44 = 0.35 * v.B
        v.R55 = 0.25 * v.L
        v.R66 = 0.25 * v.L
        v.MRB, _ = rbody(v.m, v.R44, v.R55, v.R66, np.zeros(3), r_bg)

        v.MA = 1e9 * np.array(
            [
                [0.0006, 0, 0, 0, 0, 0],
                [0, 0.0020, 0, 0.0031, 0, -0.0091],
                [0, 0, 0.0083, 0, 0.0907, 0],
                [0, 0.0031, 0, 0.0748, 0, -0.1127],
                [0, 0, 0.0907, 0, 3.9875, 0],
                [0, -0.0091, 0, -0.1127, 0, 1.2416],
            ]
        )

        v.M = v.MRB + v.MA
        v.Minv = np.linalg.inv(v.M)
        v.D = dmtrx(
            np.array([v.T1, v.T2, v.T6]),
            np.array([v.zeta4, v.zeta5]),
            v.MRB,
            v.MA,
            v.G,
        )
        return v

    # ------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: NDArray,
        ui: NDArray,
        Vc: float = 0.0,
        betaVc: float = 0.0,
        t: float = 0.0,
        beta_wave: float = 0.0,
    ) -> NDArray:
        """Evaluate the 6-DOF equations of motion.

        Args:
            x: State vector ``[nu(6), eta(6)]``.
            ui: Control input ``[n(4), alpha(2)]``.
            Vc: Ocean current speed (m/s).
            betaVc: Current direction in NED (rad).
            t: Current simulation time (s). Used for wave drift forces.
            beta_wave: Wave direction in NED (rad).

        Returns:
            Time derivative ``[nu_dot(6), eta_dot(6)]``.
        """
        v = self.params
        nu, eta = x[:6], x[6:]

        # Ocean current
        nu_c, nu_c_dot = ocean_current(Vc, betaVc, eta[5], nu[3:6])
        nu_r = nu - nu_c

        # Coriolis matrices
        _, CRB = rbody(v.m, v.R44, v.R55, v.R66, nu[3:6], v.r_bg)
        CA = m2c(v.MA, nu)  # NOTE: uses total velocity, not nu_r

        # Surge damping (replaces D[0,0])
        X, _, _ = force_surge_damping(
            nu_r[0], v.m, v.S, v.L, v.T1, v.rho, v.U_max, v.thrust_max
        )
        tau_drag = np.zeros(6)
        tau_drag[0] = X
        D = v.D.copy()
        D[0, 0] = 0.0

        # Cross-flow drag
        tau_crossflow = cross_flow_drag(v.L, v.B, v.T, nu_r)

        # Wave drift forces (Newman's approximation)
        if self.drift_coeffs is not None and len(self.wave_omega) > 0:
            # LF velocity for wave drift damping (NED)
            U_drift = self.nu_LF if np.any(self.nu_LF != 0.0) else None
            drift_3dof = newman_drift_force(
                self.drift_coeffs,
                self.wave_omega,
                self.wave_amplitudes,
                self.wave_phases,
                beta_wave,
                t,
                U=U_drift,
                spreading_s=self.spreading_s,
            )
            tau_wave = np.array(
                [drift_3dof[0], drift_3dof[1], 0.0, 0.0, 0.0, drift_3dof[2]]
            )
        else:
            tau_wave = np.zeros(6)

        self.tau_wave = tau_wave

        # First-order wave forces
        if self.force_rao is not None and len(self.wave_omega) > 0:
            tau_wave1 = first_order_wave_force(
                self.force_rao,
                self.wave_omega,
                self.wave_amplitudes,
                self.wave_phases,
                beta_wave,
                t,
            )
        else:
            tau_wave1 = np.zeros(6)

        self.tau_wave1 = tau_wave1

        # Thrust
        u_thr = np.abs(ui[:4]) * ui[:4]
        alpha = ui[4:6]
        T_thr = thruster_config(["T", "T", alpha[0], alpha[1]], v.l_x, v.l_y)
        tau_3dof = T_thr @ v.K_thr @ u_thr
        tau_thr = np.array([tau_3dof[0], tau_3dof[1], 0, 0, 0, tau_3dof[2]])
        self.tau_thr = tau_thr

        # Kinematics
        J = eulerang(eta[3], eta[4], eta[5])
        eta_dot = J @ nu
        nu_dot = nu_c_dot + v.Minv @ (
            tau_thr
            + tau_drag
            + tau_crossflow
            + tau_wave
            + tau_wave1
            - (CRB + CA + D) @ nu_r
            - v.G @ eta
        )
        return np.concatenate([nu_dot, eta_dot])

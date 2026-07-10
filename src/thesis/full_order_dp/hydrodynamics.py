"""
Hydrodynamic force models for marine vessels.

Added-mass approximations, drag coefficients, surge damping,
and cross-flow drag via strip theory (Fossen 2021, Ch. 6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def added_mass_surge(m: float, L: float, rho: float = 1025.0) -> float:
    """Approximate surge added mass (Söding 1982)."""
    nabla = m / rho
    return 2.7 * rho * nabla ** (5 / 3) / L**2


def hoerner(B: float, T: float) -> float:
    """2-D Hoerner cross-flow form coefficient."""
    CD_DATA = np.array(
        [
            [0.0108623, 1.96608],
            [0.176606, 1.96573],
            [0.353025, 1.89756],
            [0.451863, 1.78718],
            [0.472838, 1.58374],
            [0.492877, 1.27862],
            [0.493252, 1.21082],
            [0.558473, 1.08356],
            [0.646401, 0.998631],
            [0.833589, 0.87959],
            [0.988002, 0.828415],
            [1.30807, 0.759941],
            [1.63918, 0.691442],
            [1.85998, 0.657076],
            [2.31288, 0.630693],
            [2.59998, 0.596186],
            [3.00877, 0.586846],
            [3.45075, 0.585909],
            [3.7379, 0.559877],
            [4.00309, 0.559315],
        ]
    )
    ratio = B / (2 * T)
    if ratio <= CD_DATA[-1, 0]:
        return float(np.interp(ratio, CD_DATA[:, 0], CD_DATA[:, 1]))
    return 0.559315


def force_surge_damping(
    u_r: float,
    m: float,
    S: float,
    L: float,
    T1: float,
    rho: float,
    u_max: float,
    thrust_max: float | None = None,
) -> tuple[float, float, float]:
    """Surge damping force X, quadratic coeff Xuu, linear coeff Xu."""
    u_cross = 2.0
    Xudot = -added_mass_surge(m, L, rho)
    Xu = -(m - Xudot) / T1
    if thrust_max is not None:
        Xuu = -thrust_max / u_max**2
    else:
        nu_kin, k, eps = 1e-6, 0.1, 1e-10
        Rn = (L / nu_kin) * abs(u_r)
        Cf = 0.075 / (np.log10(Rn + eps) - 2) ** 2
        Xuu = -0.5 * rho * S * (1 + k) * Cf
    sigma = 1 - np.tanh(u_r / u_cross)
    X = sigma * Xu * u_r + (1 - sigma) * Xuu * abs(u_r) * u_r
    return X, Xuu, Xu


def cross_flow_drag(L: float, B: float, T: float, nu_r: NDArray) -> NDArray:
    """Cross-flow drag via strip theory (Hoerner model)."""
    rho = 1025.0
    dx = L / 20
    Cd_2D = hoerner(B, T)
    Yh = Zh = Mh = Nh = 0.0
    for xL in np.arange(-L / 2, L / 2 + dx / 2, dx):
        v_r, w_r = nu_r[1], nu_r[2]
        q, r = nu_r[4], nu_r[5]
        U_h = abs(v_r + xL * r) * (v_r + xL * r)
        U_v = abs(w_r + xL * q) * (w_r + xL * q)
        Yh -= 0.5 * rho * T * Cd_2D * U_h * dx
        Zh -= 0.5 * rho * T * Cd_2D * U_v * dx
        Mh -= 0.5 * rho * T * Cd_2D * xL * U_v * dx
        Nh -= 0.5 * rho * T * Cd_2D * xL * U_h * dx
    return np.array([0, Yh, Zh, 0, Mh, Nh])

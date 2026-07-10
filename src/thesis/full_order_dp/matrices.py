"""
System matrix builders for 6-DOF vessel models.

Hydrostatic restoring, rigid-body inertia, Coriolis-centripetal,
and linear damping matrices (Fossen 2021, Ch. 3-6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from thesis.full_order_dp.gnc import smtrx, hmtrx


def gmtrx(
    nabla: float,
    A_wp: float,
    GMT: float,
    GML: float,
    LCF: float,
    r_bp: NDArray,
) -> NDArray:
    """6x6 hydrostatic restoring matrix G about point P."""
    rho, g = 1025.0, 9.81
    r_bf = np.array([LCF, 0.0, 0.0])
    G_CF = np.diag(
        [0, 0, rho * g * A_wp, rho * g * nabla * GMT, rho * g * nabla * GML, 0]
    )
    Hf = hmtrx(r_bf)
    G_CO = Hf.T @ G_CF @ Hf
    Hp = hmtrx(r_bp)
    return Hp.T @ G_CO @ Hp


def rbody(
    m: float,
    R44: float,
    R55: float,
    R66: float,
    nu2: NDArray,
    r_bp: NDArray,
) -> tuple[NDArray, NDArray]:
    """Rigid-body mass MRB and Coriolis CRB matrices."""
    I3 = np.eye(3)
    O3 = np.zeros((3, 3))
    Ig = m * np.diag([R44**2, R55**2, R66**2])
    MRB_CG = np.block([[m * I3, O3], [O3, Ig]])
    CRB_CG = np.block(
        [
            [m * smtrx(nu2), O3],
            [O3, -smtrx(Ig @ nu2)],
        ]
    )
    H = hmtrx(r_bp)
    return H.T @ MRB_CG @ H, H.T @ CRB_CG @ H


def m2c(M: NDArray, nu: NDArray) -> NDArray:
    """Coriolis-centripetal matrix C(nu) from inertia matrix M."""
    M = 0.5 * (M + M.T)
    if len(nu) == 6:
        M11, M12 = M[:3, :3], M[:3, 3:]
        M22 = M[3:, 3:]
        nu1, nu2 = nu[:3], nu[3:]
        nu1_dot = M11 @ nu1 + M12 @ nu2
        nu2_dot = M12.T @ nu1 + M22 @ nu2
        C = np.zeros((6, 6))
        C[:3, 3:] = -smtrx(nu1_dot)
        C[3:, :3] = -smtrx(nu1_dot)
        C[3:, 3:] = -smtrx(nu2_dot)
        return C
    # 3-DOF horizontal plane
    return np.array(
        [
            [0, 0, -M[1, 1] * nu[1] - M[1, 2] * nu[2]],
            [0, 0, M[0, 0] * nu[0]],
            [M[1, 1] * nu[1] + M[1, 2] * nu[2], -M[0, 0] * nu[0], 0],
        ]
    )


def dmtrx(
    T_126: NDArray,
    zeta_45: NDArray,
    MRB: NDArray,
    MA: NDArray,
    G: NDArray,
) -> NDArray:
    """6x6 linear damping matrix for a surface craft."""
    M = MRB + MA
    T1, T2, T6 = T_126
    zeta4, zeta5 = zeta_45
    zeta3 = 0.2
    w3 = np.sqrt(G[2, 2] / M[2, 2])
    w4 = np.sqrt(G[3, 3] / M[3, 3])
    w5 = np.sqrt(G[4, 4] / M[4, 4])
    return np.diag(
        [
            M[0, 0] / T1,
            M[1, 1] / T2,
            M[2, 2] * 2 * zeta3 * w3,
            M[3, 3] * 2 * zeta4 * w4,
            M[4, 4] * 2 * zeta5 * w5,
            M[5, 5] / T6,
        ]
    )

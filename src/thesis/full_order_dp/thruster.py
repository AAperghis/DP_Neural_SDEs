"""
Thruster configuration and control allocation.

Supports tunnel thrusters, main propellers, and azimuth thrusters.
Includes both pseudoinverse and constrained (SQP) allocation methods.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, NonlinearConstraint


def thruster_config(
    alpha: list[str | float],
    l_x: list[float],
    l_y: list[float],
) -> NDArray:
    """Thruster configuration matrix T_thr (3 x n_thrusters).

    Parameters
    ----------
    alpha : list
        Thruster types: ``'T'`` (tunnel), ``'M'`` (main propeller),
        or a float (azimuth angle in rad).
    l_x, l_y : list
        Longitudinal and lateral positions of each thruster.
    """
    n = len(alpha)
    T = np.zeros((3, n))
    for i in range(n):
        a = alpha[i]
        if a == "T":
            T[:, i] = [0, 1, l_x[i]]
        elif a == "M":
            T[:, i] = [1, 0, -l_y[i]]
        else:
            az = float(a)
            T[:, i] = [
                np.cos(az),
                np.sin(az),
                l_x[i] * np.sin(az) - l_y[i] * np.cos(az),
            ]
    return T


def alloc_pseudoinverse(
    K: NDArray,
    T: NDArray,
    W: NDArray,
    tau: NDArray,
) -> NDArray:
    """Unconstrained control allocation using weighted pseudoinverse."""
    Winv = np.diag(1.0 / np.diag(W))
    Kinv = np.diag(1.0 / np.diag(K))
    return Kinv @ Winv @ T.T @ np.linalg.solve(T @ Winv @ T.T, tau)


def optimal_alloc(
    tau: NDArray,
    lb: NDArray,
    ub: NDArray,
    alpha_old: NDArray,
    u_old: NDArray,
    l_x: list[float],
    l_y: list[float],
    K_thr: NDArray,
    n_max: NDArray,
    h: float,
    x0: NDArray | None = None,
) -> tuple[NDArray, NDArray, float]:
    """Constrained control allocation matching MATLAB fmincon SQP.

    Returns ``(alpha_opt, u_opt, slack_norm)``.
    """
    max_rate_alpha = 0.3
    max_rate_u = 0.1
    w1, w2, w3, w4 = 1.0, 100.0, 1.0, 0.1

    # Pre-extract diagonal for fast constraint/Jacobian evaluation
    k_diag = np.diag(K_thr)
    eye3 = np.eye(3)

    # --- Objective and analytical gradient ---
    def objective(x):
        alpha, u, s = x[:2], x[2:6], x[6:9]
        return (
            w1 * np.dot(u, u)
            + w2 * np.dot(s, s)
            + w3 * np.dot(alpha - alpha_old, alpha - alpha_old)
            + w4 * np.dot(u - u_old, u - u_old)
        )

    def objective_grad(x):
        alpha, u, s = x[:2], x[2:6], x[6:9]
        g = np.empty(9)
        g[:2] = 2.0 * w3 * (alpha - alpha_old)
        g[2:6] = 2.0 * w1 * u + 2.0 * w4 * (u - u_old)
        g[6:9] = 2.0 * w2 * s
        return g

    # --- Equality constraint and analytical Jacobian ---
    def eq_constraint(x):
        alpha, u, s = x[:2], x[2:6], x[6:9]
        T_a = thruster_config(["T", "T", alpha[0], alpha[1]], l_x, l_y)
        return T_a @ K_thr @ u - tau + s

    def eq_constraint_jac(x):
        alpha, u = x[:2], x[2:6]
        T_a = thruster_config(["T", "T", alpha[0], alpha[1]], l_x, l_y)
        jac = np.zeros((3, 9))

        a0 = alpha[0]
        dT0 = np.array(
            [
                -np.sin(a0),
                np.cos(a0),
                l_x[2] * np.cos(a0) + l_y[2] * np.sin(a0),
            ]
        )
        jac[:, 0] = dT0 * k_diag[2] * u[2]

        a1 = alpha[1]
        dT1 = np.array(
            [
                -np.sin(a1),
                np.cos(a1),
                l_x[3] * np.cos(a1) + l_y[3] * np.sin(a1),
            ]
        )
        jac[:, 1] = dT1 * k_diag[3] * u[3]

        for j in range(4):
            jac[:, 2 + j] = T_a[:, j] * k_diag[j]

        jac[:, 6:9] = eye3
        return jac

    # --- Rate limits folded into variable bounds ---
    da_max = max_rate_alpha * h
    du_max = max_rate_u * h

    # Clamp old values to amplitude bounds before computing rate bounds
    alpha_clamped = np.clip(alpha_old, lb[:2], ub[:2])
    u_clamped = np.clip(u_old, lb[2:6], ub[2:6])

    lb_rate = np.concatenate(
        [
            np.maximum(lb[:2], alpha_clamped - da_max),
            np.maximum(lb[2:6], u_clamped - du_max),
            lb[6:],
        ]
    )
    ub_rate = np.concatenate(
        [
            np.minimum(ub[:2], alpha_clamped + da_max),
            np.minimum(ub[2:6], u_clamped + du_max),
            ub[6:],
        ]
    )

    # Fixed initial guess (matching MSS fmincon)
    x0 = np.array([np.deg2rad(-28), np.deg2rad(28), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 = np.clip(x0, lb_rate, ub_rate)

    result = minimize(
        objective,
        x0,
        method="trust-constr",
        jac=objective_grad,
        bounds=list(zip(lb_rate, ub_rate)),
        constraints=[
            NonlinearConstraint(eq_constraint, 0.0, 0.0, jac=eq_constraint_jac),
        ],
        options={"maxiter": 50, "gtol": 1e-10},
    )

    xo = np.clip(result.x, lb_rate, ub_rate)
    return xo[:2], xo[2:6], np.linalg.norm(xo[6:9])

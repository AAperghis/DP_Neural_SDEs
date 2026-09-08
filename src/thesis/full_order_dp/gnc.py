"""
Low-level GNC helper functions for marine vessel simulation.

Coordinate transformations, rotation matrices, and kinematic utilities
translated from the MSS toolbox (Fossen 2021).

Reference:
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
    Motion Control. 2nd Edition, Wiley.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def smtrx(a: NDArray) -> NDArray:
    """3x3 skew-symmetric matrix S(a) such that a x b = S(a) @ b."""
    return np.array(
        [
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0],
        ]
    )


def hmtrx(r: NDArray) -> NDArray:
    """6x6 coordinate-shift matrix H(r). Property: inv(H(r)) = H(-r)."""
    H = np.eye(6)
    H[:3, 3:] = smtrx(r).T
    return H


def Rzyx(phi: float, theta: float, psi: float) -> NDArray:
    """Euler angle rotation matrix R in SO(3), zyx convention."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + sth * spsi * cphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )


def Tzyx(phi: float, theta: float) -> NDArray:
    """Euler angle attitude transformation matrix T, zyx convention."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, sphi * sth / cth, cphi * sth / cth],
            [0, cphi, -sphi],
            [0, sphi / cth, cphi / cth],
        ]
    )


def eulerang(phi: float, theta: float, psi: float) -> NDArray:
    """6x6 Euler angle transformation J = diag(Rzyx, Tzyx)."""
    J = np.zeros((6, 6))
    J[:3, :3] = Rzyx(phi, theta, psi)
    J[3:, 3:] = Tzyx(phi, theta)
    return J


def ssa(angle: float) -> float:
    """Smallest signed angle in [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rk4(f, h: float, x: NDArray, *args, t_idx: int | None = None) -> NDArray:
    """Classic 4th-order Runge-Kutta step.

    Args:
        t_idx: Index into *args* that carries the current time.  When set
            the integrator evaluates k2/k3 at t + h/2 and k4 at t + h.
    """
    if t_idx is None:
        k1 = f(x, *args)
        k2 = f(x + 0.5 * h * k1, *args)
        k3 = f(x + 0.5 * h * k2, *args)
        k4 = f(x + h * k3, *args)
    else:
        args_half = list(args)
        args_full = list(args)
        args_half[t_idx] = args[t_idx] + 0.5 * h
        args_full[t_idx] = args[t_idx] + h
        k1 = f(x, *args)
        k2 = f(x + 0.5 * h * k1, *args_half)
        k3 = f(x + 0.5 * h * k2, *args_half)
        k4 = f(x + h * k3, *args_full)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

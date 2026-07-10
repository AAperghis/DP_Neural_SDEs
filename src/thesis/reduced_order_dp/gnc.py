#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNC functions.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley.
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
import math

# ------------------------------------------------------------------------------


def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    return angle


# ------------------------------------------------------------------------------


def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min

    return x


# ------------------------------------------------------------------------------


def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.array(
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

    return R


# ------------------------------------------------------------------------------


def Tzyx(phi, theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)

    try:
        T = np.array(
            [
                [1, sphi * sth / cth, cphi * sth / cth],
                [0, cphi, -sphi],
                [0, sphi / cth, cphi / cth],
            ]
        )

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return T


# ------------------------------------------------------------------------------


def attitudeEuler(eta, nu, sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized
    position/Euler angles eta[k+1]
    """

    p_dot = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3])
    v_dot = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6])

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta

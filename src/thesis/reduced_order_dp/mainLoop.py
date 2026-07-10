#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""

from concurrent import futures
from pathlib import Path
import time
import numpy as np
from thesis.shared.data_handling import (
    ParquetMetadata,
    finalize_df,
    make_df,
    save_df_to_parquet,
    update_df,
)
from thesis.reduced_order_dp.ornstein_uhlenbeck import (
    ou_generate_uniform,
    resample_from_base,
)
from thesis.reduced_order_dp.supply import SupplyVessel
from thesis.reduced_order_dp.gnc import attitudeEuler


MODEL = "ToyDPModel"
VERSION = "1.0"


def _match_steps(f_ext: np.ndarray, N: int) -> np.ndarray:
    """Ensure external force array has exactly N+1 rows (2-D) or elements (1-D)."""
    needed = N + 1
    if f_ext.ndim == 1:
        if f_ext.shape[0] < needed:
            f_ext = np.concatenate([f_ext, np.full(needed - f_ext.shape[0], f_ext[-1])])
        elif f_ext.shape[0] > needed:
            f_ext = f_ext[:needed]
    else:
        if f_ext.shape[0] < needed:
            pad = np.repeat(f_ext[-1:, :], needed - f_ext.shape[0], axis=0)
            f_ext = np.vstack((f_ext, pad))
        elif f_ext.shape[0] > needed:
            f_ext = f_ext[:needed, :]
    return f_ext


def generate_directional_magnitude(
    runtime: float,
    sampleTime: float,
    noise_dt: float,
    mu_mag: float,
    sigma_mag: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 1-D OU magnitude time series (one value per sample step)."""
    base = ou_generate_uniform(
        int(runtime // noise_dt),
        noise_dt,
        mu=np.array([mu_mag]),
        sigma=np.array([sigma_mag]),
        rng=rng,
    )
    return resample_from_base(base, noise_dt, sampleTime, runtime)[:, 0]


def directional_body_force(
    mag: float,
    beta: float,
    psi: float,
    x_cp: float,
    x_cog: float,
    L_underwater: float,
) -> np.ndarray:
    """
    Convert world-fixed force magnitude + direction into body-frame [Fx, Fy, Mz].

    DNV yaw moment:  Mz = Fy * ((x_cp - x_cog) + (0.05 - 0.14*|dir|/pi) * Luw)
    where dir = beta - psi  (relative direction in body frame).
    """
    rel_dir = beta - psi
    fx = mag * np.cos(rel_dir)
    fy = mag * np.sin(rel_dir)
    lever = (x_cp - x_cog) + (0.05 - 0.14 * abs(rel_dir) / np.pi) * L_underwater
    nz = fy * lever
    return np.array([fx, fy, nz])


###############################################################################
# Function printVehicleinfo(vehicle)
###############################################################################
def printInfo(vehicle, sampleTime, N):
    """
    Function to print vessel and simulation paramters

    args:
        vehicle
        sampleTime
        N
    """
    print(
        "---------------------------------------------------------------------------------------"
    )
    print("%s" % (vehicle.name))
    print("Length: %s m" % (vehicle.L))
    print("%s" % (vehicle.controlDescription))
    print("Sampling frequency: %s Hz" % round(1 / sampleTime))
    print("Simulation time: %s seconds" % round(N * sampleTime))
    print(
        "---------------------------------------------------------------------------------------"
    )


def simulate(
    N: int,
    sampleTime: float,
    vessel: SupplyVessel,
    f_ext: np.ndarray,
    meta: ParquetMetadata,
    forcing_mode: str = "per_dof_ou",
    beta: float = 0.0,
    x_cp: float = 0.0,
) -> None:
    t = 0  # initial simulation time

    # Initial state vectors
    eta = vessel.eta  # position/attitude, user editable
    nu = vessel.nu  # velocity, defined by vehicle class
    u_actual = vessel.u_actual  # actual inputs, defined by vehicle class

    df = make_df(N)

    # Main simulation loop
    for i in range(0, N + 1):
        t = i * sampleTime  # simulation time
        if t % 60 == 0:
            print(f"time: {t}s")

        # Compute body-frame external force for this step
        if forcing_mode == "directional_ou":
            f_ext_i = directional_body_force(
                mag=f_ext[i],
                beta=beta,
                psi=eta[5],
                x_cp=x_cp,
                x_cog=vessel.x_cog,
                L_underwater=vessel.L_underwater,
            )
        else:
            f_ext_i = f_ext[i]

        u_control = vessel.DPcontrol(eta, nu, sampleTime)

        tau_control = vessel.B @ (np.abs(u_control) * u_control)
        tau_actual = vessel.B @ (np.abs(u_actual) * u_actual)

        update_df(
            df, i, t, eta, nu, tau_control, tau_actual, f_ext_i, vessel.gains, u_actual
        )

        # Propagate vehicle and attitude dynamics
        nu, u_actual = vessel.dynamics(
            eta, nu, u_actual, u_control, sampleTime, f_external=f_ext_i
        )
        eta = attitudeEuler(eta, nu, sampleTime)

        # if t == 120:
        #     vessel.thrusterFailure(5) # Thruster failure of main propeller
    df = finalize_df(df)
    save_df_to_parquet(
        df,
        metadata=meta,
        path=Path(
            r"/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/directional_ou"
        ),
    )


def R2D(value):  # radians to degrees
    return value * 180 / np.pi


def run_sim(
    seed: int,
    runtime: float,
    mu_f: np.ndarray,
    sigma_f: np.ndarray,
    vessel: SupplyVessel,
    sampleTime: float,
    forcing_mode: str = "per_dof_ou",
    beta: float | None = None,
    x_cp: float = 0.0,
) -> None:
    rng = np.random.default_rng(seed)
    N = int(runtime // sampleTime)  # number of samples

    # OU process for external forces - independent of timestep
    noise_dt = 0.05
    if forcing_mode == "per_dof_ou":
        external_forces = ou_generate_uniform(
            int(runtime // noise_dt), noise_dt, mu=mu_f, sigma=sigma_f, rng=rng
        )
        external_forces = resample_from_base(
            external_forces, noise_dt, sampleTime, N * sampleTime
        )
    elif forcing_mode == "directional_ou":
        if beta is None:
            raise ValueError("beta must be specified for directional_ou mode")
        external_forces = generate_directional_magnitude(
            runtime=runtime,
            sampleTime=sampleTime,
            noise_dt=noise_dt,
            mu_mag=float(mu_f[0]),
            sigma_mag=float(sigma_f[0]),
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown forcing_mode='{forcing_mode}'")
    external_forces = _match_steps(external_forces, N)

    # Initial condition
    x_0, y_0 = rng.uniform(low=-3, high=3, size=2)
    y_0 = 0
    # psi_0 = rng.uniform(low=-0.15, high=0.15)
    psi_0 = 0
    pos_0 = (x_0, y_0, psi_0)
    vessel.eta[0] = x_0
    vessel.eta[1] = y_0
    vessel.eta[5] = psi_0

    u_0 = vessel.DPcontrol(vessel.eta, vessel.nu, sampleTime)
    vessel.u_actual = u_0

    start_time = time.time_ns()
    printInfo(vessel, sampleTime, N)

    meta = ParquetMetadata(
        model=MODEL,
        version=VERSION,
        timestep=sampleTime,
        end_time=runtime,
        seed=seed,
        n_steps=N,
        mean_force=list(mu_f),
        var_force=list(sigma_f),
        inital_pos=pos_0,
    )
    simulate(
        N,
        sampleTime,
        vessel,
        f_ext=external_forces,
        meta=meta,
        forcing_mode=forcing_mode,
        beta=beta if beta is not None else 0.0,
        x_cp=x_cp,
    )
    end_time = time.time_ns()
    print(f"Time taken: {(end_time - start_time) / 1e9:.3f}s")


if __name__ == "__main__":
    vessel = SupplyVessel("DPcontrol")
    runtime = 10800  # seconds
    sampleTime = 0.05  # Seconds [s]

    # Number of runs and seeds
    seeds = range(0, 50)

    # External forces
    mu_f = np.array([75e3, 0, 0])  # Mean
    sigma_f = np.array([50e3, 0, 0])  # Variance

    # Directional forcing settings
    forcing_mode = "directional_ou"
    beta = np.deg2rad(20.0)  # fixed direction in body frame
    x_cp = 0.0  # centre of pressure, body frame (m)

    start_time = time.time_ns()
    with futures.ProcessPoolExecutor() as pool:
        fs = []
        for seed in seeds:
            print(seed)
            fs.append(
                pool.submit(
                    run_sim,
                    seed=seed,
                    runtime=runtime,
                    mu_f=mu_f,
                    sigma_f=sigma_f,
                    vessel=vessel,
                    sampleTime=sampleTime,
                    forcing_mode=forcing_mode,
                    beta=beta,
                    x_cp=x_cp,
                )
            )
        for future in futures.as_completed(fs):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")

    print(f"Total time taken: {(time.time_ns() - start_time) / 1e9:.3f}s")

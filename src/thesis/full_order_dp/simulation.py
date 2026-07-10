"""
OSV dynamic positioning simulation runner.

Ties together the vessel model, PID controller, thruster allocation,
and environmental loads into a time-domain simulation.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from thesis.full_order_dp.gnc import ssa, rk4
from thesis.full_order_dp.thruster import (
    thruster_config,
    alloc_pseudoinverse,
    optimal_alloc,
)
from thesis.full_order_dp.control import PIDNonlinearMIMO
from thesis.full_order_dp.vessel import OSV
from thesis.full_order_dp.environment import WaveDriftCoefficients, ForceRAO


@dataclass
class SimConfig:
    """Configuration for the OSV DP simulation."""

    T_final: float = 250.0
    h: float = 0.05
    x_ref: float = 0.0
    y_ref: float = 0.0
    psi_ref: float = 0.0
    Vc: float = 0.5
    betaVc: float = np.deg2rad(-140)
    Hs: float = 0.0
    Tp: float = 8.0
    beta_wave: float = 0.0
    alloc_dynamic: bool = True
    spreading_s: float | None = None
    wave_seed: int = 0
    lf_filter_tau: float = 30.0
    setpoint_change: bool = False
    initial_transient: bool = False


def simulate_osv(
    cfg: SimConfig | None = None,
    drift_coeffs_path: Path | None = None,
    force_rao_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, NDArray], bool]:
    """Run the full OSV DP simulation.

    Returns a dict with keys: ``t``, ``eta``, ``nu``, ``n``, ``alpha``.
    """
    if cfg is None:
        cfg = SimConfig()

    # Load drift coefficients if available and waves are active
    drift_coeffs = None
    force_rao = None
    if cfg.Hs > 0 and drift_coeffs_path and drift_coeffs_path.exists():
        drift_coeffs = WaveDriftCoefficients.from_npz(drift_coeffs_path)
    if cfg.Hs > 0 and force_rao_path and force_rao_path.exists():
        force_rao = ForceRAO.from_npz(force_rao_path)

    vessel = OSV(
        drift_coeffs=drift_coeffs, force_rao=force_rao, spreading_s=cfg.spreading_s
    )
    v = vessel.params

    # Pre-generate wave realisation
    if cfg.Hs > 0:
        vessel.init_wave_realisation(cfg.Hs, cfg.Tp, seed=cfg.wave_seed)

    # Constant azimuth angles
    alpha0 = np.deg2rad(np.array([-28.0, 28.0]))

    # Thruster limits for allocation — match MSS SIMosv.m allocator parameters
    # (intentionally different from the vessel plant model in osv.m:
    #  allocator overestimates thruster 3 capacity, creating plant-model mismatch)
    K_max = np.diag([300e3, 300e3, 655e3, 655e3])
    n_max = np.array([140.0, 140.0, 150.0, 150.0])
    l_x = [37.0, 35.0, -42.0, -42.0]
    l_y = [0.0, 0.0, 7.0, -7.0]

    T_thr = thruster_config(["T", "T", alpha0[0], alpha0[1]], l_x, l_y)

    az_max = np.deg2rad(60)
    lb = np.array([-az_max, -az_max, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf])
    ub = np.array([az_max, az_max, 1, 1, 1, 1, np.inf, np.inf, np.inf])

    alpha_old = alpha0.copy()
    u_old = np.zeros(4)

    # Thruster dynamics: first-order lag time constants
    # Azimuth slew: T_alpha = full range / slew rate
    max_rate_alpha = 0.3
    T_alpha = az_max / max_rate_alpha  # ~3.5 s
    # Propeller RPM: 10% of n_max per second → 10 s linear ramp to full
    T_n = 10.0

    # Thruster dynamics state (initial actual values)
    n_actual = np.zeros(4)
    alpha_actual = alpha0.copy()

    # PID controller
    M = v.M
    wn = 0.1 * np.diag([1.0, 1.0, 3.0])
    zeta = 1.0 * np.diag([1.0, 1.0, 1.0])
    T_f = 30.0
    pid = PIDNonlinearMIMO()

    # Initial state
    eta = np.array([0.0, 0.0, 0.0, np.deg2rad(0), np.deg2rad(0), 0.0])
    if cfg.initial_transient:
        eta[0] = 5.0
        eta[1] = 5.0
    nu = np.zeros(6)
    x = np.concatenate([nu, eta])

    eta_ref = np.array([cfg.x_ref, cfg.y_ref, cfg.psi_ref])

    t = np.arange(0, cfg.T_final + cfg.h / 2, cfg.h)
    n_steps = len(t)

    # Arrays to store simulation results
    # Position and velocity states
    sim_eta = np.zeros(
        (n_steps, 6)
    )  # North, East, Heave, Roll, Pitch, Yaw positions (NED)
    sim_nu = np.zeros(
        (n_steps, 6)
    )  # Surge, sway, heave, roll, pitch, yaw velocities (Body fixed)

    # Thruster commands and actuals
    sim_n_cmd = np.zeros((n_steps, 4))  # Commanded propeller speeds
    sim_alpha_cmd = np.zeros((n_steps, 2))  # Commanded azimuth angles
    sim_n_actual = np.zeros(
        (n_steps, 4)
    )  # Realised propeller speeds (after first-order lag)
    sim_alpha_actual = np.zeros(
        (n_steps, 2)
    )  # Realised azimuth angles (after first-order lag)

    sim_tau_thr = np.zeros((n_steps, 6))  # Realised thruster forces/moments

    # Controller
    sim_tau_cmd = np.zeros((n_steps, 6))  # Total commanded force/moment
    sim_pid_z_int = np.zeros((n_steps, 3))  # PID integral state
    sim_pid_eta_d = np.zeros((n_steps, 3))  # Carrot position
    sim_pid_tau_p = np.zeros((n_steps, 3))  # PID proportional term
    sim_pid_tau_i = np.zeros((n_steps, 3))  # PID integral term
    sim_pid_tau_d = np.zeros((n_steps, 3))  # PID derivative term

    # Wave forcing
    sim_tau_wave = np.zeros((n_steps, 6))  # Total wave drift force
    sim_tau_wave1 = np.zeros(
        (n_steps, 6)
    )  # Wave drift force from first-order (linear) wave theory
    sim_nu_LF = np.zeros(
        (n_steps, 2)
    )  # Low-frequency (drift) velocity estimate from wave drift model

    # LF filter coefficient: exact discretisation of 1st-order lag
    lf_alpha = 1.0 - np.exp(-cfg.h / cfg.lf_filter_tau)

    # Exact discretisation factors for thruster dynamics
    _thr_n_alpha = 1.0 - np.exp(-cfg.h / T_n)
    _thr_az_alpha = 1.0 - np.exp(-cfg.h / T_alpha)

    # Seeded RNG for reproducible sensor noise
    _rng = np.random.default_rng(cfg.wave_seed + 1)

    sucess = False
    for i in range(n_steps):
        # Sensor noise disabled to match MSS reference (no noise)
        eta_m = eta.copy()

        # Setpoint change at t > 50 s
        if t[i] == 50 and cfg.setpoint_change:
            print(f"Time {t[i]:.1f}s: changing setpoint")
            eta_ref = np.array([cfg.x_ref, cfg.y_ref, np.deg2rad(40)])

        # PID controller
        tau = pid(eta_m, nu, eta_ref, M, wn, zeta, T_f, cfg.h)

        # Control allocation
        if not cfg.alloc_dynamic:
            alpha_c = alpha0.copy()
            u_c = alloc_pseudoinverse(K_max, T_thr, np.eye(4), tau[[0, 1, 5]])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore", message="overflow", category=RuntimeWarning
                )
                alpha_c, u_c, slack = optimal_alloc(
                    tau[[0, 1, 5]],
                    lb,
                    ub,
                    alpha_old,
                    u_old,
                    l_x,
                    l_y,
                    K_max,
                    n_max,
                    cfg.h,
                )
            alpha_old = alpha_c.copy()
            u_old = u_c.copy()

        # Scale to propeller speeds
        u_c = n_max**2 * u_c
        n_c = np.sign(u_c) * np.sqrt(np.abs(u_c))

        # Thruster dynamics: exact discretisation of first-order lag
        # n_actual += _thr_n_alpha * (n_c - n_actual)
        # alpha_actual += _thr_az_alpha * (alpha_c - alpha_actual)
        n_actual = n_c
        alpha_actual = alpha_c

        # n_actual = np.clip(n_actual, -n_max, n_max)
        # alpha_actual = np.clip(alpha_actual, -az_max, az_max)

        # Pass actual (lagged) thruster state to vessel
        ui = np.concatenate([n_actual, alpha_actual])

        sim_eta[i] = eta_m
        sim_nu[i] = nu
        sim_n_cmd[i] = n_c
        sim_alpha_cmd[i] = alpha_c
        sim_n_actual[i] = n_actual
        sim_alpha_actual[i] = alpha_actual
        sim_tau_cmd[i] = tau
        sim_pid_z_int[i] = pid.z_int
        sim_pid_eta_d[i] = pid.eta_d
        sim_pid_tau_p[i] = pid.tau_p
        sim_pid_tau_i[i] = pid.tau_i
        sim_pid_tau_d[i] = pid.tau_d

        # RK4 step (t_idx=3: time is the 4th positional arg after x)
        x = rk4(vessel, cfg.h, x, ui, cfg.Vc, cfg.betaVc, t[i], cfg.beta_wave, t_idx=3)

        # Store forces AFTER RK4 so they reflect this step's commands
        sim_tau_thr[i] = vessel.tau_thr
        sim_tau_wave[i] = vessel.tau_wave
        sim_tau_wave1[i] = vessel.tau_wave1
        sim_nu_LF[i] = vessel.nu_LF
        if not np.isfinite(x).all():
            warnings.warn(
                f"Simulation diverged at t={t[i]:.1f}s — stopping early.",
                category=RuntimeWarning,
            )
            sim_eta[i + 1 :] = np.nan
            sim_nu[i + 1 :] = np.nan
            sim_n_cmd[i + 1 :] = np.nan
            sim_alpha_cmd[i + 1 :] = np.nan
            sim_n_actual[i + 1 :] = np.nan
            sim_alpha_actual[i + 1 :] = np.nan
            sim_tau_cmd[i + 1 :] = np.nan
            sim_tau_wave[i + 1 :] = np.nan
            sim_tau_wave1[i + 1 :] = np.nan
            sim_tau_thr[i + 1 :] = np.nan
            sim_nu_LF[i + 1 :] = np.nan
            sim_pid_z_int[i + 1 :] = np.nan
            sim_pid_eta_d[i + 1 :] = np.nan
            sim_pid_tau_p[i + 1 :] = np.nan
            sim_pid_tau_i[i + 1 :] = np.nan
            sim_pid_tau_d[i + 1 :] = np.nan
            sucess = False
            break
        nu = x[:6]
        eta = x[6:]

        # Update LF filter (first-order, NED surge/sway)
        psi = eta[5]
        c, s = np.cos(psi), np.sin(psi)
        nu_ned_xy = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1]])
        Vc_ned = np.array(
            [
                cfg.Vc * np.cos(cfg.betaVc),
                cfg.Vc * np.sin(cfg.betaVc),
            ]
        )
        vessel.nu_LF = (1 - lf_alpha) * vessel.nu_LF + lf_alpha * (nu_ned_xy - Vc_ned)
        sucess = True

        if progress_callback is not None and i % 500 == 0:
            progress_callback(i, n_steps)

    if progress_callback is not None:
        progress_callback(n_steps, n_steps)

    return {
        "t": t,
        "eta": sim_eta,
        "nu": sim_nu,
        "n_cmd": sim_n_cmd,
        "alpha_cmd": sim_alpha_cmd,
        "n_actual": sim_n_actual,
        "alpha_actual": sim_alpha_actual,
        "tau_cmd": sim_tau_cmd,
        "tau_thr": sim_tau_thr,
        "tau_wave": sim_tau_wave,
        "tau_wave1": sim_tau_wave1,
        "nu_LF": sim_nu_LF,
        "pid_z_int": sim_pid_z_int,
        "pid_eta_d": sim_pid_eta_d,
        "pid_tau_p": sim_pid_tau_p,
        "pid_tau_i": sim_pid_tau_i,
        "pid_tau_d": sim_pid_tau_d,
    }, sucess


def results_to_parquet(
    results: dict[str, NDArray],
    cfg: SimConfig,
    path: Path,
    base_name: str = "osv_sim",
) -> Path:
    """Save simulation results to a Parquet file with SimConfig as metadata.

    Each array in *results* is flattened into named columns.
    For a 2-D array with key ``"eta"`` and shape ``(n, 6)`` the columns
    are ``eta_0 … eta_5``.  1-D arrays (e.g. ``"t"``) become a single column.

    Returns the path of the written file.
    """
    columns: dict[str, np.ndarray] = {}
    for key, arr in results.items():
        if arr.ndim == 1:
            columns[key] = arr
        else:
            for j in range(arr.shape[1]):
                columns[f"{key}_{j}"] = arr[:, j]

    df = pd.DataFrame(columns)
    table = pa.Table.from_pandas(df)

    meta = dict(table.schema.metadata or {})
    meta[b"sim_config"] = json.dumps(asdict(cfg)).encode("utf-8")
    table = table.replace_schema_metadata(meta)

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{base_name}_{cfg.Hs:.2f}_{cfg.Tp:.2f}_{cfg.wave_seed}.parquet"
    filepath = path / filename
    pq.write_table(table, filepath)
    return filepath


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = SimConfig(
        T_final=1000.0,
        h=0.5,
        x_ref=0.0,
        y_ref=0.0,
        psi_ref=0.0,
        Vc=0.0,
        betaVc=np.deg2rad(-140),
        Hs=4.5,
        Tp=8.0,
        beta_wave=np.deg2rad(90),
        # spreading_s=8.0,
        alloc_dynamic=True,
        setpoint_change=False,
    )
    _DRIFT_COEFFS_PATH = Path(__file__).parent / "drift_coefficients.npz"
    # _FORCE_RAO_PATH = Path(__file__).parent / "force_rao.npz"
    start = time.perf_counter()  # warm up timer
    results, success = simulate_osv(config, _DRIFT_COEFFS_PATH, None)
    print(f"Total time: {time.perf_counter() - start:.3f}s")
    t = results["t"]
    eta = results["eta"]
    nu = results["nu"]
    tau_wave = results["tau_wave"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 10))

    from matplotlib.collections import LineCollection

    points = np.column_stack([eta[:, 1], eta[:, 0]]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(list(segments), cmap="viridis")
    lc.set_array(t[:-1])
    axes[0].add_collection(lc)
    axes[0].autoscale()
    fig.colorbar(lc, ax=axes[0], label="Time (s)")
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("North-East positions")
    axes[0].grid(True)

    axes[1].plot(t, np.rad2deg(np.vectorize(ssa)(eta[:, 5])))
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Heading (deg)")
    axes[1].grid(True)

    U = np.sqrt(nu[:, 0] ** 2 + nu[:, 1] ** 2)
    axes[2].plot(t, U)
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Speed (m/s)")
    axes[2].grid(True)

    axes[3].plot(t, tau_wave[:, 0] * 1e-3, label="Surge")
    axes[3].plot(t, tau_wave[:, 1] * 1e-3, label="Sway")
    axes[3].plot(t, tau_wave[:, 5] * 1e-3, label="Yaw")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("Force (kN) / Moment (kNm)")
    axes[3].set_title("Wave drift forces")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(t, np.sqrt(eta[:, 0] ** 2 + eta[:, 1] ** 2))
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("Position (m)")
    axes[4].set_title("Position magnitude")
    axes[4].grid(True)

    fig_thrusters, ax_thrusters = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_thrusters[0].plot(t, results["alpha_actual"][:, 0], label="Thruster 1")
    ax_thrusters[0].plot(t, results["alpha_actual"][:, 1], label="Thruster 2")
    ax_thrusters[0].set_ylabel("Azimuth angle (rad)")
    ax_thrusters[0].legend()
    ax_thrusters[0].grid(True)
    ax_thrusters[1].plot(t, results["n_actual"][:, 0], label="Thruster 1")
    ax_thrusters[1].plot(t, results["n_actual"][:, 1], label="Thruster 2")
    ax_thrusters[1].plot(t, results["n_actual"][:, 2], label="Thruster 3")
    ax_thrusters[1].plot(t, results["n_actual"][:, 3], label="Thruster 4")
    ax_thrusters[1].set_xlabel("Time (s)")
    ax_thrusters[1].set_ylabel("Propeller speed (RPM)")
    ax_thrusters[1].axhline(
        140, color="k", linestyle="--", linewidth=0.8, label="Max RPM (140)"
    )
    ax_thrusters[1].legend()

    plt.tight_layout()
    plt.show()

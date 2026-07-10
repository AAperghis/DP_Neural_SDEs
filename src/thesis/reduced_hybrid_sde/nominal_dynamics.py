"""
Reduced order dynamic positioning model for control design and analysis.

This module defines a simplified physics-based model of the a ship's dynamics under dynamic positioning (DP) control.
The model uses a linear vessel model and includes the proportional and derivative terms of the PID controller, as well as some of the thruster nonlinearity
"""

import jax
import jax.typing as jtp
import jax.numpy as jnp
from matplotlib import pyplot as plt
from thesis.shared.data_structures import PhysicsConfig
from thesis.reduced_order_dp.supply import SupplyVessel


class NominalDynamicsRO:
    def __init__(self, cfg: PhysicsConfig, jit_compile: bool = True):
        """
        Initialize the reduced order DP model.

        Parameters
        ----------
        M : jax.Array
            Mass matrix of the vessel. 3DOF with surge, sway, and yaw.
        D : jax.Array
            Damping matrix of the vessel. 3DOF with surge, sway, and yaw.
        n_max : jax.Array
            Maximum actuator states (RPM or normalised).
        thrust_matrix : jax.Array
            Thruster configuration matrix mapping thruster forces to vessel forces.
            tau = thrust_matrix @ (|n| * n).
        w0 : float | jax.Array, optional
            Natural frequency for PID pole placement (default is 0.1).
        zeta : float | jax.Array, optional
            Damping ratio for PID pole placement (default is 0.7).
        T_n : float | jax.Array, optional
            Actuator time constant (s). Mutually exclusive with ``n_rate``.
        n_rate : float | jax.Array, optional
            Maximum actuator slew rate (units/s). Converted to ``T_n = n_max / n_rate``.
            Mutually exclusive with ``T_n``.
        """
        if cfg.T_n is not None and cfg.n_rate is not None:
            raise ValueError("Specify either T_n or n_rate, not both.")
        if cfg.T_n is None and cfg.n_rate is None:
            raise ValueError("Specify either T_n or n_rate.")

        if cfg.T_n is not None:
            self.T_n = jnp.asarray(cfg.T_n, dtype=float)
        else:
            self.T_n = jnp.asarray(cfg.n_max / cfg.n_rate, dtype=float)

        self.M = cfg.M
        self.D = cfg.D
        self.n_max = cfg.n_max
        self.thrust_matrix = cfg.thrust_matrix
        self.thrust_inv = jnp.linalg.pinv(cfg.thrust_matrix)
        self.M_inv = jnp.linalg.inv(cfg.M)

        m_diag = jnp.diag(jnp.diag(cfg.M))
        d_diag = jnp.diag(jnp.diag(cfg.D))
        self.w0 = cfg.w0
        self.zeta = cfg.zeta

        self.restore_n = cfg.restore_n
        self.disable_controller = cfg.disable_controller

        if isinstance(self.w0, (int, float)):
            self.w0 = self.w0 * jnp.diag(jnp.ones(3))
        if isinstance(self.zeta, (int, float)):
            self.zeta = self.zeta * jnp.diag(jnp.ones(3))
        self.w0 = jnp.asarray(self.w0)
        self.zeta = jnp.asarray(self.zeta)

        # PID gains based on pole placement (Fossen)
        self.Kp = self.w0 @ self.w0 @ m_diag
        self.Kd = 2.0 * self.zeta @ self.w0 @ m_diag - d_diag

        # Pre-compile a jitted version (self captured by closure, not traced)
        if jit_compile:
            self._jit_call = jax.jit(self._compute)
        else:
            self._jit_call = self._compute

    def Rz(self, psi):
        """Rotation matrix for yaw angle."""
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._jit_call(x)

    def _compute(self, x: jax.Array) -> jax.Array:
        """
        Compute the time derivative of the state vector.

        Parameters
        ----------
        x : jax.Array
            State vector containing [eta, nu, n], where eta is the position and orientation,
            nu is the velocity, and n is the thruster actuator states.

        Returns
        -------
        jax.Array
            Time derivative of the state vector.
        """
        eta = x[0:3]
        nu = x[3:6]
        n = x[6:]

        # Soft saturation: tanh preserves gradients near ±n_max so the
        # optimiser can still push RPMs back from saturation.
        n_sat = self.n_max * jnp.tanh(n / self.n_max)

        # Thruster nonlinearity (squared relationship)
        thrust = self.thrust_matrix @ (jnp.abs(n_sat) * n_sat)

        eta_dot = self.Rz(eta[2]) @ nu

        # Vessel dynamics
        nu_dot = self.M_inv @ (thrust - self.D @ nu)

        if self.disable_controller:
            # No control loop: the SDE must learn n_dot via C(z).
            # Quadratic restoring: negligible within bounds, grows
            # rapidly outside to prevent state windup.
            excess = n - n_sat
            n_dot = -excess * jnp.abs(excess) / (self.T_n * self.n_max)
        else:
            # Control law (PID with pole placement, Fossen Ch. 12)
            R = self.Rz(eta[2])
            e = eta  # Setpoint is zero for regulation
            tau_p = -R.T @ self.Kp @ e
            tau_d = -self.Kd @ nu
            tau = tau_p + tau_d

            # Invert quadratic thrust: tau = T @ (|n|*n) => f = T_pinv @ tau, n = sign(f)*sqrt(|f|)
            f_desired = self.thrust_inv @ tau
            n_target = jnp.sign(f_desired) * jnp.sqrt(jnp.abs(f_desired))

            # Thruster dynamics (first-order lag)
            n_dot = (n_target - n_sat) / self.T_n
        return jnp.concatenate([eta_dot, nu_dot, n_dot])


if __name__ == "__main__":
    # Example usage
    vessel = SupplyVessel()

    model = NominalDynamicsRO(
        cfg=PhysicsConfig(
            M=vessel.M3,
            D=vessel.D3,
            n_max=vessel.n_max,
            thrust_matrix=vessel.B,
            T_n=vessel.T_n,
            w0=vessel.wn,
            zeta=vessel.zeta,
        ),
        jit_compile=True,
    )

    scale_mean = 4
    scale_std = 2

    x = jnp.ones(12)  # Initial state
    x_dot = model(x) / scale_std  # Standardised derivative
    dt = 0.5
    ts = jnp.arange(0, 1200, dt)
    x_traj_rk4 = jnp.zeros((len(ts), len(x)))
    x_traj_rk4 = x_traj_rk4.at[0].set(x)
    x_traj_eu = jnp.zeros((len(ts), len(x)))
    x_traj_eu = x_traj_eu.at[0].set(x)

    for t in range(1, len(ts)):
        x_traj_eu = x_traj_eu.at[t].set(
            x_traj_eu[t - 1] + dt * model(x_traj_eu[t - 1]) / scale_std
        )

    for t in range(1, len(ts)):
        k1 = model(x_traj_rk4[t - 1]) / scale_std
        k2 = model(x_traj_rk4[t - 1] + 0.5 * dt * k1) / scale_std
        k3 = model(x_traj_rk4[t - 1] + 0.5 * dt * k2) / scale_std
        k4 = model(x_traj_rk4[t - 1] + dt * k3) / scale_std
        x = x_traj_rk4[t - 1] + (dt / 6.0) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )  # RK4 integration
        x_traj_rk4 = x_traj_rk4.at[t].set(x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(ts, x_traj_rk4[:, 0], label="x (m) - RK4")
    ax.plot(ts, x_traj_rk4[:, 1], label="y (m) - RK4")
    ax.plot(ts, x_traj_rk4[:, 2], label="psi (rad) - RK4")
    ax.plot(ts, x_traj_eu[:, 0], label="x (m) - Euler", linestyle="--")
    ax.plot(ts, x_traj_eu[:, 1], label="y (m) - Euler", linestyle="--")
    ax.plot(ts, x_traj_eu[:, 2], label="psi (rad) - Euler", linestyle="--")
    ax.set_title("Position and Orientation")
    ax.legend()

    plt.show()

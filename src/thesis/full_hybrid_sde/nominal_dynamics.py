"""Reduced-order dynamic positioning model for MSS vessel with azimuth thrusters.

3-DOF (surge, sway, yaw) vessel dynamics model compatible with JAX/JIT,
designed for use inside the :class:`FullHybridSDE`.  Extends the fixed-
thruster :class:`~thesis.reduced_hybrid_sde.nominal_dynamics.NominalDynamicsRO`
concept to azimuth thrusters whose angles are part of the state vector.

State layout::

    x = [eta(3), nu(3), n(n_thr), alpha(n_az)]

where ``n_thr`` is the number of thrusters and ``n_az`` the number of
azimuth thrusters (the remaining thrusters are tunnel-type).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from thesis.shared.data_structures import FullOrderPhysicsConfig


class NominalDynamicsFO:
    """JAX-compatible reduced-order DP model for azimuth-thruster vessels.

    Args:
        cfg: Physical parameters (mass, damping, thruster layout, …).
        jit_compile: Whether to JIT-compile the forward pass (default ``True``).
    """

    def __init__(self, cfg: FullOrderPhysicsConfig, jit_compile: bool = True) -> None:
        self.M = jnp.asarray(cfg.M, dtype=jnp.float32)
        self.D = jnp.asarray(cfg.D, dtype=jnp.float32)
        self.M_inv = jnp.linalg.inv(self.M)
        self.n_max = jnp.asarray(cfg.n_max, dtype=jnp.float32)
        self.K_thr = jnp.asarray(cfg.K_thr, dtype=jnp.float32)
        self.l_x = jnp.asarray(cfg.l_x, dtype=jnp.float32)
        self.l_y = jnp.asarray(cfg.l_y, dtype=jnp.float32)
        self.n_tunnel = cfg.n_tunnel
        self.n_thr = self.n_max.shape[0]
        self.n_azimuth = self.n_thr - self.n_tunnel
        self.T_n = jnp.asarray(cfg.T_n, dtype=jnp.float32)
        self.T_alpha = jnp.asarray(cfg.T_alpha, dtype=jnp.float32)
        self.alpha_max = cfg.alpha_max
        self.disable_controller = cfg.disable_controller

        # Index mappings into state vector
        self.eta_idx = jnp.array(cfg.eta_idx, dtype=jnp.int32)
        self.nu_idx = jnp.array(cfg.nu_idx, dtype=jnp.int32)
        self.n_idx = jnp.array(cfg.n_idx, dtype=jnp.int32)
        self.alpha_idx = jnp.array(cfg.alpha_idx, dtype=jnp.int32)
        self.state_size = (
            max(
                max(cfg.eta_idx),
                max(cfg.nu_idx),
                max(cfg.n_idx),
                max(cfg.alpha_idx),
            )
            + 1
        )

        # PID gains (pole placement, Fossen Ch. 12)
        w0 = jnp.asarray(cfg.w0, dtype=jnp.float32)
        zeta = jnp.asarray(cfg.zeta, dtype=jnp.float32)
        if w0.ndim == 0:
            w0 = w0 * jnp.eye(3)
        if zeta.ndim == 0:
            zeta = zeta * jnp.eye(3)
        m_diag = jnp.diag(jnp.diag(self.M))
        d_diag = jnp.diag(jnp.diag(self.D))
        self.Kp = w0 @ w0 @ m_diag
        self.Kd = 2.0 * zeta @ w0 @ m_diag - d_diag

        # Pseudoinverse for fixed-angle allocation
        T0 = self._thrust_matrix(jnp.zeros(self.n_azimuth))
        self.T0_pinv = jnp.linalg.pinv(T0 @ self.K_thr)

        if jit_compile:
            self._jit_call = jax.jit(self._compute)
        else:
            self._jit_call = self._compute

    # ------------------------------------------------------------------
    # Thrust geometry
    # ------------------------------------------------------------------

    def Rz(self, psi: jax.Array) -> jax.Array:
        c, s = jnp.cos(psi), jnp.sin(psi)
        return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _thrust_matrix(self, alpha: jax.Array) -> jax.Array:
        """Compute 3×n_thr allocation matrix ``T(alpha)``.

        Tunnel thrusters have a fixed angle of π/2 (pure lateral force).
        Azimuth thrusters use the provided angles.
        """
        tunnel_angles = jnp.full(self.n_tunnel, jnp.pi / 2)
        angles = jnp.concatenate([tunnel_angles, alpha])
        cos_a = jnp.cos(angles)
        sin_a = jnp.sin(angles)
        T = jnp.stack(
            [
                cos_a,
                sin_a,
                self.l_x * sin_a - self.l_y * cos_a,
            ]
        )  # (3, n_thr)
        return T

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._jit_call(x)

    def _compute(self, x: jax.Array) -> jax.Array:
        """Compute dx/dt for state ``[eta, nu, n, alpha, ...]``.

        Only the components at ``eta_idx``, ``nu_idx``, ``n_idx``,
        ``alpha_idx`` are evolved; all other state derivatives are zero
        (to be handled by the neural correction).
        """
        eta = x[self.eta_idx]
        nu = x[self.nu_idx]
        n = x[self.n_idx]
        alpha = x[self.alpha_idx]

        n_sat = self.n_max * jnp.tanh(n / self.n_max)
        alpha_sat = self.alpha_max * jnp.tanh(alpha / self.alpha_max)

        # Kinematics
        psi = eta[2]
        eta_dot = self.Rz(psi) @ nu

        # Thrust
        T_mat = self._thrust_matrix(alpha_sat)
        thrust = T_mat @ self.K_thr @ (jnp.abs(n_sat) * n_sat)

        # Vessel dynamics
        nu_dot = self.M_inv @ (thrust - self.D @ nu)

        # Actuator dynamics
        if self.disable_controller:
            excess_n = n - n_sat
            n_dot = -excess_n * jnp.abs(excess_n) / (self.T_n * self.n_max)
            excess_alpha = alpha - alpha_sat
            alpha_dot = (
                -excess_alpha * jnp.abs(excess_alpha) / (self.T_alpha * self.alpha_max)
            )
        else:
            # PID controller → desired force
            R = self.Rz(psi)
            tau_p = -R.T @ self.Kp @ eta
            tau_d = -self.Kd @ nu
            tau = tau_p + tau_d

            # Pseudoinverse allocation at current azimuth
            T_current = self._thrust_matrix(alpha)
            T_K = T_current @ self.K_thr
            T_K_pinv = jnp.linalg.pinv(T_K)
            f_desired = T_K_pinv @ tau
            n_target = jnp.sign(f_desired) * jnp.sqrt(jnp.abs(f_desired))
            n_dot = (n_target - n) / self.T_n

            # Azimuth angles: hold constant (simplified controller)
            alpha_dot = jnp.zeros_like(alpha)

        # Assemble full state derivative
        dx = jnp.zeros_like(x)
        dx = dx.at[self.eta_idx].set(eta_dot)
        dx = dx.at[self.nu_idx].set(nu_dot)
        dx = dx.at[self.n_idx].set(n_dot)
        dx = dx.at[self.alpha_idx].set(alpha_dot)
        return dx


def default_mss_physics_config() -> FullOrderPhysicsConfig:
    """Create an :class:`FullOrderPhysicsConfig` from default OSV parameters.

    Extracts 3-DOF mass and damping matrices from the 6-DOF MSS OSV
    model and uses the simulation thruster parameters.
    """
    from thesis.full_order_dp.vessel import OSV

    osv = OSV()
    v = osv.params

    idx_3dof = np.array([0, 1, 5])
    M3 = np.asarray(v.M[np.ix_(idx_3dof, idx_3dof)], dtype=np.float32)
    D3 = np.asarray(v.D[np.ix_(idx_3dof, idx_3dof)], dtype=np.float32)

    # Simulation thruster parameters (from gen_training_data / simulation.py)
    K_max = np.array([300e3, 300e3, 655e3, 655e3], dtype=np.float32)
    n_max = np.array([140.0, 140.0, 150.0, 150.0], dtype=np.float32)
    K_thr = np.diag(K_max / n_max**2).astype(np.float32)

    return FullOrderPhysicsConfig(
        M=jnp.array(M3),
        D=jnp.array(D3),
        n_max=jnp.array(n_max),
        K_thr=jnp.array(K_thr),
        l_x=jnp.array([37.0, 35.0, -42.0, -42.0]),
        l_y=jnp.array([0.0, 0.0, 7.0, -7.0]),
        n_tunnel=2,
        w0=jnp.diag(jnp.array([0.1, 0.1, 0.3])),
        zeta=jnp.eye(3),
        T_n=1.0,
        T_alpha=2.0,
        alpha_max=float(np.pi / 3),
        disable_controller=True,
    )

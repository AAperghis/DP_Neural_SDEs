"""
ReducedHybridSDE model implementation.

Based on the Latent Force framework, coupled with universal differential equations and latent SDEs. The model consists of:
- A physics-based drift term derived from a reduced-order vessel dynamics model.
- Neural network corrections to the drift, conditioned on the current state and a learned context representation of the observed trajectory.
- A diffusion term parameterized by a neural network, allowing for stochasticity in the latent dynamics.
- An encoder that processes observed trajectories to produce context for the drift correction and initial
state distribution.

References:
- Latent Force Models: https://arxiv.org/abs/1506.07371
- Universal Differential Equations: https://arxiv.org/abs/2001.04385
- Latent SDEs: https://arxiv.org/abs/2002.094

"""

import equinox as eqx

import jax
import jax.typing as jtp
import jax.numpy as jnp
import jax.random as jr
import diffrax

import lineax
from thesis.reduced_hybrid_sde.nominal_dynamics import NominalDynamicsRO
from thesis.shared.data_structures import FieldConfig, PhysicsConfig
from thesis.shared.model import AbstractHybridSDE, MeanReversion
from thesis.shared.utils import Encoder
from thesis.shared.vector_fields import (
    AbstractVectorField,
)


class ReducedHybridSDE(AbstractHybridSDE):
    encoder: Encoder
    nominal_dynamics: NominalDynamicsRO
    f: AbstractVectorField  # Posterior drift correction (context-dependent)
    h: AbstractVectorField  # Prior drift correction (context-free)
    g: AbstractVectorField  # Diffusion field
    C: eqx.nn.Linear  # Mapping from latent space to physics model control inputs
    mean_reversion: MeanReversion | None

    qz0_posterior: eqx.nn.Linear  # Maps encoder context to mean and logvar of q(z0|x)
    px0_mean: jax.Array  # Learned prior mean for x0, shape (data_size,)
    px0_logvar: jax.Array  # Learned prior log-variance for x0, shape (data_size,)
    pz0_mean: jax.Array
    pz0_logvar: jax.Array
    data_mean: jax.Array
    data_std: jax.Array
    latent_size: int = eqx.field(static=True)
    data_size: int = eqx.field(static=True)
    solver: eqx.Module = eqx.field(static=True)
    indirect_eta: bool = eqx.field(static=True, default=True)
    eta_size: int = eqx.field(static=True, default=3)
    dt: float = eqx.field(static=True, default=0.1)
    eps: float = eqx.field(static=True, default=1e-6)
    # KL-rate denominator floor (mirrors the Full-Order model so it can be tuned).
    _kl_eps: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        f_config: FieldConfig,
        h_config: FieldConfig,
        g_config: FieldConfig,
        phys_config: PhysicsConfig,
        dt: float = 0.1,
        indirect_eta: bool = True,
        kl_eps: float = 1e-6,
        *,
        key,
        **kwargs,
    ) -> None:
        # ReducedHybridSDE vector fields see concatenated [x, z]
        f_config.input_size = f_config.latent_size + data_size
        h_config.input_size = h_config.latent_size + data_size
        g_config.input_size = g_config.latent_size + data_size

        dec_key = self._init_common(
            data_size,
            latent_size,
            context_size,
            hidden_size,
            f_config,
            h_config,
            g_config,
            dt,
            key=key,
        )
        _C = eqx.nn.Linear(latent_size, data_size, key=dec_key)
        # Shrink C at init so latent forcing is ~0 and dynamics start near pure physics.
        _C_bias = None if _C.bias is None else _C.bias * 0.01
        self.C = eqx.tree_at(
            lambda c: (c.weight, c.bias),
            _C,
            (_C.weight * 0.01, _C_bias),
        )
        self.nominal_dynamics = NominalDynamicsRO(phys_config)
        self.px0_mean = jnp.zeros(data_size, dtype=jnp.float32)
        self.px0_logvar = jnp.zeros(data_size, dtype=jnp.float32)
        self.solver = diffrax.Heun()
        self.indirect_eta = indirect_eta
        self.eta_size = 3  # eta is always first 3 components (surge, sway, yaw)
        self._kl_eps = kl_eps

    def __call__(
        self,
        ts: jax.Array,
        ys: jax.Array,
        *,
        key,
        log_sigma: jax.Array,
        dt: float | None = None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute negative ELBO for given data ys and time points ts.

        Notation:
            x: State in measurement space (data_size,)
            z: State in latent space (latent_size,)
            xs: Decoded measurement space trajectory (T, data_size)
            zs: Latent space trajectory (T, latent_size)
            ys: Observed data (T, data_size)

        Uses the following approach:
            dx = nominal_dynamics(t, x) + C z ) dt + 0  dW
            dz = f(t, z, x, ctx) dt + g(t, z, x) dW  (Posterior SDE)
            dKL = (1/2 (g^-1 (f(t, z, x, ctx) - h(t, z, x)))^2) dt + 0 dW

        Args:
            ts: Time points corresponding to ys, shape (T),
            ys: Observed data, shape (T, data_size).
            key: JAX random key for sampling.
            log_sigma: Log observation noise standard deviation, shape (data_size,).

        Returns:
           Tuple of (log_likelihood, kl_divergence_initial, kl_divergence_path, xs, zs).
           log_likelihood: Scalar log p(x|z) averaged over time.
           kl_divergence_initial: Scalar KL divergence between q(z0|x) and p(z0).
           kl_divergence_path: Scalar KL divergence accumulated along the SDE path.
           xs: Measurement space trajectory, shape (T, data_size).
           zs: Latent space trajectory, shape (T, latent_size).
        """
        print("[Debug] ReducedHybridSDE() compiled")
        bm_key, z0_key = jr.split(key)

        # Encode context (reverse time GRU)
        ctx = self.encoder(ys)  # (T, ctx_size)

        # SDE integration setup
        t0, t1 = ts[0], ts[-1]
        if self.g.diagonal:
            bm_shape = (self.data_size + self.latent_size + 1,)
        else:
            bm_shape = (self.g.control_size,)
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=self.dt / 2, shape=bm_shape, key=bm_key
        )

        drift_term = diffrax.ODETerm(self.posterior_drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, bm)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)

        saveat = diffrax.SaveAt(ts=ts)

        # Initial x0 directly from observations
        x0 = ys[0]  # (data_size,)

        # Sample initial z0 from posterior
        z0, qz0_mean, qz0_logstd = self._sample_z0_posterior(ctx[0], z0_key)

        # Augmented initial state: [x0, z0, 0] (KL accumulator starts at 0)
        y0_aug = jnp.concatenate([x0, z0, jnp.zeros(1)])

        sol: diffrax.Solution = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt if dt is not None else self.dt,
            y0_aug,
            saveat=saveat,
            args=(ts, ctx),
            max_steps=int(ts.shape[0] * 1.1),
        )
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs: jax.Array = sol.ys[..., : self.data_size]  # Extract measurement space part of state
        zs: jax.Array = sol.ys[..., self.data_size : -1]  # Extract latent space part of state
        duration = ts[-1] - ts[0]
        logqp_path = (
            sol.ys[-1, -1] / duration
        )  # Normalize by trajectory duration (per unit time)

        # --- Log-likelihood of observations ---
        sigma = jnp.exp(log_sigma)  # (data_size,)

        T = ys.shape[0]
        log_pxs = (jax.scipy.stats.norm.logpdf(ys, xs, sigma).sum(axis=-1).sum()) / (
            T * self.data_size
        )  # mean over time and features

        # --- KL divergence of initial conditions ---
        logqp0 = self._kl_initial(qz0_mean, qz0_logstd)

        return zs, xs, log_pxs, logqp0, logqp_path

    def _solve_prior(
        self,
        ts: jax.Array,
        *,
        key,
        x0: jax.Array | None = None,
        dt: float | None = None,
        **kwargs,
    ) -> diffrax.Solution:
        """Integrate the prior SDE and return the raw Diffrax solution."""
        bm_key, z0_key, x0_key = jr.split(key, 3)

        t0, t1 = ts[0], ts[-1]
        if self.g.diagonal:
            bm_shape = (self.data_size + self.latent_size,)
        else:
            bm_shape = (self.g.control_size,)
        bm = diffrax.UnsafeBrownianPath(shape=bm_shape, key=bm_key)

        if self.g.diagonal:

            def _g_diag(t, y, args):
                y_state = y
                g_val = self.g(t, y_state, args)
                g_aug = jnp.concatenate([jnp.zeros(self.data_size), g_val])
                return lineax.DiagonalLinearOperator(g_aug)

            diffusion_term = diffrax.ControlTerm(_g_diag, bm)
        else:

            def _g_mat(t, y, args):
                y_state = y
                g_val = self.g(t, y_state, args)
                return jnp.pad(g_val, ((self.data_size, 0), (0, 0)))

            diffusion_term = diffrax.ControlTerm(_g_mat, bm)

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(self.prior_drift),
            diffusion_term,
        )
        saveat = diffrax.SaveAt(ts=ts)

        if x0 is None:
            px0_std = jnp.exp(jnp.clip(self.px0_logvar * 0.5, -5.0, 2.0))
            x0 = self.px0_mean + px0_std * jr.normal(x0_key, shape=self.px0_mean.shape)

        z0 = self._sample_z0_prior(z0_key)
        y0 = jnp.concatenate([x0, z0])

        return diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt if dt is not None else self.dt,
            y0,
            saveat=saveat,
            max_steps=ts.shape[0] + 1,
            adjoint=diffrax.ForwardMode(),
        )

    @eqx.filter_jit
    def sample_prior(
        self,
        ts: jax.Array,
        *,
        key,
        x0: jax.Array | None = None,
        unscale: bool = True,
        long_sample: bool = False,
        dt: float | None = None,

        **kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample from the prior SDE.

        Args:
            ts: Time points, shape (T,).
            key: JAX random key.
            x0: Initial state in standardised space, shape (data_size,).
                If None, samples from the learned prior p(x0).
            unscale: If True, return trajectory in physical units.
            long_sample: Allow trajectories longer than 50 000 steps.
            return_latents: Also return the latent-state trajectory ``zs`` so the
                cause of divergent behaviour can be analysed post-hoc.

        Returns:
            Trajectory in measurement space, shape (T, data_size). When
            ``return_latents`` is True, returns ``(xs, zs)`` with ``zs`` of shape
            (T, latent_size) in latent space.
        """
        if ts.shape[0] > 50000 and not long_sample:
            raise RuntimeError(
                f"Refusing to sample long trajectory with {ts.shape[0]} time points. Set long_sample=True to override."
            )

        sol = self._solve_prior(ts, key=key, x0=x0, dt=dt, **kwargs)
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs = sol.ys[..., : self.data_size]
        if unscale:
            xs = jax.vmap(self.unscale)(xs)

        return xs, sol.ys[..., self.data_size :]
        

    @eqx.filter_jit
    def sample_prior_ode(
        self,
        ts: jax.Array,
        *,
        key,
        x0: jax.Array | None = None,
        z0: jax.Array | None = None,
        unscale: bool = True,
        dt: float | None = None,
        **kwargs,
    ) -> jax.Array:
        """Integrate the prior drift ODE with no diffusion (deterministic).

        Uses the same initial-state sampling as :meth:`sample_prior` so that
        passing the same *key* with ``common_noise=True`` in
        :class:`~thesis.deep_ensemble.DeepEnsemble` gives comparable
        starting points across members.  The only source of difference
        between members is then the learned drift function.

        Args:
            ts: Time points, shape ``(T,)``.
            key: PRNG key used to sample the initial state ``(x0, z0)``.
            x0: Override the sampled initial observation state.
            z0: Override the sampled initial latent state.
            unscale: Return physical units if ``True``.
            dt: Timestep override.

        Returns:
            Deterministic trajectory, shape ``(T, data_size)``.
        """
        _, z0_key, x0_key = jr.split(key, 3)

        if x0 is None:
            px0_std = jnp.exp(jnp.clip(self.px0_logvar * 0.5, -5.0, 2.0))
            x0 = self.px0_mean + px0_std * jr.normal(x0_key, shape=self.px0_mean.shape)
        if z0 is None:
            z0 = self._sample_z0_prior(z0_key)
        y0 = jnp.concatenate([x0, z0])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.prior_drift),
            self.solver,
            ts[0],
            ts[-1],
            dt if dt is not None else self.dt,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=ts.shape[0] + 1,
            adjoint=diffrax.ForwardMode(),
        )
        xs = sol.ys[..., : self.data_size]
        if unscale:
            xs = jax.vmap(self.unscale)(xs)
        return xs

    @eqx.filter_jit
    def sample_prior_full(
        self,
        ts: jax.Array,
        *,
        key,
        x0: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample from the prior SDE and return full state plus drift/diffusion.

        Returns:
            Tuple of (xs, zs, h_vals, g_norms) each shape (T, ...).
            xs: (T, data_size) in standardised space.
            zs: (T, latent_size).
            h_vals: (T, latent_size) prior drift evaluated along trajectory.
            g_norms: (T, latent_size) per-dim diffusion magnitude.
        """
        sol = self._solve_prior(ts, key=key, x0=x0)
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs = sol.ys[..., : self.data_size]
        zs = sol.ys[..., self.data_size :]

        # Evaluate h and g along the trajectory
        def _eval_fields(y_t, t):
            h_val = self.h(t, y_t, None)
            g_val = self.g(t, y_t, None)
            if self.g.diagonal:
                g_norm = g_val
            else:
                g_norm = jnp.sqrt(jnp.sum(g_val**2, axis=-1))
            return h_val, g_norm

        h_vals, g_norms = jax.vmap(_eval_fields)(sol.ys, ts)
        return xs, zs, h_vals, g_norms

    @eqx.filter_jit
    def sample_posterior(
        self, xs: jax.Array, ts: jax.Array, *, key, unscale: bool = True, **kwargs
    ) -> jax.Array:
        """Sample from the posterior SDE given observations xs and time points ts.
        Returns trajectory in measurement space, shape (T, data_size).
        If *unscale* is True (default), output is in physical units.
        """
        bm_key, z0_key = jr.split(key)

        # Encode context (reverse time GRU)
        ctx = self.encoder(xs)  # (T, ctx_size)

        # SDE integration setup
        t0, t1 = ts[0], ts[-1]
        if self.g.diagonal:
            bm_shape = (self.data_size + self.latent_size + 1,)
        else:
            bm_shape = (self.g.control_size,)
        bm = diffrax.UnsafeBrownianPath(shape=bm_shape, key=bm_key)

        drift_term = diffrax.ODETerm(self.posterior_drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, bm)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)

        saveat = diffrax.SaveAt(ts=ts)

        # Initial x0 directly from observations
        x0 = xs[0]  # (data_size,)

        # Sample initial z0 from posterior
        z0, _, _ = self._sample_z0_posterior(ctx[0], z0_key)

        # Augmented initial state: [x0, z0, 0] (KL accumulator starts at 0)
        y0_aug = jnp.concatenate([x0, z0, jnp.zeros(1)])

        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            self.dt,
            y0_aug,
            saveat=saveat,
            args=(ts, ctx),
            max_steps=int(ts.shape[0] * 1.1),
            adjoint=diffrax.ForwardMode(),
        )
        xs_hat = sol.ys[..., : self.data_size]  # Extract measurement space
        if unscale:
            xs_hat = jax.vmap(self.unscale)(xs_hat)
        return xs_hat

    def prior_drift(self, t: jtp.ArrayLike, y: jax.Array, args) -> jax.Array:
        """Compute prior drift for state [x, z].
        Physics model evolves x, prior correction h evolves z.
        """
        x = y[..., : self.data_size]
        z = y[..., self.data_size :]
        xz = y  # Full [x, z] state

        # Unscale x to physical units, compute physics, rescale derivative
        # x = x_scaled * data_std + data_mean
        # dx/dt = dx_scaled/dt * data_std
        x_phys = x * self.data_std + self.data_mean
        cz = self.C(z)
        if self.indirect_eta:
            # Zero out eta components so forcing only affects eta through kinematics
            cz = cz.at[: self.eta_size].set(0.0)
        dx = self.nominal_dynamics(x_phys) / self.data_std + cz

        dz = self.h(t, xz, args)
        if self.mean_reversion is not None:
            dz = dz + self.mean_reversion(z)
        return jnp.concatenate([dx, dz])

    def posterior_drift(self, t: jtp.ArrayLike, y: jax.Array, args) -> jax.Array:
        """Compute drift and KL integrand for augmented state [z, kl_accumulator].
        KL drift is based on Girsanov's theorem.

        When an ODE field is present the frozen pretrained drift is added
        to both f and h, so it cancels in the KL and the divergence
        measures only the difference between correction networks.

        Let f = posterior correction, h = prior correction, g = diffusion.
        The instantaneous KL divergence rate is:
            0.5 * (f - h)^T @ Sigma^{-1} @ (f - h)
        where Sigma = g g^T (matrix case) or diag(g^2) (diagonal case).
        """
        x = y[..., : self.data_size]  # Measurement space part of state
        z = y[
            ..., self.data_size : -1
        ]  # Latent space part of state (exclude KL accumulator)
        xz = y[..., :-1]  # Full state excluding KL accumulator

        z_f = self.f(t, xz, args)
        z_h = self.h(t, xz, args)
        g_val = self.g(t, xz, args)

        # Mean reversion is shared: cancels in the KL (delta = f - h),
        # but gets trained through the posterior's reconstruction loss.
        kl_rate = self._kl_rate(z_f, z_h, g_val)

        # Posterior z dynamics include mean reversion (shared with prior)
        dz = z_f
        if self.mean_reversion is not None:
            dz = dz + self.mean_reversion(z)

        # Physics model drift with latent forcing (unscale → physics → rescale)
        x_phys = x * self.data_std + self.data_mean
        cz = self.C(z)
        if self.indirect_eta:
            # Zero out eta components so forcing only affects eta through kinematics
            cz = cz.at[: self.eta_size].set(0.0)
        dx = self.nominal_dynamics(x_phys) / self.data_std + cz

        return jnp.concatenate([dx, dz, jnp.array([kl_rate])])

    def diffusion(self, t: jtp.ArrayLike, y: jax.Array, args):
        """Compute diffusion matrix for state [x, z].
        Only z has diffusion; x is deterministic given z.
        Augment with zeros for x and KL accumulator, and return in correct shape for diffrax.
        """
        y_state = y[..., :-1]
        g_val = self.g(t, y_state, args)

        if self.g.diagonal:
            # g_val is (L,); prepend data_size zeros for x, append 0 for KL accumulator
            g_aug = jnp.concatenate(
                [jnp.zeros(self.data_size), g_val, jnp.array([0.0])]
            )
            return lineax.DiagonalLinearOperator(g_aug)
        else:
            # g_val is (L, C); pad with data_size zero rows above for x and 1 zero row below for KL
            return jnp.pad(g_val, ((self.data_size, 1), (0, 0)))

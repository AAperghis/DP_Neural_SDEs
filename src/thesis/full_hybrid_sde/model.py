"""FullHybridSDE — physics-informed latent SDE for MSS vessel data.

Extends :class:`~thesis.shared.model.AbstractHybridSDE` with:

* **Azimuth-thruster physics** via
  :class:`~thesis.full_order_dp.reduced_order_model.NominalDynamicsFO`.
* **Wave conditioning** — ``(Hs, Tp, beta_w)`` is concatenated to
  the vector-field inputs so the drift/diffusion networks can adapt to
  the current sea state.
* **Flexible state layout** — the physics model operates on a
  configurable subset of the data dimensions (eta, nu, n, alpha)
  while the neural network sees the full ``[x, z, wave_cond]`` input.

The augmented ODE/SDE state during training is::

    [x(data_size), z(latent_size), kl_accumulator(1)]

and wave conditioning is passed through ``args``.
"""

from __future__ import annotations


import diffrax
import equinox as eqx
import jax
import jax.typing as jtp
import jax.numpy as jnp
import jax.random as jr
import lineax

from thesis.full_hybrid_sde.nominal_dynamics import NominalDynamicsFO
from thesis.shared.data_structures import FieldConfig, FieldType, FullOrderPhysicsConfig
from thesis.shared.model import AbstractHybridSDE, MeanReversion
from thesis.shared.utils import Encoder
from thesis.shared.vector_fields import AbstractVectorField


class FullHybridSDE(AbstractHybridSDE):
    """Physics-informed latent SDE for MSS vessel data.

    The vector fields ``f``, ``h``, ``g`` receive the concatenated input
    ``[x, z, wave_cond]`` so they can condition on the sea state.  The
    physics model contributes deterministic kinematics and thrust
    dynamics to the measurement-space drift.
    """

    encoder: Encoder
    nominal_dynamics: NominalDynamicsFO
    f: AbstractVectorField
    h: AbstractVectorField
    g: AbstractVectorField
    C: eqx.nn.Linear
    mean_reversion: MeanReversion | None

    qz0_posterior: eqx.nn.Linear
    px0_mean: jax.Array
    px0_logvar: jax.Array
    pz0_mean: jax.Array
    pz0_logvar: jax.Array
    data_mean: jax.Array
    data_std: jax.Array

    latent_size: int = eqx.field(static=True)
    data_size: int = eqx.field(static=True)
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    n_wave_params: int = eqx.field(static=True, default=3)
    indirect_eta: bool = eqx.field(static=True, default=True)
    eta_size: int = eqx.field(static=True, default=3)
    dt: float = eqx.field(static=True, default=0.1)
    eps: float = eqx.field(static=True, default=1e-6)
    _h_uses_film: bool = eqx.field(static=True, default=False)
    _g_uses_film: bool = eqx.field(static=True, default=False)
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
        phys_config: FullOrderPhysicsConfig,
        n_wave_params: int = 3,
        dt: float = 0.1,
        indirect_eta: bool = True,
        kl_eps: float = 1e-6,
        *,
        key,
        **kwargs,
    ) -> None:
        # Field input is [x, z, wave_cond]
        field_input = latent_size + data_size + n_wave_params
        f_config.input_size = field_input

        # h: FiLM receives wave_cond separately, trunk gets [x, z] only
        self._h_uses_film = h_config.field_type == FieldType.FILM_STATE
        if self._h_uses_film:
            h_config.input_size = latent_size + data_size
            h_config.film_size = n_wave_params
        else:
            h_config.input_size = field_input

        # g: FiLM receives wave_cond separately, trunk gets [x, z] only
        self._g_uses_film = g_config.field_type == FieldType.FILM_STATE
        if self._g_uses_film:
            g_config.input_size = latent_size + data_size
            g_config.film_size = n_wave_params
        else:
            g_config.input_size = field_input

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
        # Scale down C so the latent forcing is near-zero at init.
        _C_bias = None if _C.bias is None else _C.bias * 0.01
        self.C = eqx.tree_at(
            lambda c: (c.weight, c.bias),
            _C,
            (_C.weight * 0.01, _C_bias),
        )
        self.nominal_dynamics = NominalDynamicsFO(phys_config)
        self.px0_mean = jnp.zeros(data_size, dtype=jnp.float32)
        self.px0_logvar = jnp.zeros(data_size, dtype=jnp.float32)
        self.solver = diffrax.Heun()
        self.indirect_eta = indirect_eta
        self.eta_size = 3
        self.n_wave_params = n_wave_params
        self._kl_eps = kl_eps

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        ts: jax.Array,
        ys: jax.Array,
        *,
        key,
        log_sigma: jax.Array,
        wave_cond: jax.Array | None = None,
        dt: float | None = None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute negative ELBO for observations *ys* and sea state *wave_cond*.

        Args:
            ts: Time points, shape ``(T,)``.
            ys: Observed data, shape ``(T, data_size)``.
            key: PRNG key.
            log_sigma: Log observation noise std, shape ``(data_size,)``.
            wave_cond: Sea-state conditioning, shape ``(n_wave_params,)``.
            dt: Integration timestep override.

        Returns:
            ``(zs, xs, log_pxs, kl_initial, kl_path)``
        """
        if wave_cond is None:
            raise ValueError("wave_cond is required.")
        bm_key, z0_key = jr.split(key)

        ctx = self.encoder(ys)

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

        x0 = ys[0]
        z0, qz0_mean, qz0_logstd = self._sample_z0_posterior(ctx[0], z0_key)
        y0_aug = jnp.concatenate([x0, z0, jnp.zeros(1)])

        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt if dt is not None else self.dt,
            y0_aug,
            saveat=saveat,
            args=(ts, ctx, wave_cond),
            max_steps=int(ts.shape[0] * 1.1),
        )
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")

        xs = sol.ys[..., : self.data_size]
        zs = sol.ys[..., self.data_size : -1]
        duration = ts[-1] - ts[0]
        logqp_path = sol.ys[-1, -1] / duration

        sigma = jnp.exp(log_sigma)
        T = ys.shape[0]
        log_pxs = (jax.scipy.stats.norm.logpdf(ys, xs, sigma).sum(axis=-1).sum()) / (
            T * self.data_size
        )

        logqp0 = self._kl_initial(qz0_mean, qz0_logstd)

        return zs, xs, log_pxs, logqp0, logqp_path

    # ------------------------------------------------------------------
    # Prior sampling
    # ------------------------------------------------------------------

    def _solve_prior(
        self,
        ts: jax.Array,
        *,
        key,
        wave_cond: jax.Array,
        x0: jax.Array | None = None,
        dt: float | None = None,
        **kwargs,
    ) -> diffrax.Solution:
        bm_key, z0_key, x0_key = jr.split(key, 3)

        t0, t1 = ts[0], ts[-1]
        if self.g.diagonal:
            bm_shape = (self.data_size + self.latent_size,)
        else:
            bm_shape = (self.g.control_size,)
        bm = diffrax.UnsafeBrownianPath(shape=bm_shape, key=bm_key)

        if self.g.diagonal:

            def _g_diag(t, y, args):
                g_val = self._g_with_wave(t, y, args)
                g_aug = jnp.concatenate([jnp.zeros(self.data_size), g_val])
                return lineax.DiagonalLinearOperator(g_aug)

            diffusion_term = diffrax.ControlTerm(_g_diag, bm)
        else:

            def _g_mat(t, y, args):
                g_val = self._g_with_wave(t, y, args)
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
            args=wave_cond,
            max_steps=ts.shape[0] + 1,
            adjoint=diffrax.ForwardMode(),
        )

    def _solve_prior_for_loss(
        self,
        ts: jax.Array,
        *,
        key,
        wave_cond: jax.Array,
        x0: jax.Array,
        dt: float | None = None,
    ) -> jax.Array:
        """Prior solve usable inside a training loss (reverse-mode safe).

        Mirrors :meth:`_solve_prior` but uses ``VirtualBrownianTree`` and
        the default reverse-mode adjoint so gradients can flow back to
        the model parameters. ``x0`` is required (taken from the data
        batch so prior samples start from observed initial conditions).
        Returns ``xs`` only — no KL accumulator.
        """
        bm_key, z0_key = jr.split(key)

        t0, t1 = ts[0], ts[-1]
        if self.g.diagonal:
            bm_shape = (self.data_size + self.latent_size,)
        else:
            bm_shape = (self.g.control_size,)
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=self.dt / 2, shape=bm_shape, key=bm_key
        )

        if self.g.diagonal:

            def _g_diag(t, y, args):
                g_val = self._g_with_wave(t, y, args)
                g_aug = jnp.concatenate([jnp.zeros(self.data_size), g_val])
                return lineax.DiagonalLinearOperator(g_aug)

            diffusion_term = diffrax.ControlTerm(_g_diag, bm)
        else:

            def _g_mat(t, y, args):
                g_val = self._g_with_wave(t, y, args)
                return jnp.pad(g_val, ((self.data_size, 0), (0, 0)))

            diffusion_term = diffrax.ControlTerm(_g_mat, bm)

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(self.prior_drift),
            diffusion_term,
        )
        saveat = diffrax.SaveAt(ts=ts)

        z0 = self._sample_z0_prior(z0_key)
        y0 = jnp.concatenate([x0, z0])

        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt if dt is not None else self.dt,
            y0,
            saveat=saveat,
            args=wave_cond,
            max_steps=int(ts.shape[0] * 1.1),
        )
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        return sol.ys[..., : self.data_size]

    @eqx.filter_jit
    def sample_prior(
        self,
        ts: jax.Array,
        *,
        key,
        wave_cond: jax.Array,
        x0: jax.Array | None = None,
        unscale: bool = True,
        long_sample: bool = False,
        dt: float | None = None,
        return_latents: bool = False,
        **kwargs,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Sample from the prior SDE given sea-state conditioning.

        Args:
            ts: Time points, shape ``(T,)``.
            key: PRNG key.
            wave_cond: ``(n_wave_params,)`` sea-state vector.
            x0: Optional initial state (standardised).
            unscale: Return physical units if ``True``.
            long_sample: Allow trajectories > 50 000 steps.
            dt: Timestep override.
            return_latents: Also return the latent-state trajectory ``zs`` so the
                cause of divergent behaviour can be analysed post-hoc.

        Returns:
            Trajectory shape ``(T, data_size)``. When ``return_latents`` is True,
            returns ``(xs, zs)`` with ``zs`` of shape ``(T, latent_size)``.
        """
        if ts.shape[0] > 50000 and not long_sample:
            raise RuntimeError(
                f"Refusing to sample {ts.shape[0]} steps. Set long_sample=True."
            )
        sol = self._solve_prior(ts, key=key, wave_cond=wave_cond, x0=x0, dt=dt)
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs = sol.ys[..., : self.data_size]
        if unscale:
            xs = jax.vmap(self.unscale)(xs)
        if return_latents:
            return xs, sol.ys[..., self.data_size :]
        return xs

    @eqx.filter_jit
    def sample_prior_ode(
        self,
        ts: jax.Array,
        *,
        key,
        wave_cond: jax.Array,
        x0: jax.Array | None = None,
        z0: jax.Array | None = None,
        unscale: bool = True,
        dt: float | None = None,
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
            wave_cond: ``(n_wave_params,)`` sea-state vector.
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
            args=wave_cond,
            max_steps=ts.shape[0] + 1,
            adjoint=diffrax.ForwardMode(),
        )
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
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
        wave_cond: jax.Array,
        x0: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample from prior and return ``(xs, zs, h_vals, g_norms)``."""
        sol = self._solve_prior(ts, key=key, wave_cond=wave_cond, x0=x0)
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs = sol.ys[..., : self.data_size]
        zs = sol.ys[..., self.data_size :]

        def _eval_fields(y_t, t):
            x_t = y_t[..., : self.data_size]
            z_t = y_t[..., self.data_size :]
            h_val = self._eval_h(t, x_t, z_t, wave_cond)
            g_val = self._eval_g(t, x_t, z_t, wave_cond)
            g_norm = g_val if self.g.diagonal else jnp.sqrt(jnp.sum(g_val**2, axis=-1))
            return h_val, g_norm

        h_vals, g_norms = jax.vmap(_eval_fields)(sol.ys, ts)
        return xs, zs, h_vals, g_norms

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def sample_posterior(
        self,
        xs: jax.Array,
        ts: jax.Array,
        *,
        key,
        wave_cond: jax.Array,
        unscale: bool = True,
    ) -> jax.Array:
        """Sample from the posterior SDE conditioned on observations."""
        bm_key, z0_key = jr.split(key)

        ctx = self.encoder(xs)
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

        x0 = xs[0]
        z0, _, _ = self._sample_z0_posterior(ctx[0], z0_key)
        y0_aug = jnp.concatenate([x0, z0, jnp.zeros(1)])

        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            self.dt,
            y0_aug,
            saveat=saveat,
            args=(ts, ctx, wave_cond),
            max_steps=int(ts.shape[0] * 1.1),
            adjoint=diffrax.ForwardMode(),
        )
        if sol.ys is None:
            raise RuntimeError("Diffrax solution failed: ys is None.")
        xs_hat = sol.ys[..., : self.data_size]
        if unscale:
            xs_hat = jax.vmap(self.unscale)(xs_hat)
        return xs_hat

    # ------------------------------------------------------------------
    # Drift / diffusion
    # ------------------------------------------------------------------

    def _eval_h(
        self, t: jtp.ArrayLike, x: jax.Array, z: jax.Array, wave_cond: jax.Array
    ) -> jax.Array:
        """Evaluate h with either FiLM or concatenated wave conditioning."""
        if self._h_uses_film:
            xz = jnp.concatenate([x, z])
            return self.h(t, xz, wave_cond)
        else:
            xz_w = jnp.concatenate([x, z, wave_cond])
            return self.h(t, xz_w, None)

    def _eval_g(
        self, t: jtp.ArrayLike, x: jax.Array, z: jax.Array, wave_cond: jax.Array
    ) -> jax.Array:
        """Evaluate g with either FiLM or concatenated wave conditioning."""
        if self._g_uses_film:
            xz = jnp.concatenate([x, z])
            return self.g(t, xz, wave_cond)
        else:
            xz_w = jnp.concatenate([x, z, wave_cond])
            return self.g(t, xz_w, None)

    def prior_drift(self, t: jtp.ArrayLike, y: jax.Array, args) -> jax.Array:
        """Prior drift for ``[x, z]`` with wave conditioning in *args*."""
        wave_cond = args  # (n_wave_params,)
        x = y[..., : self.data_size]
        z = y[..., self.data_size :]

        # Physics in physical units → rescale
        x_phys = x * self.data_std + self.data_mean
        cz = self.C(z)
        if self.indirect_eta:
            cz = cz.at[: self.eta_size].set(0.0)
        dx = self.nominal_dynamics(x_phys) / self.data_std + cz

        dz = self._eval_h(t, x, z, wave_cond)
        if self.mean_reversion is not None:
            dz = dz + self.mean_reversion(z)
        return jnp.concatenate([dx, dz])

    def posterior_drift(self, t: jtp.ArrayLike, y: jax.Array, args) -> jax.Array:
        """Posterior drift with KL accumulation and wave conditioning."""
        ts, ctx, wave_cond = args
        x = y[..., : self.data_size]
        z = y[..., self.data_size : -1]
        xz_w = jnp.concatenate([x, z, wave_cond])

        z_f = self.f(t, xz_w, (ts, ctx))
        z_h = self._eval_h(t, x, z, wave_cond)
        g_val = self._eval_g(t, x, z, wave_cond)

        kl_rate = self._kl_rate(z_f, z_h, g_val)

        dz = z_f
        if self.mean_reversion is not None:
            dz = dz + self.mean_reversion(z)

        x_phys = x * self.data_std + self.data_mean
        cz = self.C(z)
        if self.indirect_eta:
            cz = cz.at[: self.eta_size].set(0.0)
        dx = self.nominal_dynamics(x_phys) / self.data_std + cz

        return jnp.concatenate([dx, dz, jnp.array([kl_rate])])

    def _g_with_wave(self, t: jtp.ArrayLike, y: jax.Array, args) -> jax.Array:
        """Evaluate diffusion g with wave conditioning (for prior sampling)."""
        wave_cond = args
        x = y[..., : self.data_size]
        z = y[..., self.data_size :]
        return self._eval_g(t, x, z, wave_cond)

    def diffusion(self, t: jtp.ArrayLike, y: jax.Array, args):
        """Diffusion for posterior ``[x, z, kl]``; wave_cond from *args*."""
        _, _, wave_cond = args
        y_state = y[..., :-1]
        x = y_state[..., : self.data_size]
        z = y_state[..., self.data_size :]
        g_val = self._eval_g(t, x, z, wave_cond)

        if self.g.diagonal:
            g_aug = jnp.concatenate(
                [jnp.zeros(self.data_size), g_val, jnp.array([0.0])]
            )
            return lineax.DiagonalLinearOperator(g_aug)
        else:
            return jnp.pad(g_val, ((self.data_size, 1), (0, 0)))

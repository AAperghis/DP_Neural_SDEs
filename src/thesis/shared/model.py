from abc import ABC, abstractmethod


import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from thesis.shared.data_structures import FieldConfig
from thesis.shared.utils import Encoder
from thesis.shared.vector_fields import AbstractVectorField, init_vector_field


class MeanReversion(eqx.Module):
    raw_rates: jax.Array  # shape (d,)
    mean: jax.Array  # shape (d,)

    def __init__(self, dim: int, init_rate: float = 0.1):
        # raw_rates are unconstrained; we map them to positive rates with softplus.
        # Use inverse-softplus to initialise near init_rate.
        self.raw_rates = jnp.log(jnp.expm1(jnp.full((dim,), init_rate)))
        self.mean = jnp.zeros((dim,))

    @property
    def rates(self):
        # strictly positive decay rates
        return jax.nn.softplus(self.raw_rates)

    def __call__(self, z):
        # z shape (..., d)
        return -(self.rates * (z - self.mean))


class SDE(eqx.Module, ABC):
    """Abstract base class for SDE models."""

    data_std: eqx.AbstractVar[jax.Array]
    data_mean: eqx.AbstractVar[jax.Array]

    @abstractmethod
    def __call__(
        self, ts: jax.Array, ys: jax.Array, *, key: jax.Array, log_sigma: jax.Array,  **kwargs
    ) -> tuple[jax.Array, ...]:
        """
        Evaluate the SDE drift and diffusion at time t, state y, and context.

        Args:
            ts: Time points corresponding to ys, shape (T,).
            ys: Observed data, shape (T, data_size).
            key: JAX random key for sampling.
            log_sigma: Log observation noise standard deviation, shape (data_size,).
            **kwargs: Additional arguments for specific SDE implementations.
        Returns:
            Tuple of JAX arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_prior(
        self, ts: jax.Array, *, key, unscale: bool = True, **kwargs
    ) -> jax.Array:
        """Sample from the SDE prior distribution."""
        raise NotImplementedError

    @abstractmethod
    def sample_posterior(
        self, ts: jax.Array, ys: jax.Array, *, key, unscale: bool = True, **kwargs
    ) -> jax.Array:
        """Sample from the SDE posterior distribution given observed data."""
        raise NotImplementedError

    def unscale(self, x: jax.Array) -> jax.Array:
        """Map standardised decoder output back to physical units."""
        return x * self.data_std + self.data_mean

    def trainable_filter(self):
        """Return a PyTree filter spec marking only trainable leaves.

        ``data_mean`` and ``data_std`` are excluded so the optimiser
        never updates the normalisation statistics.
        """
        filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)
        return eqx.tree_at(
            lambda m: (m.data_mean, m.data_std), filter_spec, (False, False)
        )


class AbstractHybridSDE(SDE):
    """Intermediate ABC for latent SDE models with shared structure.

    Provides common abstract variables and helper methods for models
    that use an encoder, latent prior/posterior, and KL divergence
    computation via Girsanov's theorem.
    """

    encoder: eqx.AbstractVar[Encoder]
    f: eqx.AbstractVar[AbstractVectorField]
    h: eqx.AbstractVar[AbstractVectorField]
    g: eqx.AbstractVar[AbstractVectorField]
    mean_reversion: eqx.AbstractVar[MeanReversion] | None
    qz0_posterior: eqx.AbstractVar[eqx.nn.Linear]
    pz0_mean: eqx.AbstractVar[jax.Array]
    pz0_logvar: eqx.AbstractVar[jax.Array]
    latent_size: eqx.AbstractVar[int]
    data_size: eqx.AbstractVar[int]
    dt: eqx.AbstractVar[float]
    solver: eqx.AbstractVar[diffrax.AbstractSolver]
    eps: eqx.AbstractVar[float]

    def _sample_z0_posterior(
        self, ctx_0: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample z0 from the posterior q(z0 | x).

        Returns:
            Tuple of (z0, qz0_mean, qz0_logstd).
        """
        qz0_mean, qz0_logstd = jnp.split(self.qz0_posterior(ctx_0), 2, axis=-1)
        qz0_logstd = jnp.clip(qz0_logstd, -5.0, 2.0)
        z0 = qz0_mean + jnp.exp(qz0_logstd) * jr.normal(key, shape=qz0_mean.shape)
        return z0, qz0_mean, qz0_logstd

    def _sample_z0_prior(self, key: jax.Array) -> jax.Array:
        """Sample z0 from the prior p(z0)."""
        pz0_std = jnp.exp(jnp.clip(self.pz0_logvar * 0.5, -5.0, 2.0))
        return self.pz0_mean + pz0_std * jr.normal(key, shape=self.pz0_mean.shape)

    def _kl_initial(
        self, qz0_mean: jax.Array, qz0_logstd: jax.Array
    ) -> jax.Array:
        """Analytic KL divergence KL(q(z0|x) || p(z0))."""
        pz0_logstd = jnp.clip(self.pz0_logvar * 0.5, -5.0, 2.0)
        kl_z0 = (
            pz0_logstd
            - qz0_logstd
            + (jnp.exp(2 * qz0_logstd) + (qz0_mean - self.pz0_mean) ** 2)
            / (2 * jnp.exp(2 * pz0_logstd))
            - 0.5
        )
        return kl_z0.sum(axis=-1)

    def _kl_rate(
        self, f_val: jax.Array, h_val: jax.Array, g_val: jax.Array
    ) -> jax.Array:
        """Instantaneous KL divergence rate via Girsanov's theorem.

        Computes 0.5 * (f - h)^T Sigma^{-1} (f - h) where Sigma = g g^T.
        Handles both diagonal and matrix-valued diffusions, including
        rank-deficient cases where control_size < latent_size.
        """
        delta = f_val - h_val
        eps = getattr(self, "_kl_eps", 1e-6)
        if self.g.diagonal:
            return 0.5 * jnp.sum(delta**2 / (g_val**2 + eps))
        else:
            A = g_val.T @ g_val
            C = A.shape[0]
            A_reg = A + eps * jnp.eye(C)
            v = g_val.T @ delta
            w = jnp.linalg.solve(A_reg, v)
            return 0.5 * jnp.dot(w, v)

    def _init_common(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        f_config: FieldConfig,
        h_config: FieldConfig,
        g_config: FieldConfig,
        dt: float,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """Initialise fields shared across all AbstractHybridSDE subclasses.

        Sets: data_size, latent_size, encoder, qz0_posterior,
        f, h, g, mean_reversion, pz0_mean, pz0_logvar, data_mean, data_std,
        eps, dt.

        Returns:
            dec_key: remaining JAX key for subclass-specific initialisation
                     (e.g. decoder MLP or linear control mapping).
        """
        enc_key, qz0_key, dec_key, mean_key, logvar_key = jr.split(key, 5)

        self.data_size = data_size
        self.latent_size = latent_size
        self.encoder = Encoder(data_size, hidden_size, context_size, key=enc_key)
        self.qz0_posterior = eqx.nn.Linear(context_size, 2 * latent_size, key=qz0_key)

        self._validate_field_configs(
            f_config, h_config, g_config, latent_size, context_size
        )

        self.f = init_vector_field(f_config)
        self.h = init_vector_field(h_config)
        self.g = init_vector_field(g_config)
        self.mean_reversion = (
            MeanReversion(latent_size) if h_config.mean_reversion else None
        )

        self.pz0_mean = jr.normal(mean_key, (latent_size,), dtype=jnp.float32) * 0.1
        self.pz0_logvar = jr.uniform(
            logvar_key, (latent_size,), minval=-0.5, maxval=0.5, dtype=jnp.float32
        )

        self.data_mean = jnp.zeros(data_size, dtype=jnp.float32)
        self.data_std = jnp.ones(data_size, dtype=jnp.float32)
        self.eps = 1e-6
        self.dt = dt

        return dec_key

    @staticmethod
    def _validate_field_configs(
        f_config: FieldConfig,
        h_config: FieldConfig,
        g_config: FieldConfig,
        latent_size: int,
        context_size: int,
    ) -> None:
        """Validate that field configs are consistent with model dimensions."""
        if f_config.context_size != context_size:
            raise ValueError(
                f"f_config context_size {f_config.context_size} does not match "
                f"encoder context_size {context_size}"
            )
        if f_config.latent_size != latent_size:
            raise ValueError(
                f"f_config latent_size {f_config.latent_size} does not match "
                f"latent_size {latent_size}"
            )
        if h_config.latent_size != latent_size:
            raise ValueError(
                f"h_config latent_size {h_config.latent_size} does not match "
                f"latent_size {latent_size}"
            )
        if g_config.latent_size != latent_size:
            raise ValueError(
                f"g_config latent_size {g_config.latent_size} does not match "
                f"latent_size {latent_size}"
            )

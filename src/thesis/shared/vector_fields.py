"""
Vector field modules for latent SDEs using Diffrax and Equinox.

Implements neural vector fields that can condition on time, state, and context for use in latent SDE models.

References:
    - https://github.com/patrick-kidger/equinox
    - https://github.com/patrick-kidger/diffrax

"""
import jax

import abc
from typing import Any, Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.typing as jtp

from thesis.shared.data_structures import FieldConfig, FieldType
from thesis.shared.utils import lipswish, construct


class AbstractVectorField(eqx.Module):
    """
    Abstract base class for vector fields.

    Subclasses must define all fields and ``__init__``, and implement ``__call__``.
    Follows the Equinox abstract/final pattern.
    """

    scale: eqx.AbstractVar[int | jax.Array]
    mlp: eqx.AbstractVar[eqx.nn.MLP | None]
    control_size: eqx.AbstractVar[int]
    diagonal: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        raise NotImplementedError

    def init_common(self, config: FieldConfig, *, key):
        """Shared init: sets diagonal, control_size, scale and returns (in_size, out_size, mlp_key)."""
        scale_key, mlp_key = jr.split(key)
        self.diagonal = config.diagonal
        in_size = (
            config.input_size if config.input_size is not None else config.latent_size
        )
        if config.diagonal:
            out_size = config.latent_size
            self.control_size = config.latent_size
        else:
            out_size = config.latent_size * config.control_size
            self.control_size = config.control_size

        if config.scale:
            if not config.diagonal and config.control_size > 1:
                scale_shape = (config.latent_size, config.control_size)
            else:
                scale_shape = (config.latent_size,)
            self.scale = jr.uniform(scale_key, scale_shape, minval=0.9, maxval=1.1)
        else:
            self.scale = 1

        return in_size, out_size, mlp_key

    def init_mlp(
        self, in_size: int, out_size: int, mlp_key: jax.Array, config: FieldConfig
    ):
        """Helper to init the MLP."""
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=config.hidden_layer_width,
            depth=config.depth,
            activation=self.get_activation(config.hidden_activation),
            final_activation=self.get_activation(config.final_activation),
            key=mlp_key,
        )

    def _format_output(self, out: jax.Array) -> jax.Array:
        """Reshape and scale the MLP output."""
        if not self.diagonal:
            out = out.reshape(-1, self.control_size)
            if self.control_size == 1:
                out = out.squeeze(axis=-1)
        return self.scale * out

    @staticmethod
    def get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
        match name.lower():
            case "lipswish":
                return lipswish
            case "silu" | "swish":
                return jnn.silu
            case "relu":
                return jnn.relu
            case "tanh":
                return jnp.tanh
            case "sigmoid":
                return jnn.sigmoid
            case "softplus":
                return jnn.softplus
            case _:
                raise ValueError(f"Unknown activation function: {name}")


class TimeStateField(AbstractVectorField):
    """
    Vector field that conditions on time and state.

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: int | jax.Array
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        super().__init__(**kwargs)
        in_size, out_size, mlp_key = self.init_common(config, key=key)
        self.mlp = self.init_mlp(in_size + 1, out_size, mlp_key, config)

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        """
        Evaluates the vector field at a given time and state.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Unused, for API compatibility.

        Returns:
            Output of the vector field (same shape as y).
        """
        t = jnp.asarray(t)
        out = self.mlp(jnp.concatenate([t[None], y]))
        return self._format_output(out)


class StateField(AbstractVectorField):
    """
    Vector field that conditions on state only.

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: int | jax.Array
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        super().__init__(**kwargs)
        in_size, out_size, mlp_key = self.init_common(config, key=key)
        self.mlp = self.init_mlp(in_size, out_size, mlp_key, config)

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        """
        Evaluates the vector field at a given state.

        Args:
            t: Time (unused, for API compatibility).
            y: State vector.
            args: Unused, for API compatibility.

        Returns:
            Output of the vector field (same shape as y).
        """
        out = self.mlp(y)
        return self._format_output(out)


class ContextTimeStateField(AbstractVectorField):
    """
    Vector field that conditions on time, state, and context (e.g. encoder output).

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: jax.Array | int
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs) -> None:
        super().__init__(**kwargs)
        in_size, out_size, mlp_key = self.init_common(config, key=key)
        self.mlp = self.init_mlp(
            in_size + config.context_size + 1, out_size, mlp_key, config
        )

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        """
        Evaluates the vector field at a given time, state, and context.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Tuple of (ts, ctx), where ts is a time array and ctx is a context array.

        Returns:
            Output of the vector field (same shape as y or output_size).
        """
        t = jnp.asarray(t)
        ts, ctx = args
        i = jnp.minimum(jnp.searchsorted(ts, t, side="right"), ts.shape[0] - 1)
        out = self.mlp(jnp.concatenate([t[None], y, ctx[i]]))
        return self._format_output(out)


class ContextStateField(AbstractVectorField):
    """
    Vector field that conditions on time, state, and context (e.g. encoder output).

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: jax.Array | int
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs) -> None:
        super().__init__(**kwargs)
        in_size, out_size, mlp_key = self.init_common(config, key=key)
        self.mlp = self.init_mlp(
            in_size + config.context_size, out_size, mlp_key, config
        )

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        """
        Evaluates the vector field at a given time, state, and context.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Tuple of (ts, ctx), where ts is a time array and ctx is a context array.

        Returns:
            Output of the vector field (same shape as y or output_size).
        """
        ts, ctx = args
        i = jnp.minimum(jnp.searchsorted(ts, t, side="right"), ts.shape[0] - 1)
        out = self.mlp(jnp.concatenate([y, ctx[i]]))
        return self._format_output(out)


class ConstantField(AbstractVectorField):
    """Learnable constant diffusion (independent of time, state, and context)."""

    scale: int | jax.Array
    mlp: None
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)
    value: jax.Array

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        super().__init__(**kwargs)
        self.diagonal = config.diagonal
        self.control_size = (
            config.latent_size if config.diagonal else config.control_size
        )
        self.scale = 1
        self.mlp = None

        if config.diagonal:
            shape = (config.latent_size,)
        elif config.control_size == 1:
            shape = (config.latent_size,)
        else:
            shape = (config.latent_size, config.control_size)

        self.value = jr.uniform(key, shape, minval=0.9, maxval=1.1)

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        return self.value


class FiLMStateField(AbstractVectorField):
    """State field with FiLM (Feature-wise Linear Modulation) conditioning.

    A small MLP maps a conditioning vector (e.g. wave parameters) to
    per-layer scale (gamma) and shift (beta) vectors.  These are applied
    after each hidden activation: ``h_i = gamma * h_i + beta``.

    The conditioning vector is **not** concatenated to the trunk input;
    it enters exclusively through FiLM.

    Call signature: ``__call__(t, y, args)`` where *args* is the
    conditioning vector (e.g. ``wave_cond``).
    """

    scale: int | jax.Array
    mlp: eqx.nn.MLP  # trunk network
    control_size: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)
    film_generator: eqx.nn.MLP  # conditioning → (gamma, beta) per layer

    n_hidden: int = eqx.field(static=True)
    hidden_width: int = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        super().__init__(**kwargs)
        if config.film_size is None:
            raise ValueError("film_size must be set for FILM_STATE fields")

        in_size, out_size, mlp_key = self.init_common(config, key=key)
        film_key, mlp_key = jr.split(mlp_key)

        # Trunk MLP — receives state only (no conditioning)
        self.mlp = self.init_mlp(in_size, out_size, mlp_key, config)

        # Number of hidden layers = depth (eqx.nn.MLP has depth+1 Linear layers)
        self.n_hidden = config.depth
        self.hidden_width = config.hidden_layer_width

        # FiLM generator: cond → (gamma, beta) for each hidden layer
        # Output size = n_hidden * hidden_width * 2 (gamma + beta per layer)
        film_out = self.n_hidden * config.hidden_layer_width * 2
        film_hidden = max(64, config.film_size * 4)
        self.film_generator = construct(
            eqx.nn.MLP,
            in_size=config.film_size,
            out_size=film_out,
            width_size=film_hidden,
            depth=2,
            activation=jnn.silu,
            key=film_key,
        )

    def __call__(self, t: jtp.ArrayLike, y: jax.Array, args: Any) -> jax.Array:
        """Evaluate the vector field with FiLM conditioning.

        Args:
            t: Time (unused).
            y: State vector (trunk input, no conditioning).
            args: Conditioning vector for FiLM (e.g. wave_cond).
        """
        cond = args  # (film_size,)

        # Generate all FiLM parameters at once
        film_params = self.film_generator(cond)
        # Reshape to (n_hidden, 2, hidden_width): [gamma, beta] per layer
        film_params = film_params.reshape(self.n_hidden, 2, self.hidden_width)

        # Manual forward pass through trunk with FiLM at each hidden layer
        x = y
        layers = self.mlp.layers
        activation = self.mlp.activation
        final_activation = self.mlp.final_activation

        for i in range(self.n_hidden):
            x = layers[i](x)
            x = activation(x)
            # FiLM: gamma * x + beta  (gamma centered at 1)
            gamma = 1.0 + film_params[i, 0]
            beta = film_params[i, 1]
            x = gamma * x + beta

        # Final layer (no FiLM, applies final_activation)
        x = layers[self.n_hidden](x)
        x = final_activation(x)

        return self._format_output(x)


def init_vector_field(
    config: FieldConfig,
    ) -> AbstractVectorField: 
    match config.field_type:
        case FieldType.CONTEXT_TIME_STATE:
            return construct(ContextTimeStateField, config, key=config.key)

        case FieldType.CONTEXT_STATE:
            return construct(ContextStateField, config, key=config.key)

        case FieldType.TIME_STATE:
            return construct(TimeStateField, config, key=config.key)

        case FieldType.STATE:
            return construct(StateField, config, key=config.key)

        case FieldType.CONSTANT:
            return construct(ConstantField, config, key=config.key)

        case FieldType.FILM_STATE:
            return construct(FiLMStateField, config, key=config.key)

        case _:
            raise ValueError(f"Unknown field type: {config.field_type}")


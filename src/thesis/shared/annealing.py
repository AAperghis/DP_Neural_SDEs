import jax
from thesis.shared.data_structures import AnnealConfig, AnnealStrategy, HyperParameters
import jax.numpy as jnp
import jax.typing as jtp

def linear(
    begin: int | float, w0: jtp.ArrayLike | float, w1: jtp.ArrayLike | float, end: int | float, t: float
) -> jax.Array:
    """Linear annealing from w0 to w1 between epochs begin and end."""
    w0, w1 = jnp.asarray(w0, dtype=jnp.float32), jnp.asarray(w1, dtype=jnp.float32)
    duration = jnp.maximum(end - begin, 1e-8)
    return jnp.where(
        end <= begin,
        jnp.where(t >= end, w1, w0),
        jnp.where(
            t < begin,
            w0,
            jnp.where(t > end, w1, w0 + (w1 - w0) * (t - begin) / duration),
        ),
    )


def cosine(
    begin: int | float, w0: jtp.ArrayLike | float, w1: jtp.ArrayLike | float, end: int | float, t: float
) -> jax.Array:
    """Cosine annealing from w0 to w1 between epochs begin and end."""
    w0, w1 = jnp.asarray(w0, dtype=jnp.float32), jnp.asarray(w1, dtype=jnp.float32)
    duration = jnp.maximum(end - begin, 1e-8)
    return jnp.where(
        end <= begin,
        jnp.where(t >= end, w1, w0),
        jnp.where(
            t < begin,
            w0,
            jnp.where(
                t > end,
                w1,
                w0 + 0.5 * (w1 - w0) * (1 - jnp.cos(jnp.pi * (t - begin) / duration)),
            ),
        ),
    )


def anneal(config: AnnealConfig, step: int) -> jax.Array:
    length = config.end - config.start
    begin = config.start + length * config.warmup
    match config.annealing_strategy:
        case AnnealStrategy.LINEAR:
            return linear(
                begin, config.initial_weight, config.final_weight, config.end, step
            )
        case AnnealStrategy.COSINE:
            return cosine(
                begin, config.initial_weight, config.final_weight, config.end, step
            )
        case _:
            raise ValueError(f"Unknown annealing strategy: {config.annealing_strategy}")


def _get_weight(config: AnnealConfig | list[AnnealConfig], step: int) -> jax.Array:
    """Resolve a single scalar schedule (one AnnealConfig or piecewise list)."""
    if isinstance(config, AnnealConfig):
        return anneal(config, step)
    # Piecewise: find the active segment, fall back to last segment's final value.
    for cfg in config:
        if cfg.start <= step < cfg.end:
            return anneal(cfg, step)
    return jnp.asarray(config[-1].final_weight, dtype=jnp.float32)


def anneal_params(hyperparams: HyperParameters, step: int):
    """Calculate annealed KL weight and noise log sigma for the current step.

    ``annealing`` may be:
    - A single :class:`AnnealConfig` → scalar log_sigma.
    - A flat ``list[AnnealConfig]`` → piecewise scalar schedule.
    """
    kl_config = hyperparams.training.kl_annealing
    noise_config = hyperparams.training.noise_annealing

    kl_weight = _get_weight(kl_config, step)
    log_sigma = _get_weight(noise_config, step)

    return kl_weight, log_sigma

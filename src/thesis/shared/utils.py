import subprocess as sp
import tempfile
import time
from pathlib import Path
from typing import Any, TypeVar

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.typing as jtp
import mlflow

from thesis.utils import AsyncLogger


"""
Utilities for the Diffrax Latent SDE project.

This module provides utility functions and classes for model training, logging, device management, and custom activation functions.

References:
- Equinox: https://github.com/patrick-kidger/equinox
- JAX: https://github.com/google/jax
- MLflow: https://mlflow.org/
"""

mlflow.enable_system_metrics_logging()


_ModuleT = TypeVar("_ModuleT", bound=eqx.Module)


def construct(cls: type[_ModuleT], *args: Any, **kwargs: Any) -> _ModuleT:
    """Instantiate an equinox ``Module`` while preserving its concrete type.

    ``ty`` resolves ``Module`` subclass construction through equinox's
    metaclass ``__call__``, whose return type widens to ``eqx.Module`` and
    discards the subclass. That makes every direct ``SubModule(...)`` call read
    as ``Module`` at type-check time, breaking precise return/assignment types.
    This wrapper restores the exact type with a single, centralised ``cast`` so
    call sites stay clean and correctly typed.
    """
    return cls(*args, **kwargs)


def lipswish(x: jtp.ArrayLike) -> jax.Array:
    """
    LiPSwish activation function.

    Args:
        x: Input array.

    Returns:
        Output array after applying LiPSwish activation.
    """
    return 0.909 * jnn.silu(x)


class Encoder(eqx.Module):
    """
    GRU-based encoder for sequential data.

    Encodes a sequence of observations into a context representation using a GRU cell followed by a linear layer.
    """

    gru: eqx.nn.GRUCell
    lin: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, data_size: int, hidden_size: int, ctx_size: int, *, key) -> None:
        """
        Initializes the Encoder.

        Args:
            data_size: Dimensionality of input data.
            hidden_size: Size of the GRU hidden state.
            ctx_size: Size of the output context vector.
            key: JAX PRNG key for initialization.
        """
        gru_key, lin_key = jr.split(key)
        self.hidden_size = hidden_size
        self.gru = eqx.nn.GRUCell(
            input_size=data_size, hidden_size=hidden_size, key=gru_key
        )
        self.lin = eqx.nn.Linear(hidden_size, ctx_size, key=lin_key)

    def __call__(self, xs: jtp.ArrayLike) -> jax.Array:
        """
        Encode a sequence of observations.

        Args:
            xs: Input sequence of shape (T, data_size).

        Returns:
            Encoded context of shape (T, ctx_size).
        """

        def step(h, x):
            h = self.gru(x, h)
            return h, self.lin(h)

        _, ctx = jax.lax.scan(step, jnp.zeros(self.hidden_size), xs, reverse=True)
        return ctx


def get_gpu_memory_usage() -> float | None:
    """
    Get current GPU memory usage in MB using nvidia-smi.

    Returns:
        The memory usage in MB, or None if unavailable.
    """
    try:
        result = sp.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return None


def get_gpu_utilization() -> float | None:
    """
    Get current GPU utilization percentage using nvidia-smi.

    Returns:
        The GPU utilization percentage, or None if unavailable.
    """
    try:
        result = sp.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return None


def log_summary(
    logger: AsyncLogger,
    step: int,
    model: eqx.Module,
    best_loss: float,
    window_metrics: list[dict[str, Any]],
    step_logs: list[tuple[int, dict[str, Any]]],
    train_start_time: float,
) -> float:
    """
    Log aggregated metrics over a window of recent steps and save best model.

    Args:
        logger: AsyncLogger instance for logging.
        step: Current step number.
        model: eqx.Module model instance.
        best_loss: Best loss value so far.
        window_metrics: List of per-step metric dicts for the window.
        step_logs: List of (step, metrics_dict) for intermediate logging.
        train_start_time: Wall time at training start.

    Returns:
        Updated best_loss value.
    """
    if window_metrics:
        keys = window_metrics[0].keys()
        means = {
            k: float(jnp.mean(jnp.stack([m[k] for m in window_metrics]))) for k in keys
        }

        if step_logs:
            for s, metrics in step_logs:
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items()},
                    step=s,
                )

        logger.log("-" * 50)
        summary = " | ".join(f"{k}: {v:.4f}" for k, v in means.items())
        logger.log(f"Step {step:07d} | {summary}")
        logger.log("=" * 50)

        logger.flush()

        # Log mean-reversion summary statistics
        mr = getattr(model, "mean_reversion", None)
        if mr is not None:
            rates = mr.rates
            mean = mr.mean
            mlflow.log_metrics(
                {
                    "mr_rate_min": float(jnp.min(rates)),
                    "mr_rate_max": float(jnp.max(rates)),
                    "mr_rate_mean": float(jnp.mean(rates)),
                    "mr_rate_std": float(jnp.std(rates)),
                    "mr_mean_min": float(jnp.min(mean)),
                    "mr_mean_max": float(jnp.max(mean)),
                    "mr_mean_mean": float(jnp.mean(mean)),
                    "mr_mean_std": float(jnp.std(mean)),
                },
                step=step,
            )

        mean_loss = means["loss"]
        if mean_loss < best_loss:
            best_loss = mean_loss
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / f"best_{step:07d}.eqx"
                eqx.tree_serialise_leaves(tmp_path, model)
                mlflow.log_metric("best_loss", best_loss, step=step)
                mlflow.log_artifact(str(tmp_path), artifact_path="best")
    return best_loss


def log_batch(
    logger: AsyncLogger,
    hyperparams,
    step: int,
    metrics: dict[str, Any],
    schedule,
    step_logs: list[tuple[int, dict[str, Any]]],
    interval_start_time: float,
    print: bool = True,
) -> float:
    """
    Log metrics and timing for a training step.

    Only blocks on JAX results every ``log_every`` steps to measure the true
    wall-clock mean batch time over the interval without slowing down every step.

    Args:
        logger: AsyncLogger instance for logging.
        hyperparams: Hyperparameters object.
        step: Current step number.
        metrics: Dict of metric name -> JAX scalar for this step.
        schedule: Learning rate schedule callable.
        step_logs: List to append step metrics to.
        interval_start_time: Wall-clock time at the start of this logging interval.

    Returns:
        Updated interval_start_time (reset after logging).
    """
    if step % hyperparams.training.log_every == 0 and step > 0:
        jax.block_until_ready(tuple(metrics.values()))
        now = time.perf_counter()
        n_steps = hyperparams.training.log_every
        mean_batch_time = (now - interval_start_time) / n_steps
        step_logs.append(
            (
                step,
                metrics
                | {
                    "learning_rate": schedule(step),
                    "mean_batch_time": mean_batch_time,
                },
            )
        )
        return now  # Reset interval
    return interval_start_time


def get_device(device: str, logger: AsyncLogger) -> Any:
    """
    Get JAX device based on user input and log device info.

    Args:
        device: Desired device platform ('gpu' or 'cpu').
        logger: AsyncLogger instance for logging.

    Returns:
        The selected JAX device.
    """
    cuda_available = any(d.platform == device for d in jax.devices())
    # if not cuda_available and device == "gpu":
    #     raise RuntimeError("Requested GPU device but no compatible GPU found.")

    device = jax.devices(device)[0]
    logger.log(f"CUDA available: {cuda_available}")
    logger.log(f"Using device: {device}")
    return device

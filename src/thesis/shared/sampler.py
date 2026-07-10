"""Base class and null implementation for per-step sample generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import jax.typing as jtp

from thesis.shared.data_structures import HyperParameters
from thesis.utils import AsyncLogger


class Sampler(ABC):
    """Abstract base for periodic sample generation during training.

    ``setup`` is called once by the training loop after the pools and run_id
    are available.  Constant arguments are stored as instance attributes so
    ``sample`` only receives what changes each step.
    """

    def setup(
        self,
        model,
        dataset,
        hyperparams: HyperParameters,
        compute_pool: ProcessPoolExecutor,
        log_thread: ThreadPoolExecutor,
        run_id: str,
    ) -> None:
        """Store per-run constants and precompute any reference caches.

        Called once inside the training loop before the first step.
        Subclasses should call ``super().setup(...)`` to store the shared
        attributes, then add their own precomputation.
        """
        self.hyperparams = hyperparams
        self.dataset = dataset
        self.compute_pool = compute_pool
        self.log_thread = log_thread
        self.run_id = run_id

    @abstractmethod
    def sample(
        self,
        model,
        step: int,
        step_key: jtp.ArrayLike,
        *,
        logger: AsyncLogger | None = None,
    ) -> None:
        """Generate and log samples for the current training step."""
        raise NotImplementedError


class NullSampler(Sampler):
    """No-op sampler used when periodic sampling is not needed."""

    def sample(
        self,
        model,
        step: int,
        step_key: jtp.ArrayLike,
        *,
        logger: AsyncLogger | None = None,
    ) -> None:
        pass

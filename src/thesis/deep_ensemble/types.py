"""Output data structures for deep-ensemble model sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelEnsemble:
    """Sampled output from a :class:`~thesis.deep_ensemble.loading.DeepEnsemble`.

    Attributes
    ----------
    data : np.ndarray, shape ``(M, N, T, F)``
        Sampled trajectories in physical units.
        *M* = number of ensemble members,
        *N* = trajectories per member,
        *T* = time steps,
        *F* = output features.
    time : np.ndarray, shape ``(T,)``
        Shared time vector in seconds.
    dt : float
        Timestep in seconds.
    feature_names : list[str]
        Feature column names matching axis 3 of *data*.
    member_ids : list[str]
        Identifier for each member (file stem or MLflow run ID).
    metadata : dict[str, Any]
        Optional key-value metadata (e.g. wave conditions used).
    latents : np.ndarray | None, shape ``(M, N, T, L)``
        Latent-state trajectories (``L`` = latent dimension) when sampling was run
        with ``return_latents=True``; otherwise ``None``. Kept so the cause of
        divergent behaviour can be analysed in latent space.
    """

    data: np.ndarray
    time: np.ndarray
    dt: float
    feature_names: list[str]
    member_ids: list[str]
    metadata: dict[str, Any]
    latents: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def M(self) -> int:
        """Number of ensemble members."""
        return self.data.shape[0]

    @property
    def N(self) -> int:
        """Number of trajectories per member."""
        return self.data.shape[1]

    @property
    def T(self) -> int:
        """Number of time steps."""
        return self.data.shape[2]

    @property
    def F(self) -> int:
        """Number of output features."""
        return self.data.shape[3]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """Reshape to ``(M * N, T, F)``.

        Useful when passing to statistics functions that expect a 2-D
        member axis, treating every (member, trajectory) pair as an
        independent realisation.
        """
        return self.data.reshape(self.M * self.N, self.T, self.F)

    def member(self, index: int) -> np.ndarray:
        """Return trajectories for a single member, shape ``(N, T, F)``."""
        return self.data[index]

    def truncate(self, t_start: float = 0.0, t_end: float | None = None) -> "ModelEnsemble":
        """Return a new :class:`ModelEnsemble` restricted to ``[t_start, t_end]``."""
        mask = self.time >= t_start
        if t_end is not None:
            mask &= self.time <= t_end
        idx = np.where(mask)[0]
        return ModelEnsemble(
            data=self.data[:, :, idx, :],
            time=self.time[idx],
            dt=self.dt,
            feature_names=self.feature_names,
            member_ids=self.member_ids,
            metadata=self.metadata,
        )

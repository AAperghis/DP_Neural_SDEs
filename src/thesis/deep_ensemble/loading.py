"""Deep-ensemble loader and sampling interface.

Supports loading :class:`~thesis.full_hybrid_sde.model.FullHybridSDE` and
:class:`~thesis.reduced_hybrid_sde.model.ReducedHybridSDE` members from
local ``.eqx`` checkpoint files or MLflow artifact stores, and provides a
unified ``sample_prior`` / ``sample_posterior`` interface that samples from
every member and returns a :class:`~thesis.deep_ensemble.types.ModelEnsemble`.

Typical usage
-------------
**Local disk** ::

    ensemble = DeepEnsemble.from_paths(
        ckpt_paths=["runs/member_0/final.eqx", ..., "runs/member_9/final.eqx"],
        hp_path="runs/hyperparams.json",   # single shared file is common
    )

**MLflow** ::

    ensemble = DeepEnsemble.from_mlflow(
        run_ids=["abc123", ..., "xyz789"],
    )

**Sampling** ::

    samples = ensemble.sample_prior(ts, n_trajectories=50, wave_cond=wave, key=jr.key(0))
    # samples.data  →  (N_members, 50, T, F)

    posterior = ensemble.sample_posterior(xs, ts, wave_cond=wave, key=jr.key(0))
    # posterior.data  →  (N_members, n_traj, T, F)
"""

from __future__ import annotations
from thesis.reduced_hybrid_sde.model import ReducedHybridSDE
from thesis.full_hybrid_sde.model import FullHybridSDE

from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from thesis.deep_ensemble.types import ModelEnsemble
from thesis.shared.data_structures import HyperParameters, FullOrderFeatures, hyperparams_from_json


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_full_order(hp: HyperParameters) -> bool:
    """Return ``True`` if *hp* describes a :class:`FullHybridSDE` model."""
    if not hp.data.features:
        raise ValueError("HyperParameters.data.features is empty; cannot detect model type.")
    return isinstance(hp.data.features[0], FullOrderFeatures)


def _build_template(hp: HyperParameters, key: jax.Array) -> FullHybridSDE | ReducedHybridSDE:
    """Instantiate a fresh (untrained) model from *hp*.

    Returns either a :class:`~thesis.full_hybrid_sde.model.FullHybridSDE`
    or :class:`~thesis.reduced_hybrid_sde.model.ReducedHybridSDE` depending
    on the feature enum encoded in *hp*.
    """
    if _is_full_order(hp):
        from thesis.full_hybrid_sde.train import init_model as fo_init_model
        return fo_init_model(hp, key)
    else:
        from thesis.reduced_hybrid_sde.train import init_model as ro_init_model
        return ro_init_model(hp, key)


def _resolve_artifact(source: str | Path, artifact_path: str) -> Path:
    """Download *artifact_path* from an MLflow run and return local path.

    ``source`` is treated as an MLflow run ID when it contains no path
    separator; otherwise it is returned unchanged (local file).
    """
    source = str(source)
    if "/" not in source and "\\" not in source:
        import mlflow
        local = mlflow.artifacts.download_artifacts(
            run_id=source, artifact_path=artifact_path
        )
        return Path(local)
    return Path(source)


def _feature_names(hp: HyperParameters) -> list[str]:
    """Extract string feature names from *hp* in order."""
    return [f.value if hasattr(f, "value") else str(f) for f in hp.data.features]


# ---------------------------------------------------------------------------
# Single-member loader
# ---------------------------------------------------------------------------

def load_member(
    ckpt_source: str | Path,
    hp_source: str | Path,
    key: jax.Array,
) -> tuple[FullHybridSDE | ReducedHybridSDE, HyperParameters]:
    """Load a single trained model from a checkpoint.

    Args:
        ckpt_source: Either a local path to a ``.eqx`` weights file, or an
            MLflow run ID (no path separator) — the artifact
            ``checkpoint/final.eqx`` will be downloaded automatically.
        hp_source: Either a local path to a ``hyperparams.json`` file, or an
            MLflow run ID — the artifact ``hyperparams.json`` will be
            downloaded.
        key: PRNG key used to initialise the template before loading weights.

    Returns:
        ``(model, hyperparams)`` — the loaded model and its parsed
        :class:`~thesis.shared.data_structures.HyperParameters`.
    """
    hp_path = _resolve_artifact(hp_source, "hyperparams.json")
    ckpt_path = _resolve_artifact(ckpt_source, "checkpoint/final.eqx")

    hp = hyperparams_from_json(str(hp_path))
    template = _build_template(hp, key)
    model = eqx.tree_deserialise_leaves(str(ckpt_path), template)
    return model, hp


# ---------------------------------------------------------------------------
# DeepEnsemble
# ---------------------------------------------------------------------------

class DeepEnsemble:
    """A collection of independently trained SDE models.

    Provides efficient batched sampling via :meth:`sample_prior` and
    :meth:`sample_posterior`.  All members must share the same architecture
    (same ``HyperParameters`` schema) but have different initialised weights.

    Args:
        models: List of loaded model instances.
        hyperparams: Shared hyper-parameters (architecture only; training
            config is ignored at inference time).
        member_ids: Human-readable identifier for each member (e.g. run ID or
            file stem).
    """

    def __init__(
        self,
        models: list[FullHybridSDE | ReducedHybridSDE],
        hyperparams: HyperParameters,
        member_ids: list[str],
    ) -> None:
        if len(models) != len(member_ids):
            raise ValueError(
                f"models and member_ids must have the same length, "
                f"got {len(models)} and {len(member_ids)}"
            )
        self.models = models
        self.hyperparams = hyperparams
        self.member_ids = member_ids
        self._feature_names = _feature_names(hyperparams)

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        model_cls = type(self.models[0]).__name__ if self.models else "?"
        return (
            f"DeepEnsemble(n_members={len(self)}, "
            f"model_type={model_cls}, "
            f"features={self._feature_names})"
        )

    # ------------------------------------------------------------------
    # Factory: local disk
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        ckpt_paths: Sequence[str | Path],
        hp_path: str | Path | Sequence[str | Path],
        *,
        keys: Sequence[jax.Array] | None = None,
        member_ids: Sequence[str] | None = None,
    ) -> "DeepEnsemble":
        """Load ensemble members from local ``.eqx`` checkpoint files.

        Args:
            ckpt_paths: Paths to each member's ``.eqx`` weights file.
            hp_path: Either a single ``hyperparams.json`` shared by all
                members, or a sequence of per-member paths (same length as
                *ckpt_paths*).
            keys: PRNG keys for template initialisation — one per member.
                If ``None``, keys are derived from ``jr.key(0)`` by folding
                in the member index.
            member_ids: Identifiers for each member.  Defaults to the file
                stem of each checkpoint path.

        Returns:
            The loaded ensemble.
        """
        ckpt_paths = [Path(p) for p in ckpt_paths]
        n = len(ckpt_paths)

        # Normalise hp_path to a per-member list
        if isinstance(hp_path, (str, Path)):
            hp_paths: list[Path] = [Path(hp_path)] * n
        else:
            hp_paths = [Path(p) for p in hp_path]
            if len(hp_paths) != n:
                raise ValueError(
                    f"hp_path sequence length ({len(hp_paths)}) must match "
                    f"ckpt_paths length ({n})"
                )

        if keys is None:
            base = jr.key(0)
            keys = [jr.fold_in(base, i) for i in range(n)]

        if member_ids is None:
            member_ids = [p.stem for p in ckpt_paths]

        models = []
        hp_ref: HyperParameters | None = None
        for ckpt, hp_p, key in zip(ckpt_paths, hp_paths, keys):
            model, hp = load_member(ckpt, hp_p, key)
            models.append(model)
            if hp_ref is None:
                hp_ref = hp

        if hp_ref is None:
            raise ValueError("No checkpoints provided; cannot determine hyperparameters.")
        return cls(models, hp_ref, list(member_ids))

    # ------------------------------------------------------------------
    # Factory: MLflow
    # ------------------------------------------------------------------

    @classmethod
    def from_mlflow(
        cls,
        run_ids: Sequence[str],
        *,
        artifact_ckpt: str = "checkpoint/final.eqx",
        artifact_hp: str = "hyperparams.json",
        keys: Sequence[jax.Array] | None = None,
    ) -> "DeepEnsemble":
        """Load ensemble members from MLflow artifact stores.

        Args:
            run_ids: MLflow run IDs, one per member.
            artifact_ckpt: Artifact path for the weights file within each run.
            artifact_hp: Artifact path for the hyperparameters file within
                each run.
            keys: PRNG keys for template initialisation.  Defaults to
                folding the member index into ``jr.key(0)``.

        Returns:
            The loaded ensemble.
        """
        import mlflow

        n = len(run_ids)
        if keys is None:
            base = jr.key(0)
            keys = [jr.fold_in(base, i) for i in range(n)]

        models = []
        hp_ref: HyperParameters | None = None

        for run_id, key in zip(run_ids, keys):
            hp_path = Path(
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_hp)
            )
            ckpt_path = Path(
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_ckpt)
            )
            hp = hyperparams_from_json(str(hp_path))
            template = _build_template(hp, key)
            model = eqx.tree_deserialise_leaves(str(ckpt_path), template)
            models.append(model)
            if hp_ref is None:
                hp_ref = hp
        if hp_ref is None:
            raise ValueError("No checkpoints provided; cannot determine hyperparameters.")
        return cls(models, hp_ref, list(run_ids))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _stacked_model(self) -> FullHybridSDE | ReducedHybridSDE:
        """Return one model PyTree with member array leaves stacked on axis 0.

        All members share a single architecture, so the static (non-array)
        structure is taken from the first member and only the array leaves are
        stacked.  The result is cached and consumed by
        :func:`equinox.filter_vmap`, so every member is evaluated in a single
        batched dispatch instead of a sequential Python loop.
        """
        cached = getattr(self, "_stacked_cache", None)
        if cached is None:
            arrays = [eqx.filter(m, eqx.is_array) for m in self.models]
            stacked_arrays = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves), *arrays
            )
            static = eqx.filter(self.models[0], eqx.is_array, inverse=True)
            cached = eqx.combine(stacked_arrays, static)
            self._stacked_cache = cached
        return cached

    def _member_keys(self, key: jax.Array, common_noise: bool) -> jax.Array:
        """Build an ``(N_members, ...)`` key array for the vmapped members.

        With *common_noise* every member shares *key* (identical Brownian
        paths); otherwise each member receives an independent split.
        """
        n = len(self.models)
        if common_noise:
            return jnp.broadcast_to(key, (n, *key.shape))
        return jr.split(key, n)

    def sample_prior(
        self,
        ts: jax.Array,
        n_trajectories: int,
        *,
        key: jax.Array,
        wave_cond: jax.Array | None = None,
        unscale: bool = True,
        common_noise: bool = False,
        **kwargs,
    ) -> ModelEnsemble:
        """Sample prior trajectories from every ensemble member.

        Args:
            ts: Time points, shape ``(T,)``.
            n_trajectories: Number of independent trajectories to draw per
                member.
            key: PRNG key; split across members and trajectories
                automatically.
            wave_cond: Sea-state conditioning vector, shape
                ``(n_wave_params,)``. Required for
                :class:`~thesis.full_hybrid_sde.model.FullHybridSDE` members;
                silently ignored by
                :class:`~thesis.reduced_hybrid_sde.model.ReducedHybridSDE`.
            unscale: Return trajectories in physical units when ``True``
                (default).
            common_noise: If ``True``, every member receives the same PRNG
                key so their Brownian paths are identical.  Differences in
                output are then solely due to model weights.  If ``False``
                (default), each member gets an independent key.

        Returns:
            A :class:`ModelEnsemble` whose ``data`` has shape
            ``(N_members, n_trajectories, T, F)``.
        """
        stacked = self._stacked_model()
        member_keys = self._member_keys(key, common_noise)

        # A 2-D ``wave_cond`` of shape ``(n_trajectories, n_wave_params)`` lets
        # callers batch several distinct sea states into a single vmapped call
        # (one conditioning vector per trajectory) instead of looping.
        batched_wc = wave_cond is not None and np.ndim(wave_cond) == 2

        @eqx.filter_vmap
        def _run(model, mkey):
            traj_keys = jr.split(mkey, n_trajectories)
            if batched_wc:
                return jax.vmap(
                    lambda k, wc: model.sample_prior(
                        ts, key=k, unscale=unscale, wave_cond=wc, **kwargs
                    )
                )(traj_keys, wave_cond)
            return jax.vmap(
                lambda k: model.sample_prior(
                    ts, key=k, unscale=unscale, wave_cond=wave_cond, **kwargs
                )
            )(traj_keys)

        out = _run(stacked, member_keys)
        if isinstance(out, (tuple, list)):
            data = np.asarray(out[0])  # (N, M, T, F)
            latents = np.asarray(out[1])
        else:
            data = np.asarray(out)  # (N, M, T, F)
            latents = None
        time_np = np.asarray(ts)
        dt = float(time_np[1] - time_np[0]) if len(time_np) > 1 else self.hyperparams.data.dt

        meta: dict = {}
        if wave_cond is not None:
            meta["wave_cond"] = np.asarray(wave_cond).tolist()

        return ModelEnsemble(
            data=data,
            time=time_np,
            dt=dt,
            feature_names=self._feature_names,
            member_ids=self.member_ids,
            metadata=meta,
            latents=latents,
        )

    def sample_prior_ode(
        self,
        ts: jax.Array,
        *,
        key: jax.Array,
        wave_cond: jax.Array | None = None,
        unscale: bool = True,
        common_noise: bool = True,
        mean_initial_state: bool = False,
        **kwargs,
    ) -> ModelEnsemble:
        """Integrate the prior drift ODE with no diffusion (deterministic).

        Removes the stochastic term entirely so the only source of spread
        between members is the learned drift function and initial-state
        prior parameters.  Useful for visualising the deterministic "mean
        path" each member has learned.

        Args:
            ts: Time points, shape ``(T,)``.
            key: PRNG key used to sample the initial state ``(x0, z0)`` for
                each member (ignored when *mean_initial_state* is ``True``).
            wave_cond: Sea-state conditioning vector, shape
                ``(n_wave_params,)``. Required for
                :class:`~thesis.full_hybrid_sde.model.FullHybridSDE`.
            unscale: Return trajectories in physical units when ``True``
                (default).
            common_noise: If ``True`` (default for ODE mode), every member
                uses the same key to sample its initial state, making the
                starting points as comparable as possible.  Ignored when
                *mean_initial_state* is ``True``.
            mean_initial_state: If ``True``, override each member's sampled
                ``(x0, z0)`` with the element-wise mean of ``px0_mean`` and
                ``pz0_mean`` across all members.  This pins the starting
                point to a single shared value so that the only source of
                spread between trajectories is the learned drift function.

        Returns:
            A :class:`ModelEnsemble` whose ``data`` has shape
            ``(N_members, 1, T, F)`` — one deterministic trajectory per
            member.
        """
        # Compute shared initial state if requested
        x0_override: jax.Array | None = None
        z0_override: jax.Array | None = None
        if mean_initial_state:
            x0_override = jnp.mean(
                jnp.stack([m.px0_mean for m in self.models], axis=0), axis=0
            )
            z0_override = jnp.mean(
                jnp.stack([m.pz0_mean for m in self.models], axis=0), axis=0
            )

        if common_noise and not mean_initial_state:
            member_keys = jnp.broadcast_to(key, (len(self.models), *key.shape))
        else:
            member_keys = jr.split(key, len(self.models))

        stacked = self._stacked_model()

        @eqx.filter_vmap
        def _run(model, mkey):
            return model.sample_prior_ode(
                ts,
                key=mkey,
                unscale=unscale,
                wave_cond=wave_cond,
                x0=x0_override,
                z0=z0_override,
                **kwargs,
            )  # (T, F)

        # (N, T, F) → insert the singleton trajectory axis → (N, 1, T, F)
        data = np.asarray(_run(stacked, member_keys))[:, np.newaxis]
        time_np = np.asarray(ts)
        dt = float(time_np[1] - time_np[0]) if len(time_np) > 1 else self.hyperparams.data.dt

        meta: dict = {}
        if wave_cond is not None:
            meta["wave_cond"] = np.asarray(wave_cond).tolist()

        return ModelEnsemble(
            data=data,
            time=time_np,
            dt=dt,
            feature_names=self._feature_names,
            member_ids=self.member_ids,
            metadata=meta,
        )

    def sample_posterior(
        self,
        xs: jax.Array,
        ts: jax.Array,
        *,
        key: jax.Array,
        wave_cond: jax.Array | None = None,
        unscale: bool = True,
        common_noise: bool = False
    ) -> ModelEnsemble:
        """Sample posterior trajectories conditioned on observations.

        Args:
            xs: Conditioning observations, shape ``(n_trajectories, T, F)``
                in standardised space.
            ts: Time points corresponding to *xs*, shape ``(T,)``.
            key: PRNG key; split across members and trajectories
                automatically.
            wave_cond: Sea-state conditioning vector, shape
                ``(n_wave_params,)``. Required for
                :class:`~thesis.full_hybrid_sde.model.FullHybridSDE`.
            unscale: Return trajectories in physical units when ``True``
                (default).
            common_noise: If ``True``, every member receives the same PRNG
                key so their Brownian paths are identical.  Differences in
                output are then solely due to model weights.  If ``False``
                (default), each member gets an independent key.

        Returns:
            A :class:`ModelEnsemble` whose ``data`` has shape
            ``(N_members, n_trajectories, T, F)``.
        """
        n_trajectories = xs.shape[0]
        stacked = self._stacked_model()
        member_keys = self._member_keys(key, common_noise)

        @eqx.filter_vmap
        def _run(model, mkey):
            traj_keys = jr.split(mkey, n_trajectories)
            return jax.vmap(
                lambda x_i, k: model.sample_posterior(
                    x_i, ts, key=k, unscale=unscale, wave_cond=wave_cond
                )
            )(xs, traj_keys)

        data = np.asarray(_run(stacked, member_keys))  # (N, M, T, F)
        time_np = np.asarray(ts)
        dt = float(time_np[1] - time_np[0]) if len(time_np) > 1 else self.hyperparams.data.dt

        meta: dict = {}
        if wave_cond is not None:
            meta["wave_cond"] = np.asarray(wave_cond).tolist()

        return ModelEnsemble(
            data=data,
            time=time_np,
            dt=dt,
            feature_names=self._feature_names,
            member_ids=self.member_ids,
            metadata=meta,
        )

"""Abstract base class for parquet-backed time series datasets.

Provides the shared public API (batching, windowing, standardisation,
wave conditioning, prefetching) used by both :class:`MultiFileDataset`
and :class:`ConsolidatedDataset`.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Iterator, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

# Batch type: (t, x, wave_cond, meta_list)
Batch = tuple[jax.Array, jax.Array, jax.Array, list[dict[str, Any]]]


class BaseParquetDataset(ABC):
    """Shared interface for parquet-backed time series datasets.

    Every batch is a 4-tuple ``(t, x, wave_cond, meta_list)`` where
    ``wave_cond`` has shape ``(B, n_wave_params)`` and is min-max
    scaled to [0, 1].  When wave parameters are unavailable a zero
    vector is returned.

    Subclasses must set the following attributes during ``__init__``:

    * ``dtype``
    * ``current_epoch``
    * ``series_length``
    * ``n_features``
    * ``wave_keys``
    * ``resample_every``
    * ``resample_dt``
    * ``group_scaling``
    * ``standardise``        - ``{"mean": ndarray, "std": ndarray}``
    * ``_wave_cond``         - ``dict[int, ndarray]``  (run/file id → scaled wc)
    * ``_wave_min``          - ``ndarray``
    * ``_wave_max``          - ``ndarray``
    * ``_sample_length``
    * ``n_per_series``
    * ``all_indices``        - ``list[tuple[int, int]]``

    And implement the abstract helpers listed below.
    """

    # Attributes set by subclasses (declared here for type checkers)
    dtype: jnp.dtype
    current_epoch: int
    series_length: int
    n_features: int
    wave_keys: list[str]
    angular_wave_keys: list[str]
    resample_every: int | None
    resample_dt: float | None
    group_scaling: bool
    standardise: dict[str, np.ndarray]
    _wave_cond: dict[int, np.ndarray]
    _wave_min: np.ndarray
    _wave_max: np.ndarray
    _sample_length: int
    n_per_series: int
    all_indices: list[tuple[int, int]]
    verbose: bool

    # ------------------------------------------------------------------
    # Abstract helpers that subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _n_series(self) -> int:
        """Return the number of runs / files."""

    @abstractmethod
    def _series_id(self, local_idx: int) -> int:
        """Map a local series index to the key used in ``_wave_cond``."""

    @abstractmethod
    def _get_array(self, local_idx: int) -> np.ndarray:
        """Return the (T, 1+F) array for the series at *local_idx*."""

    @abstractmethod
    def _get_meta(self, local_idx: int) -> dict[str, Any]:
        """Return per-series metadata dict (will be shallow-copied)."""

    @property
    @abstractmethod
    def files(self) -> list:
        """Return a list of file paths or run IDs (for logging / compat)."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_length(self) -> int:
        return self._sample_length

    @sample_length.setter
    def sample_length(self, value: int) -> None:
        self._sample_length = int(value)
        self.n_per_series = max(1, self.series_length // self._sample_length)
        self.all_indices = self._build_indices()

    @property
    def n_wave_params(self) -> int:
        """Width of the model-facing conditioning vector.

        Angular keys are expanded to a ``(cos, sin)`` pair, so they
        contribute two columns each; all other keys contribute one.
        """
        angular = set(getattr(self, "angular_wave_keys", ()) or ())
        return sum(2 if k in angular else 1 for k in self.wave_keys)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(self._n_windows(ri) for ri in range(self._n_series()))

    def steps_per_epoch(self, batch_size: int, drop_last: bool = False) -> int:
        total = len(self)
        n_full = total // batch_size
        return n_full if drop_last else n_full + (1 if total % batch_size else 0)

    def sample_random(self, key: jax.Array, n_samples: int) -> Batch:
        """Return ``(t, x, wave_cond, meta_list)``."""
        total = len(self)
        indices = jax.random.choice(key, total, shape=(n_samples,), replace=False)
        return self._load_batch([self.all_indices[i] for i in indices])

    @abstractmethod
    def split(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> tuple["BaseParquetDataset", "BaseParquetDataset"]:
        """Split into train/test datasets."""
        raise NotImplementedError

    def sample_full_trajectories(
        self, key: jax.Array, n_samples: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return full-length ``(ts, xs, wave_cond)``."""
        n = self._n_series()
        file_indices = jax.random.choice(
            key, n, shape=(n_samples,), replace=n_samples > n
        )
        file_indices = np.asarray(file_indices)

        xs_list, wc_list = [], []
        t = None
        for fi in file_indices:
            arr = self._get_array(int(fi))
            step = self._resample_step(arr)
            resampled = arr[::step]
            t = resampled[:, 0]
            x = resampled[:, 1:]
            x = (x - self.standardise["mean"]) / self.standardise["std"]
            xs_list.append(x)
            wc_list.append(self._wave_cond[self._series_id(int(fi))])

        min_len = min(x.shape[0] for x in xs_list)
        xs = np.stack([x[:min_len] for x in xs_list])
        assert t is not None
        t = np.round(t[:min_len] - t[0], 2).astype(np.float32)
        wave_cond = np.stack(wc_list)

        return jnp.array(t), jnp.array(xs), jnp.array(wave_cond)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        prefetch: int = 2,
        key: jax.Array | None = None,
        device: Any | None = None,
    ) -> Iterator[Batch]:
        """Yield ``(t, x, wave_cond, meta_list)`` batches."""
        if shuffle:
            if key is None:
                rng = np.random.default_rng()
                rng.shuffle(self.all_indices)
            else:
                perm = jax.random.permutation(key, len(self.all_indices))
                self.all_indices = [self.all_indices[i] for i in np.asarray(perm)]

        total = len(self.all_indices)
        n_full = total // batch_size
        n_batches = n_full if drop_last else n_full + (1 if total % batch_size else 0)

        def _gen():
            for b in range(n_batches):
                start = b * batch_size
                stop = min(start + batch_size, total)
                batch_idx = self.all_indices[start:stop]
                if len(batch_idx) < batch_size and drop_last:
                    continue
                t, x, wc, meta = self._load_batch(batch_idx)
                put = lambda a: (
                    jax.device_put(a, device) if device else jax.device_put(a)
                )
                yield put(t), put(x), put(wc), meta

        return _prefetch(_gen(), buffer_size=prefetch)

    def iter_steps(
        self,
        batch_size: int,
        num_steps: int,
        prefetch: int = 2,
        key: jax.Array | None = None,
        device: Any | None = None,
    ) -> Iterator[Batch]:
        """Yield exactly *num_steps* ``(t, x, wave_cond, meta)`` batches."""
        steps_emitted = 0

        def _infinite():
            nonlocal steps_emitted
            epoch = 0
            while steps_emitted < num_steps:
                epoch_key = jr.fold_in(key, epoch) if key is not None else None
                for batch in self.iter_batches(
                    batch_size,
                    shuffle=True,
                    drop_last=True,
                    prefetch=0,
                    key=epoch_key,
                    device=device,
                ):
                    if steps_emitted >= num_steps:
                        return
                    yield batch
                    steps_emitted += 1
                epoch += 1
                self.current_epoch = epoch

        return _prefetch(_infinite(), buffer_size=prefetch)

    def inverse_scale(self, states: jax.Array) -> jax.Array:
        """Undo z-score standardisation."""
        mean = jnp.asarray(self.standardise["mean"], dtype=states.dtype)
        std = jnp.asarray(self.standardise["std"], dtype=states.dtype)
        return states * std + mean

    def inverse_scale_wave_cond(self, wc: jax.Array) -> jax.Array:
        """Map model-facing wave conditioning back to physical units.

        Returns one column per entry in ``wave_keys`` (in order):
        non-angular keys are inverted from ``[0, 1]`` to physical units;
        angular keys (stored as a ``cos/sin`` pair) are reconstructed to
        an angle in radians via ``atan2(sin, cos)``.  The output always
        has ``len(wave_keys)`` columns, independent of the cos/sin
        expansion, preserving the ``[Hs, Tp, Dir]`` contract used by the
        conditioning metric and sea-state plots.
        """
        angular = set(getattr(self, "angular_wave_keys", ()) or ())
        na_min = jnp.asarray(self._wave_min, dtype=wc.dtype)
        na_max = jnp.asarray(self._wave_max, dtype=wc.dtype)
        cols = []
        in_i = 0  # column index into wc (model layout)
        na_i = 0  # index into _wave_min / _wave_max (non-angular only)
        for k in self.wave_keys:
            if k in angular:
                cos = wc[..., in_i]
                sin = wc[..., in_i + 1]
                cols.append(jnp.arctan2(sin, cos))
                in_i += 2
            else:
                cols.append(
                    wc[..., in_i] * (na_max[na_i] - na_min[na_i]) + na_min[na_i]
                )
                in_i += 1
                na_i += 1
        return jnp.stack(cols, axis=-1)

    # ------------------------------------------------------------------
    # Shared internal helpers
    # ------------------------------------------------------------------

    def _build_indices(self) -> list[tuple[int, int]]:
        idx = []
        for ri in range(self._n_series()):
            for wi in range(self._n_windows(ri)):
                idx.append((ri, wi))
        return idx

    def _n_windows(self, local_idx: int) -> int:
        """Number of non-overlapping windows in series *local_idx*.

        Defaults to the global ``n_per_series`` (derived from the first
        series).  Subclasses with per-series length info should override
        this to avoid emitting indices past the end of shorter series.
        """
        return self.n_per_series

    def _resample_step(self, arr: np.ndarray) -> int:
        if self.resample_every is not None:
            return int(self.resample_every)
        if self.resample_dt is not None:
            base_dt = float(arr[1, 0] - arr[0, 0])
            return max(1, int(round(float(self.resample_dt) / base_dt)))
        return 1

    def _window(self, arr: np.ndarray, win_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = win_idx * self._sample_length
        stop = start + self._sample_length
        chunk = arr[start:stop]

        step = self._resample_step(chunk)
        chunk = chunk[::step]

        t = chunk[:, 0]
        x = chunk[:, 1:]
        x = (x - self.standardise["mean"]) / self.standardise["std"]
        t = np.round(t - t[0], 2).astype(np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return t, x

    def _load_batch(self, batch_idx: Sequence[tuple[int, int]]) -> Batch:
        xs, ts, wcs, metas = [], [], [], []
        for local_idx, win_idx in batch_idx:
            arr = self._get_array(local_idx)
            t, x = self._window(arr, win_idx)
            xs.append(x)
            ts.append(t)
            sid = self._series_id(local_idx)
            wcs.append(self._wave_cond[sid])
            meta = self._get_meta(local_idx)
            meta["window_index"] = win_idx
            metas.append(meta)

        t0 = ts[0]
        x_btf = np.stack(xs, axis=0)
        wc = np.stack(wcs, axis=0)
        return (
            jnp.asarray(t0, dtype=self.dtype),
            jnp.asarray(x_btf, dtype=self.dtype),
            jnp.asarray(wc, dtype=self.dtype),
            metas,
        )

    def _init_wave_cond(
        self,
        metas_iter,
        wave_keys: list[str],
        angular_wave_keys: list[str] | None = None,
    ) -> None:
        """Compute scaled wave conditioning from metadata.

        Non-angular keys are min-max scaled to ``[0, 1]``.  Angular keys
        (e.g. wave direction, stored in radians) are encoded as a
        ``(cos θ, sin θ)`` pair instead — a wrap-safe, continuous
        representation occupying two columns.  The model-facing
        conditioning vector therefore has width
        ``len(wave_keys) + len(angular_wave_keys)``.

        Args:
            metas_iter: Iterable yielding ``(series_id, metadata_dict)`` pairs.
            wave_keys: Keys to extract from each metadata dict, in output order.
            angular_wave_keys: Subset of ``wave_keys`` to encode as
                ``(cos, sin)``.
        """
        angular = set(angular_wave_keys or ())
        non_angular = [k for k in wave_keys if k not in angular]

        raw: dict[int, dict[str, float]] = {}
        for sid, meta in metas_iter:
            raw[sid] = {k: float(meta.get(k, 0.0)) for k in wave_keys}

        # Min-max statistics over non-angular keys only.
        if raw and non_angular:
            stacked = np.array(
                [[vals[k] for k in non_angular] for vals in raw.values()],
                dtype=np.float32,
            )
            na_min = stacked.min(axis=0).astype(np.float32)
            na_max = stacked.max(axis=0).astype(np.float32)
        else:
            na_min = np.zeros(len(non_angular), dtype=np.float32)
            na_max = np.zeros(len(non_angular), dtype=np.float32)

        na_range = na_max - na_min
        na_range[na_range < 1e-8] = 1.0
        na_pos = {k: i for i, k in enumerate(non_angular)}

        self._wave_min = na_min
        self._wave_max = na_max

        self._wave_cond = {}
        for sid, vals in raw.items():
            vec: list[float] = []
            for k in wave_keys:
                if k in angular:
                    theta = vals[k]
                    vec.append(float(np.cos(theta)))
                    vec.append(float(np.sin(theta)))
                else:
                    i = na_pos[k]
                    vec.append(float((vals[k] - na_min[i]) / na_range[i]))
            self._wave_cond[sid] = np.asarray(vec, dtype=np.float32)
    
    @staticmethod
    def _compute_statistics_from_arrays(
        arrays,
        feat_names: list[str] | None,
        group_scaling: bool,
    ) -> dict[str, np.ndarray]:
        """Compute z-score stats from an iterable of (T, 1+F) arrays."""
        from thesis.shared.data_structures import scaling_groups_for_features

        all_feat = np.concatenate([arr[:, 1:] for arr in arrays], axis=0).astype(
            np.float64
        )
        mean = np.mean(all_feat, axis=0).astype(np.float32)
        std = np.std(all_feat, axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        result = {"mean": mean, "std": std}

        if group_scaling and feat_names:
            for _, idxs in scaling_groups_for_features(feat_names).items():
                g_mean = np.mean(result["mean"][idxs])
                g_std = np.sqrt(np.mean(result["std"][idxs] ** 2))
                if g_std < 1e-8:
                    g_std = 1.0
                for i in idxs:
                    result["mean"][i] = g_mean
                    result["std"][i] = g_std
        return result


# ------------------------------------------------------------------
# Prefetch utility
# ------------------------------------------------------------------


def _prefetch(it, buffer_size: int = 2):
    """Simple threaded prefetcher."""
    q: Queue = Queue(maxsize=max(1, buffer_size))
    sentinel = object()

    def _worker():
        try:
            for item in it:
                q.put(item)
        finally:
            q.put(sentinel)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

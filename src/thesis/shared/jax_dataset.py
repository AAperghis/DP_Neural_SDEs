"""Multi-file parquet dataset for JAX time series training.

Loads one parquet file per ensemble member / simulation run and provides
the same batching / windowing / wave-conditioning API as
:class:`~thesis.shared.consolidated_dataset.ConsolidatedDataset`.

Backwards-compatible alias :data:`JAXParquetDataset` is provided.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

import jax.numpy as jnp

from thesis.shared.base_dataset import BaseParquetDataset
from thesis.shared.data_structures import Features, FullOrderFeatures
import tqdm


class _LRUCache:
    """Simple LRU cache for numpy arrays."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: dict[int, np.ndarray] = {}
        self._order = collections.OrderedDict()

    def get(self, key: int, loader) -> np.ndarray:
        if key in self._data:
            self._order.move_to_end(key)
            return self._data[key]
        arr = loader()
        self._data[key] = arr
        self._order[key] = None
        if len(self._data) > self.capacity:
            k, _ = self._order.popitem(last=False)
            del self._data[k]
        return arr


class MultiFileDataset(BaseParquetDataset):
    """Parquet-backed dataset where each file is one ensemble member.

    Every batch yields ``(t, x, wave_cond, meta_list)`` -- the same
    4-tuple as :class:`~thesis.shared.consolidated_dataset.ConsolidatedDataset`.
    When wave parameters are not present in the file metadata, a zero
    vector is returned for ``wave_cond``.

    Parameters
    ----------
    files : sequence of Path
        One parquet file per run / ensemble member.
    columns : list, optional
        Feature columns to load.  ``'time'`` is always prepended.
    wave_keys : list[str], optional
        Metadata keys to extract as wave conditioning
        (default ``["Hs", "Tp", "beta_wave"]``).
    sample_length : int, optional
        Window length in raw timesteps (before resampling).
    resample_every : int, optional
        Keep every k-th row.
    resample_dt : float, optional
        Target dt; step is computed from the base dt in the data.
    standardise : bool
        Z-score standardise features.
    standardise_dict : dict, optional
        Pre-computed ``{"mean": ..., "std": ...}`` arrays.
    meta_key : str
        Schema metadata key for per-file JSON config.
    cache_size : int
        Number of files to cache in RAM (0 = no cache).
    dtype : jnp.dtype
        JAX dtype for returned arrays.
    truncate_seconds : float
        Discard this many seconds from the start of each file.
    group_scaling : bool
        Pool standardisation within scaling groups.
    n_runs : int | None
        If smaller than the number of (filtered) files, draw a random subset
        of that many files using a fixed seed.  ``None`` / <= 0 / >= available
        uses all files.
    filter_fn : callable, optional
        Predicate ``(meta_dict) -> bool`` applied per-file metadata; only files
        for which it returns ``True`` are kept.
    """

    def __init__(
        self,
        files: Sequence[Path],
        columns: list | None = None,
        wave_keys: list[str] | None = None,
        sample_length: int | None = None,
        resample_every: int | None = None,
        resample_dt: float | None = None,
        standardise: bool = True,
        standardise_dict: dict[str, np.ndarray] | None = None,
        meta_key: str = "run_params",
        cache_size: int = 0,
        dtype: jnp.dtype = jnp.float32,
        truncate_seconds: float = 0.0,
        group_scaling: bool = True,
        verbose: bool = False,
        n_runs: int | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        if len(files) == 0:
            raise AttributeError("File list must include at least one entry")

        self._files: list[Path] = list(files)

        # Optionally drop files by a metadata predicate, then (optionally) draw
        # a random subset.  ``n_runs`` None / <= 0 / >= available keeps all; a
        # smaller value samples randomly (fixed seed) rather than the first N,
        # so the wave-conditioning range is not biased.
        if filter_fn is not None or (n_runs is not None and n_runs > 0):
            def _meta(f: Path) -> dict[str, Any]:
                raw = pq.read_schema(f).metadata or {}
                kb = meta_key.encode("utf-8")
                return json.loads(raw[kb].decode("utf-8")) if kb in raw else {}

            kept = [
                f for f in self._files
                if filter_fn is None or filter_fn(_meta(f))
            ]
            if n_runs is not None and 0 < n_runs < len(kept):
                rng = np.random.default_rng(42)
                idx = np.sort(rng.choice(len(kept), size=n_runs, replace=False))
                kept = [kept[i] for i in idx]
            if not kept:
                raise ValueError("No files remain after filtering / subsampling")
            self._files = kept
        self.dtype = dtype
        self.truncate_seconds = truncate_seconds
        self.current_epoch: int = 0
        self.wave_keys = wave_keys or ["Hs", "Tp", "beta_wave"]
        self.angular_wave_keys: list[str] = []
        self.group_scaling = group_scaling
        self.verbose = verbose

        # Normalise columns
        if columns is not None:
            cols = [
                c.value if isinstance(c, (Features, FullOrderFeatures)) else c
                for c in columns
            ]
            if "time" in cols:
                cols.remove("time")
            self.columns: list[str] | None = ["time"] + cols
        else:
            self.columns = None

        if resample_every is not None and resample_dt is not None:
            raise ValueError("Provide only one of resample_every or resample_dt")
        self.resample_every = resample_every
        self.resample_dt = resample_dt

        # Read first file for shape info
        first = self._read_parquet(self._files[0], columns=self.columns)
        self.series_length = first.shape[0]
        self.n_features = first.shape[1] - 1

        self.cache_in_ram = len(self._files) <= cache_size and cache_size > 0
        self._lru = (
            _LRUCache(capacity=cache_size)
            if not self.cache_in_ram and cache_size > 0
            else None
        )
        self._cache: list[np.ndarray | None] = [None] * len(self._files)

        if self.cache_in_ram:
            for i, f in tqdm.tqdm(
                enumerate(self._files),
                desc="Caching files in RAM",
                disable=not self.verbose,
                total=len(self._files),
                unit="files",
            ):
                arr = self._read_parquet(f, columns=self.columns)
                self._cache[i] = np.ascontiguousarray(arr, dtype=np.float32)
            self._series_lengths: list[int] | None = [
                c.shape[0]
                for c in self._cache
                if c is not None
            ]
        else:
            self._series_lengths = None

        # Standardisation
        if standardise:
            if standardise_dict is not None:
                self.standardise = {
                    "mean": np.asarray(standardise_dict["mean"], dtype=np.float32),
                    "std": np.asarray(standardise_dict["std"], dtype=np.float32),
                }
            else:
                self.standardise = self._compute_statistics()
        else:
            self.standardise = {
                "mean": np.zeros(self.n_features, dtype=np.float32),
                "std": np.ones(self.n_features, dtype=np.float32),
            }

        # Per-file metadata
        self.metas: list[dict[str, Any]] = []
        for f in self._files:
            schema = pq.read_schema(f)
            meta = schema.metadata or {}
            key_b = meta_key.encode("utf-8")
            if key_b in meta:
                params = json.loads(meta[key_b].decode("utf-8"))
            else:
                params = {}
            self.metas.append(params)

        # Wave conditioning (min-max scaled to [0, 1])
        self._init_wave_cond(
            ((i, m) for i, m in enumerate(self.metas)),
            self.wave_keys,
        )

        # Windowing
        self._sample_length = 1
        self.n_per_series = 1
        self.all_indices: list[tuple[int, int]] = []
        self.sample_length = sample_length if sample_length else self.series_length

    # ------------------------------------------------------------------
    # ABC implementations
    # ------------------------------------------------------------------

    def _n_series(self) -> int:
        return len(self._files)

    def _series_id(self, local_idx: int) -> int:
        return local_idx

    def _get_array(self, local_idx: int) -> np.ndarray:
        return self._maybe_read(local_idx)

    def _n_windows(self, local_idx: int) -> int:
        if self._series_lengths is None:
            return self.n_per_series
        return max(1, self._series_lengths[local_idx] // self._sample_length)

    def _get_meta(self, local_idx: int) -> dict[str, Any]:
        m = self.metas[local_idx].copy()
        m["file_index"] = local_idx
        return m

    @property
    def files(self) -> list[Path]:
        return self._files

    @property
    def feature_names(self) -> list[str]:
        if self.columns is None:
            return [f"feat_{i}" for i in range(self.n_features)]
        return self.columns[1:]

    @property
    def _feat_names(self) -> list[str]:
        return self.feature_names

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> tuple["MultiFileDataset", "MultiFileDataset"]:
        """Split into train/test by file.  Test reuses train statistics."""
        n = len(self._files)
        n_test = max(1, int(round(n * test_fraction)))
        n_train = n - n_test

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_idx = sorted(perm[:n_train])
        test_idx = sorted(perm[n_train:])

        train_files = [self._files[i] for i in train_idx]
        test_files = [self._files[i] for i in test_idx]

        train_ds = MultiFileDataset(
            train_files,
            columns=self.columns,
            wave_keys=self.wave_keys,
            sample_length=self._sample_length,
            resample_every=self.resample_every,
            resample_dt=self.resample_dt,
            standardise=True,
            cache_size=len(train_files),
            truncate_seconds=self.truncate_seconds,
            group_scaling=self.group_scaling,
        )

        test_ds = MultiFileDataset(
            test_files,
            columns=self.columns,
            wave_keys=self.wave_keys,
            sample_length=self._sample_length,
            resample_every=self.resample_every,
            resample_dt=self.resample_dt,
            standardise=True,
            standardise_dict=train_ds.standardise,
            cache_size=len(test_files),
            truncate_seconds=self.truncate_seconds,
            group_scaling=self.group_scaling,
        )

        return train_ds, test_ds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_read(self, file_idx: int) -> np.ndarray:
        if self.cache_in_ram:
            cached = self._cache[file_idx]
            assert cached is not None
            return cached
        if self._lru is not None:
            return self._lru.get(
                file_idx,
                lambda: self._read_parquet(self._files[file_idx], columns=self.columns),
            )
        return self._read_parquet(self._files[file_idx], columns=self.columns)

    def _read_parquet(self, path: Path, columns: list[str] | None) -> np.ndarray:
        table = pq.read_table(path, columns=columns, memory_map=True)

        if columns is None:
            cols = table.column_names
            if "time" in cols and cols[0] != "time":
                time_idx = cols.index("time")
                take = [time_idx] + [i for i in range(len(cols)) if i != time_idx]
                table = table.select(take)

        arrays = [
            table.column(i).to_numpy(zero_copy_only=False)
            for i in range(table.num_columns)
        ]
        vals = np.column_stack(arrays).astype(np.float32, copy=False)

        if self.truncate_seconds > 0 and vals.shape[0] > 0:
            mask = vals[:, 0] >= (vals[0, 0] + self.truncate_seconds)
            vals = vals[mask]

        return np.ascontiguousarray(vals, dtype=np.float32)

    def _compute_statistics(self) -> dict[str, np.ndarray]:
        chunks: list[np.ndarray] = []
        if self.cache_in_ram:
            for cached in self._cache:
                if cached is not None:
                    chunks.append(cached)
        else:
            for f in self._files:
                chunks.append(self._read_parquet(f, columns=self.columns))

        return self._compute_statistics_from_arrays(
            chunks, self._feat_names, self.group_scaling
        )


# Backwards-compatible alias
JAXParquetDataset = MultiFileDataset

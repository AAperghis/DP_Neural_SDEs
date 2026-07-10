"""Consolidated parquet dataset for single-file training data.

Reads a single consolidated parquet file (created by
:mod:`thesis.shared.consolidate_parquet`) that contains a ``run_id``
column to distinguish ensemble members.

Backwards-compatible alias :data:`ConsolidatedParquetDataset` is provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

import jax.numpy as jnp

from thesis.shared.base_dataset import BaseParquetDataset
from thesis.shared.data_structures import (
    Features,
    FullOrderFeatures,
)


class ConsolidatedDataset(BaseParquetDataset):
    """Dataset backed by a single consolidated parquet file.

    Every batch yields ``(t, x, wave_cond, meta_list)`` where
    ``wave_cond`` has shape ``(B, n_wave_params)`` and is min-max
    scaled to [0, 1].

    Parameters
    ----------
    path : Path
        Path to the consolidated parquet file.
    columns : list, optional
        Feature columns to load.  ``"t"`` is always loaded for timestamps.
    wave_keys : list[str], optional
        Metadata keys to extract as wave conditioning
        (default ``["Hs", "Tp", "beta_wave"]``).
    sample_length : int | None
        Window length in raw timesteps (before resampling).
    resample_every : int | None
        Keep every k-th row.
    resample_dt : float | None
        Target dt; computes step from base dt in data.
    standardise : bool
        Whether to z-score standardise features.
    standardise_dict : dict | None
        Pre-computed ``{"mean": ..., "std": ...}`` arrays.
    truncate_seconds : float
        Drop this many seconds from the start of each run.
    group_scaling : bool
        Pool standardisation within scaling groups.
    n_runs : int | None
        If smaller than the number of available runs, draw a random subset of
        that many runs (after filtering) using a fixed seed for
        reproducibility.  ``None`` / <= 0 / >= the available count uses all
        runs.
    filter_fn : callable, optional
        Predicate ``(meta_dict) -> bool`` applied per-run metadata.
        Only runs for which *filter_fn* returns ``True`` are kept.
    meta_key : str
        Parquet schema key containing per-run metadata JSON.
    dtype : jnp.dtype
        JAX dtype for returned arrays.
    """

    def __init__(
        self,
        path: Path,
        columns: list | None = None,
        wave_keys: list[str] | None = None,
        angular_wave_keys: list[str] | None = None,
        sample_length: int | None = None,
        resample_every: int | None = None,
        resample_dt: float | None = None,
        standardise: bool = True,
        standardise_dict: dict[str, np.ndarray] | None = None,
        truncate_seconds: float = 0.0,
        group_scaling: bool = True,
        n_runs: int | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        meta_key: str = "run_metadata",
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self.truncate_seconds = truncate_seconds
        self.current_epoch: int = 0
        self.wave_keys = wave_keys or ["Hs", "Tp", "beta_wave"]
        self.angular_wave_keys = angular_wave_keys or []
        self.group_scaling = group_scaling

        if resample_every is not None and resample_dt is not None:
            raise ValueError("Provide only one of resample_every or resample_dt")
        self.resample_every = resample_every
        self.resample_dt = resample_dt

        # Normalize columns
        if columns is not None:
            cols = [
                c.value if isinstance(c, (Features, FullOrderFeatures)) else c
                for c in columns
            ]
            if "t" in cols:
                cols.remove("t")
            self.columns = ["t"] + cols
        else:
            self.columns = None

        # --- Read the consolidated parquet ---
        table = pq.read_table(self.path, columns=None)

        # Extract per-run metadata
        schema_meta = table.schema.metadata or {}
        meta_key_b = meta_key.encode("utf-8")
        if meta_key_b in schema_meta:
            raw_meta = json.loads(schema_meta[meta_key_b].decode("utf-8"))
        else:
            raw_meta = {}
        self.run_metadata: dict[int, dict] = {int(k): v for k, v in raw_meta.items()}

        # Get run_id column and split data by run
        run_ids = table.column("run_id").to_numpy()
        unique_runs = np.unique(run_ids)

        # Apply metadata filter
        if filter_fn is not None:
            unique_runs = np.array(
                [
                    rid
                    for rid in unique_runs
                    if int(rid) in self.run_metadata
                    and filter_fn(self.run_metadata[int(rid)])
                ]
            )

        # ``n_runs`` None / <= 0 / >= available -> use all; a smaller value
        # draws a *random* subset (fixed seed for reproducibility across
        # training and downstream evaluation).  Sampling randomly rather than
        # taking the first N avoids biasing the wave-conditioning range.
        if n_runs is not None and 0 < n_runs < len(unique_runs):
            rng = np.random.default_rng(42)
            unique_runs = np.sort(
                rng.choice(unique_runs, size=n_runs, replace=False)
            )

        # Select only requested columns (if any)
        if self.columns is not None:
            keep = [c for c in self.columns if c in table.column_names]
            if "run_id" not in keep:
                keep.append("run_id")
            table = table.select(keep)

        # Convert to numpy
        col_names = table.column_names
        arrays = [
            table.column(i).to_numpy(zero_copy_only=False)
            for i in range(table.num_columns)
        ]
        full_data = np.column_stack(arrays).astype(np.float32, copy=False)

        # Build column index map
        col_map = {name: i for i, name in enumerate(col_names)}

        # Reorder so time is col 0, then features in self.columns order
        if self.columns is not None:
            ordered = [col_map[c] for c in self.columns if c in col_map]
        else:
            ordered = [col_map[c] for c in col_names if c not in ("run_id",)]
            t_idx = col_map.get("t")
            if t_idx is not None and ordered[0] != t_idx:
                ordered.remove(t_idx)
                ordered.insert(0, t_idx)

        # Split by run and store as contiguous arrays (time + features)
        self._run_data: list[np.ndarray] = []
        self._run_ids: list[int] = []
        for rid in unique_runs:
            if int(rid) not in self.run_metadata and len(self.run_metadata) > 0:
                continue
            mask = run_ids == rid
            run_arr = full_data[mask][:, ordered]

            if self.truncate_seconds > 0 and run_arr.shape[0] > 0:
                t_col = run_arr[:, 0]
                keep = t_col >= (t_col[0] + self.truncate_seconds)
                run_arr = run_arr[keep]

            self._run_data.append(np.ascontiguousarray(run_arr, dtype=np.float32))
            self._run_ids.append(int(rid))

        if not self._run_data:
            raise ValueError("No valid runs found in consolidated file")

        self.series_length = self._run_data[0].shape[0]
        self.n_features = self._run_data[0].shape[1] - 1

        # Feature name list for scaling groups
        if self.columns is not None:
            self._feat_names = [c for c in self.columns if c != "t"]
        else:
            self._feat_names = [c for c in col_names if c not in ("run_id", "t")]

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

        # Wave conditioning (min-max scaled to [0, 1]; angular keys as cos/sin)
        self._init_wave_cond(
            ((rid, self.run_metadata.get(rid, {})) for rid in self._run_ids),
            self.wave_keys,
            self.angular_wave_keys,
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
        return len(self._run_data)

    def _series_id(self, local_idx: int) -> int:
        return self._run_ids[local_idx]

    def _get_array(self, local_idx: int) -> np.ndarray:
        return self._run_data[local_idx]

    def _n_windows(self, local_idx: int) -> int:
        return max(1, self._run_data[local_idx].shape[0] // self._sample_length)

    def _get_meta(self, local_idx: int) -> dict[str, Any]:
        rid = self._run_ids[local_idx]
        meta = self.run_metadata.get(rid, {}).copy()
        meta["run_id"] = rid
        return meta

    @property
    def files(self) -> list:
        """Compatibility shim -- returns list of run IDs."""
        return self._run_ids

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> tuple["ConsolidatedDataset", "ConsolidatedDataset"]:
        """Split into train/test by run.  Test reuses train statistics."""
        n = len(self._run_data)
        n_test = max(1, int(round(n * test_fraction)))

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_idx = sorted(perm[: n - n_test])
        test_idx = sorted(perm[n - n_test :])

        train_ds = self._subset(train_idx, standardise_dict=None)
        test_ds = self._subset(test_idx, standardise_dict=train_ds.standardise)
        return train_ds, test_ds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> dict[str, np.ndarray]:
        return self._compute_statistics_from_arrays(
            self._run_data, self._feat_names, self.group_scaling
        )

    def _subset(
        self,
        local_indices: Sequence[int],
        standardise_dict: dict[str, np.ndarray] | None,
    ) -> "ConsolidatedDataset":
        """Create a lightweight view over selected runs."""
        ds = object.__new__(ConsolidatedDataset)
        ds.path = self.path
        ds.dtype = self.dtype
        ds.truncate_seconds = self.truncate_seconds
        ds.current_epoch = 0
        ds.wave_keys = self.wave_keys
        ds.angular_wave_keys = self.angular_wave_keys
        ds.group_scaling = self.group_scaling
        ds.columns = self.columns
        ds.resample_every = self.resample_every
        ds.resample_dt = self.resample_dt
        ds._feat_names = self._feat_names

        ds._run_data = [self._run_data[i] for i in local_indices]
        ds._run_ids = [self._run_ids[i] for i in local_indices]
        ds.series_length = ds._run_data[0].shape[0] if ds._run_data else 0
        ds.n_features = self.n_features
        ds.run_metadata = {rid: self.run_metadata.get(rid, {}) for rid in ds._run_ids}

        if standardise_dict is not None:
            ds.standardise = {
                "mean": np.asarray(standardise_dict["mean"], dtype=np.float32),
                "std": np.asarray(standardise_dict["std"], dtype=np.float32),
            }
        else:
            ds.standardise = ds._compute_statistics()

        ds._wave_cond = {rid: self._wave_cond[rid] for rid in ds._run_ids}
        ds._wave_min = self._wave_min.copy()
        ds._wave_max = self._wave_max.copy()

        ds._sample_length = self._sample_length
        ds.n_per_series = self.n_per_series
        ds.all_indices = ds._build_indices()
        return ds


# Backwards-compatible alias
ConsolidatedParquetDataset = ConsolidatedDataset

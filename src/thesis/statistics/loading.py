"""Ensemble loader and feature registry for DP time series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from thesis.shared.data_handling import read_consolidated_metadata, read_metadata
from thesis.shared.data_structures import DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------


def _read_parquet(path: Path, columns: list[str]) -> np.ndarray:
    """Read selected columns from a parquet file → contiguous float32 array."""
    table = pq.read_table(path, columns=columns, memory_map=True)
    arrays = [
        table.column(i).to_numpy(zero_copy_only=False) for i in range(table.num_columns)
    ]
    return np.ascontiguousarray(np.column_stack(arrays), dtype=np.float32)


# ---------------------------------------------------------------------------
# Ensemble loader
# ---------------------------------------------------------------------------


@dataclass
class Ensemble:
    """Container for a loaded time series ensemble.

    Attributes:
        data: Feature values in physical units, shape ``(N, T, F)``.
        time: Shared time vector in seconds, shape ``(T,)``.
        dt: Timestep in seconds (derived from *time*).
        feature_names: Ordered feature column names matching axis 2 of
            *data*.
        metadata: Per-file simulation metadata.
    """

    data: np.ndarray
    time: np.ndarray
    dt: float
    feature_names: list[str]
    metadata: list[dict[str, Any]]

    @property
    def N(self) -> int:
        return self.data.shape[0]

    @property
    def T(self) -> int:
        return self.data.shape[1]

    @property
    def F(self) -> int:
        return self.data.shape[2]

    def truncate(self, t_start: float = 0.0, t_end: float | None = None) -> "Ensemble":
        """Return a new Ensemble restricted to [t_start, t_end]."""
        mask = self.time >= t_start
        if t_end is not None:
            mask &= self.time <= t_end
        idx = np.where(mask)[0]
        return Ensemble(
            data=self.data[:, idx, :],
            time=self.time[idx] - self.time[idx[0]],
            dt=self.dt,
            feature_names=self.feature_names,
            metadata=self.metadata,
        )

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    def select_features(self, names: Sequence[str]) -> "Ensemble":
        """Return a new Ensemble with only the requested features."""
        idx = [self.feature_index(n) for n in names]
        return Ensemble(
            data=self.data[:, :, idx],
            time=self.time,
            dt=self.dt,
            feature_names=list(names),
            metadata=self.metadata,
        )


def _load_consolidated(
    path: Path,
    features: list[str],
    resample_step: int | None,
    t_warmup: float,
    t_name: str,
) -> Ensemble:
    """Load a consolidated parquet (with ``run_id`` column) into an Ensemble."""
    columns = ["run_id", t_name] + list(features)
    table = pq.read_table(path, columns=columns, memory_map=True)
    run_ids = table.column("run_id").to_numpy(zero_copy_only=False)
    arr = np.ascontiguousarray(
        np.column_stack(
            [
                table.column(c).to_numpy(zero_copy_only=False)
                for c in [t_name] + list(features)
            ]
        ),
        dtype=np.float32,
    )

    # Per-run metadata from schema
    run_meta_dict = read_consolidated_metadata(path)

    unique_ids = np.unique(run_ids)
    all_data: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    time_vec: np.ndarray | None = None

    for rid in sorted(unique_ids):
        mask = run_ids == rid
        chunk = arr[mask]  # (T_raw, 1+F)
        t = chunk[:, 0]
        x = chunk[:, 1:]

        if resample_step is not None and resample_step > 1:
            t = t[::resample_step]
            x = x[::resample_step]

        if t_warmup > 0:
            keep = t >= t_warmup
            t = t[keep]
            x = x[keep]

        t = t - t[0]

        if time_vec is None:
            time_vec = t
        else:
            if len(t) != len(time_vec):
                min_len = min(len(t), len(time_vec))
                t = t[:min_len]
                x = x[:min_len]
                time_vec = time_vec[:min_len]

        if np.isfinite(x).all():
            all_data.append(x)
            all_meta.append(run_meta_dict.get(str(int(rid)), {}))
        else:
            print(f"Warning: Skipping run_id {rid} due to non-finite values.")

    min_len = min(d.shape[0] for d in all_data)
    data = np.stack([d[:min_len] for d in all_data], axis=0)
    assert time_vec is not None
    time_vec = time_vec[:min_len]
    dt = float(time_vec[1] - time_vec[0]) if len(time_vec) > 1 else 0.05

    return Ensemble(
        data=data,
        time=time_vec,
        dt=dt,
        feature_names=list(features),
        metadata=all_meta,
    )


def load_ensemble(
    files: Sequence[Path],
    features: list[str] | None = None,
    resample_step: int | None = None,
    t_warmup: float = 600.0,
    *,
    t_name: str = "time",
    meta_key: str = "run_params",
) -> Ensemble:
    """Load parquet files into an Ensemble in physical units.

    Args:
        files: Parquet files to load (one per ensemble member).  A single
            consolidated parquet file (containing a ``run_id`` column) is
            also accepted.
        features: Feature column names to load.  Defaults to all 12
            standard features.
        resample_step: Keep every *resample_step*-th row (after time column
            extraction).
        t_warmup: Seconds to discard from the start of each trajectory.

    Returns:
        Loaded data with shared time vector and metadata.
    """
    if features is None:
        features = DEFAULT_FEATURES

    # Detect single consolidated parquet file
    if len(files) == 1:
        p = Path(files[0])
        schema = pq.read_schema(p)
        if "run_id" in schema.names:
            return _load_consolidated(p, features, resample_step, t_warmup, t_name)

    columns = [t_name] + list(features)

    all_data: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    time_vec: np.ndarray | None = None

    for path in files:
        arr = _read_parquet(Path(path), columns)  # (T_raw, 1+F)
        meta = read_metadata(Path(path), meta_key=meta_key)
        t = arr[:, 0]
        x = arr[:, 1:]

        # Resample
        if resample_step is not None and resample_step > 1:
            t = t[::resample_step]
            x = x[::resample_step]

        # Warm-up truncation
        if t_warmup > 0:
            keep = t >= t_warmup
            t = t[keep]
            x = x[keep]

        # Normalise time to start at 0
        t = t - t[0]

        if time_vec is None:
            time_vec = t
        else:
            if len(t) != len(time_vec):
                min_len = min(len(t), len(time_vec))
                t = t[:min_len]
                x = x[:min_len]
                time_vec = time_vec[:min_len]
        if np.isfinite(x).all():
            all_data.append(x)
            all_meta.append(meta)
        else:
            print(f"Warning: Skipping file {path.stem} due to non-finite values.")

    # Trim all to the same length (safety for edge cases)
    min_len = min(d.shape[0] for d in all_data)
    data = np.stack([d[:min_len] for d in all_data], axis=0)  # (N, T, F)
    assert time_vec is not None
    time_vec = time_vec[:min_len]
    dt = float(time_vec[1] - time_vec[0]) if len(time_vec) > 1 else 0.05

    return Ensemble(
        data=data,
        time=time_vec,
        dt=dt,
        feature_names=list(features),
        metadata=all_meta,
    )

"""Tests for parquet consolidation and the training dataset."""

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pyarrow.parquet as pq
import pytest

from thesis.shared.consolidated_dataset import ConsolidatedDataset
from thesis.shared.data_structures import FO_3DOF_FEATURES

N_FEATURES = len(FO_3DOF_FEATURES)
SAMPLE_LENGTH = 20  # window length in 0.5 s steps


def test_consolidate_merges_runs_and_metadata(consolidated_path, sim_outputs):
    table = pq.read_table(consolidated_path)

    assert "run_id" in table.column_names
    run_ids = np.unique(table.column("run_id").to_numpy())
    assert len(run_ids) == len(sim_outputs)

    meta = json.loads(table.schema.metadata[b"run_metadata"].decode("utf-8"))
    assert set(meta.keys()) == {str(int(r)) for r in run_ids}
    hs_in_meta = sorted(m["Hs"] for m in meta.values())
    hs_expected = sorted(cfg.Hs for cfg, _, _ in sim_outputs)
    assert hs_in_meta == hs_expected


@pytest.fixture(scope="module")
def dataset(consolidated_path) -> ConsolidatedDataset:
    return ConsolidatedDataset(
        consolidated_path,
        columns=FO_3DOF_FEATURES,
        wave_keys=["Hs", "Tp", "beta_wave"],
        sample_length=SAMPLE_LENGTH,
        resample_dt=0.5,
        standardise=True,
        group_scaling=True,
    )


def test_dataset_windows_and_shapes(dataset, sim_outputs):
    n_runs = len(sim_outputs)
    windows_per_run = dataset.series_length // SAMPLE_LENGTH
    assert len(dataset) == n_runs * windows_per_run

    t, x, wave_cond, meta = dataset.sample_random(jr.key(0), 4)
    assert t.shape == (SAMPLE_LENGTH,)
    assert x.shape == (4, SAMPLE_LENGTH, N_FEATURES)
    assert wave_cond.shape == (4, 3)
    assert len(meta) == 4

    assert bool(jnp.all(jnp.isfinite(x)))
    # Time axis starts at zero with the requested resampled dt
    assert t[0] == 0.0
    np.testing.assert_allclose(np.diff(np.asarray(t)), 0.5, atol=1e-6)
    # Wave conditioning is min-max scaled to [0, 1]
    assert bool(jnp.all((wave_cond >= 0.0) & (wave_cond <= 1.0)))


def test_dataset_standardisation(dataset):
    stats = dataset.standardise
    assert stats["mean"].shape == (N_FEATURES,)
    assert stats["std"].shape == (N_FEATURES,)
    assert np.all(np.isfinite(stats["mean"]))
    assert np.all(stats["std"] > 0.0)

    # Standardised data should be in a sane range for such short runs
    _, x, _, _ = dataset.sample_random(jr.key(1), 4)
    assert float(jnp.max(jnp.abs(x))) < 100.0


def test_dataset_iter_steps_batches(dataset):
    batches = list(dataset.iter_steps(batch_size=2, num_steps=3, key=jr.key(2)))
    assert len(batches) == 3
    for t, x, wave_cond, meta in batches:
        assert t.shape == (SAMPLE_LENGTH,)
        assert x.shape == (2, SAMPLE_LENGTH, N_FEATURES)
        assert wave_cond.shape == (2, 3)
        assert len(meta) == 2

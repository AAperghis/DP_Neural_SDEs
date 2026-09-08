"""Shared fixtures for the pipeline tests.

Two short full-order DP simulations are run once per session and reused
by the parquet, consolidation, dataset, and training tests.
"""

import os

# Force CPU and stop JAX grabbing GPU memory before jax is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import numpy as np
import pytest

from thesis.full_order_dp.simulation import (
    SimConfig,
    results_to_parquet,
    simulate_osv,
)
from thesis.shared.consolidate_parquet import consolidate

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIFT_COEFFS_PATH = REPO_ROOT / "src/thesis/full_order_dp/data/drift_coefficients.npz"

# Short but non-trivial: 61 timesteps at the production timestep of 0.5 s.
SIM_T_FINAL = 30.0
SIM_H = 0.5


def make_sim_configs() -> list[SimConfig]:
    return [
        SimConfig(
            T_final=SIM_T_FINAL,
            h=SIM_H,
            Hs=2.0,
            Tp=9.0,
            beta_wave=np.deg2rad(30.0),
            alloc_dynamic=True,
            wave_seed=0,
        ),
        SimConfig(
            T_final=SIM_T_FINAL,
            h=SIM_H,
            Hs=3.5,
            Tp=11.0,
            beta_wave=np.deg2rad(150.0),
            alloc_dynamic=True,
            wave_seed=1,
        ),
    ]


@pytest.fixture(scope="session")
def sim_outputs() -> list[tuple[SimConfig, dict, bool]]:
    """Run two short wave simulations (the data generation pipeline core)."""
    assert DRIFT_COEFFS_PATH.exists(), f"Missing {DRIFT_COEFFS_PATH}"
    outputs = []
    for cfg in make_sim_configs():
        results, success = simulate_osv(cfg, drift_coeffs_path=DRIFT_COEFFS_PATH)
        outputs.append((cfg, results, success))
    return outputs


@pytest.fixture(scope="session")
def raw_parquet_dir(sim_outputs, tmp_path_factory) -> Path:
    """Per-run parquet files as written by gen_training_data."""
    out_dir = tmp_path_factory.mktemp("raw_parquets")
    for cfg, results, success in sim_outputs:
        assert success, "Simulation failed; cannot build parquet fixtures"
        results_to_parquet(results, cfg, path=out_dir)
    return out_dir


@pytest.fixture(scope="session")
def consolidated_path(raw_parquet_dir, tmp_path_factory) -> Path:
    """Single consolidated parquet file as used for training."""
    out = tmp_path_factory.mktemp("consolidated") / "consolidated.parquet"
    return consolidate(raw_parquet_dir, out)

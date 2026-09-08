"""Tests for the full-order data generation pipeline."""

import json

import numpy as np
import pyarrow.parquet as pq

from thesis.full_order_dp.sample_params import sample_params
from thesis.full_order_dp.simulation import (
    SimConfig,
    results_to_parquet,
    simulate_osv,
)

from conftest import SIM_H, SIM_T_FINAL

EXPECTED_KEYS = [
    "t",
    "eta",
    "nu",
    "n_cmd",
    "alpha_cmd",
    "n_actual",
    "alpha_actual",
    "tau_cmd",
    "tau_thr",
    "tau_wave",
]


def test_simulate_osv_wave_runs(sim_outputs):
    n_steps = int(SIM_T_FINAL / SIM_H) + 1
    for cfg, results, success in sim_outputs:
        assert success
        for key in EXPECTED_KEYS:
            assert key in results, f"missing output key {key}"
        assert results["t"].shape == (n_steps,)
        assert results["eta"].shape == (n_steps, 6)
        assert results["nu"].shape == (n_steps, 6)
        assert results["n_actual"].shape == (n_steps, 4)
        assert results["alpha_actual"].shape == (n_steps, 2)
        assert np.all(np.isfinite(results["eta"]))
        assert np.all(np.isfinite(results["nu"]))
        # DP holds station: excursions over 30 s stay bounded
        assert np.max(np.abs(results["eta"][:, :2])) < 20.0
        # With waves active there must be non-zero wave forcing
        assert np.any(results["tau_wave"] != 0.0)


def test_simulate_osv_calm_water_stays_at_setpoint():
    cfg = SimConfig(T_final=10.0, h=0.5, Hs=0.0, Vc=0.0)
    results, success = simulate_osv(cfg)
    assert success
    # No waves, no current: vessel should barely move off the origin
    assert np.max(np.abs(results["eta"][:, :2])) < 0.5
    assert np.allclose(results["tau_wave"], 0.0)


def test_results_to_parquet_roundtrip(sim_outputs, tmp_path):
    cfg, results, _ = sim_outputs[0]
    filepath = results_to_parquet(results, cfg, path=tmp_path)

    assert filepath.exists()
    table = pq.read_table(filepath)

    n_steps = int(SIM_T_FINAL / SIM_H) + 1
    assert table.num_rows == n_steps
    for col in ["t", "eta_0", "eta_5", "nu_0", "n_actual_3", "alpha_actual_1"]:
        assert col in table.column_names

    # SimConfig must round-trip through the schema metadata
    meta = json.loads(table.schema.metadata[b"sim_config"].decode("utf-8"))
    assert meta["Hs"] == cfg.Hs
    assert meta["Tp"] == cfg.Tp
    assert meta["wave_seed"] == cfg.wave_seed

    np.testing.assert_allclose(
        table.column("eta_0").to_numpy(), results["eta"][:, 0]
    )


def test_sample_params_bounds():
    n = 16
    hss, tps, dirs = sample_params(n, h_hi_override=4.5, h_lo_override=0.5)
    assert hss.shape == (n,)
    assert tps.shape == (n,)
    assert dirs is not None and dirs.shape == (n,)
    assert np.all((hss >= 0.5) & (hss <= 4.5))
    assert np.all(np.isfinite(tps)) and np.all(tps > 0.0)
    assert np.all((dirs >= 0.0) & (dirs <= 360.0))

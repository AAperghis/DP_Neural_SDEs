from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
import os
import sys

import numpy as np
from tqdm import tqdm
import warnings

from thesis.full_order_dp.sample_params import sample_params
from thesis.full_order_dp.simulation import SimConfig, results_to_parquet, simulate_osv

# ---------- worker initializer ----------
_worker_position: int | None = None


def _init_worker(counter, lock):
    """Assign each pool worker a unique tqdm bar position."""
    global _worker_position
    with lock:
        _worker_position = counter.value
        counter.value += 1


# ---------- single simulation ----------
def run_sim(
    Hs: float, Tp: float, dir: float, *, seed: int = 0, drift_coeffs_path: Path
) -> tuple[SimConfig, dict[str, np.ndarray], bool]:
    if drift_coeffs_path is None:
        warnings.warn("No drift coefficients path provided", UserWarning)
    elif not drift_coeffs_path.exists():
        warnings.warn(
            f"Drift coefficients file not found: {drift_coeffs_path}", UserWarning
        )

    cfg = SimConfig(
        T_final=10800.0,
        h=0.5,
        x_ref=0.0,
        y_ref=0.0,
        psi_ref=0.0,
        Vc=0.0,
        betaVc=np.deg2rad(-140),
        Hs=Hs,
        Tp=Tp,
        beta_wave=np.deg2rad(dir),
        alloc_dynamic=True,
        wave_seed=seed,
    )

    n_steps = int(cfg.T_final / cfg.h) + 1
    pos = (_worker_position or 0) + 1  # +1 leaves row 0 for the overall bar
    bar = tqdm(
        total=n_steps,
        desc=f"seed={seed:>3d}",
        position=pos,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    def _progress(step: int, total: int) -> None:
        bar.n = step
        bar.refresh()

    result, success = simulate_osv(
        cfg,
        drift_coeffs_path=drift_coeffs_path,
        force_rao_path=None,
        progress_callback=_progress,
    )
    bar.close()
    return cfg, result, success


# ---------- entry point ----------
def main(save_path: Path, drift_coeffs_path: Path) -> None:
    N = 256
    n_workers = min((os.cpu_count() or 6) - 2 or 4, N)
    seeds = range(N)
    hss, tps, dirs = sample_params(n=N, h_hi_override=4.5, h_lo_override=0.0)

    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    overall = tqdm(total=N, desc="Overall", position=0)

    assert dirs is not None
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(counter, lock),
    ) as pool:
        futures = {
            pool.submit(
                run_sim,
                Hs=hs,
                Tp=tp,
                dir=dir,
                seed=seed,
                drift_coeffs_path=drift_coeffs_path,
            ): seed
            for hs, tp, dir, seed in zip(hss, tps, dirs, seeds)
        }
        for future in as_completed(futures):
            cfg, data, success = future.result()
            if success:
                results_to_parquet(
                    data,
                    cfg,
                    path=save_path,
                )
            overall.update(1)

    overall.close()


if __name__ == "__main__":
    save_path = r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data\fo_data_full_state_v4"
    drift_coeffs_path = (
        r"C:\Soft_dev\MSc_thesis\src\thesis\mss_model\drift_coefficients.npz"
    )

    if sys.platform == "linux":
        save_path = save_path.replace("\\", "/").replace("C:", "/mnt/c")
        drift_coeffs_path = (
            r"/home/aag/MSc_thesis/src/thesis/full_order_dp/drift_coefficients.npz"
        )

    main(
        save_path=Path(save_path),
        drift_coeffs_path=Path(drift_coeffs_path),
    )

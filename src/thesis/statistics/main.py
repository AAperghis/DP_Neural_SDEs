"""Run ensemble statistics on DP simulation data.

Usage:
    python -m thesis.statistics.main [--data-path PATH] [--n-files N] [--warmup S]
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np

from thesis.shared.data_handling import find_parquet_files
from thesis.shared.data_structures import FEATURE_REGISTRY, FeatureGroup
from thesis.statistics.loading import (
    Ensemble,
    load_ensemble,
)
from thesis.statistics.ensemble_stats import (
    MomentResult,
    PSDResult,
    ensemble_moments,
    ensemble_psd,
    footprint_radius,
)
from thesis.statistics.extreme_values import EVResult, pot_extreme_values
from thesis.statistics.convergence import (
    SampleConvergenceResult,
    TimeConvergenceResult,
    convergence_time,
    convergence_samples,
)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _feat_table(names: list[str], values: np.ndarray, fmt: str = ".4f") -> None:
    """Print a table of feature names → scalar values."""
    col_w = max(len(n) for n in names) + 2
    for name, val in zip(names, values):
        info = FEATURE_REGISTRY.get(name)
        unit = f" [{info.unit}]" if info else ""
        print(f"  {name:<{col_w}} {val:{fmt}}{unit}")


def _group_summary(
    names: list[str], values: np.ndarray, label: str, fmt: str = ".4f"
) -> None:
    """Print group-level aggregates (mean across features in each group)."""
    for group in FeatureGroup:
        group_idx = [
            i
            for i, n in enumerate(names)
            if FEATURE_REGISTRY.get(n) and FEATURE_REGISTRY[n].group == group
        ]
        if not group_idx:
            continue
        group_vals = values[group_idx]
        print(f"  {group.value:<12} {label} = {np.mean(group_vals):{fmt}}")


# ---------------------------------------------------------------------------
# Print routines for each analysis
# ---------------------------------------------------------------------------


def print_moments(ens: Ensemble) -> MomentResult:
    _header("Statistical Moments (stationary, after warm-up)")
    mom = ensemble_moments(ens.data)

    for label, vals in [
        ("Mean", mom.mean_scalar),
        ("Variance", mom.var_scalar),
        ("Skewness", mom.skew_scalar),
        ("Excess kurtosis", mom.kurt_scalar),
    ]:
        print(f"\n  --- {label} ---")
        _feat_table(ens.feature_names, vals)

    print("\n  --- Group summary (mean of absolute values) ---")
    for label, vals in [("Mean", mom.mean_scalar), ("Variance", mom.var_scalar)]:
        print(f"  {label}:")
        _group_summary(ens.feature_names, np.abs(vals), "mean|val|")

    return mom


def print_maxima(ens: Ensemble) -> EVResult:
    _header("Extreme Value Statistics (POT pipeline)")
    mx = pot_extreme_values(ens.data, ens.dt)

    for label, vals in [
        ("Max mean", mx.max_mean),
        ("Max std", mx.max_std),
        ("MPM (compound-max mode)", mx.mpm),
    ]:
        print(f"\n  --- {label} ---")
        _feat_table(ens.feature_names, vals)

    # Per-feature POT parameters
    print("\n  --- POT fit parameters ---")
    col_w = max(len(n) for n in ens.feature_names) + 2
    print(
        f"  {'feature':<{col_w}} {'shape':>8} {'scale':>8} {'threshold':>10} {'peak_rate':>10} {'n_peaks':>8} {'KS p':>8}"
    )
    for f_idx, name in enumerate(ens.feature_names):
        r = mx[f_idx]
        print(
            f"  {name:<{col_w}} {r.shape:>8.4f} {r.scale:>8.4f}"
            f" {r.threshold:>10.4f} {r.peak_rate:>10.1f} {r.n_peaks:>8d} {r.ks_pvalue:>8.4f}"
        )

    # Footprint
    if "pos_eta_x" in ens.feature_names and "pos_eta_y" in ens.feature_names:
        ix = ens.feature_index("pos_eta_x")
        iy = ens.feature_index("pos_eta_y")
        fp = footprint_radius(ens.data, ix, iy)
        print("\n  --- Footprint radius (max √(ηx² + ηy²)) ---")
        print(f"    mean = {np.mean(fp):.4f} m")
        print(f"    std  = {np.std(fp, ddof=1):.4f} m")
        print(f"    max  = {np.max(fp):.4f} m")

    return mx


def print_psd(ens: Ensemble) -> PSDResult:
    _header("Power Spectral Density (Welch)")
    psd = ensemble_psd(ens.data, ens.dt)

    print(f"  Frequency resolution: {psd.freqs[1] - psd.freqs[0]:.6f} Hz")
    print(f"  Frequency range: [{psd.freqs[0]:.4f}, {psd.freqs[-1]:.4f}] Hz")
    print(f"  Number of frequency bins: {len(psd.freqs)}")

    for label, vals, fmt in [
        ("m₀ (mean square)", psd.m0, ".6f"),
        ("Significant (4√m₀)", psd.significant, ".4f"),
        ("Tz (zero-crossing period)", psd.Tz, ".4f"),
        ("Bandwidth (ε)", psd.bandwidth, ".4f"),
        ("Peak frequency", psd.f_peak, ".6f"),
    ]:
        print(f"\n  --- {label} ---")
        _feat_table(ens.feature_names, vals, fmt=fmt)

    return psd


def print_time_convergence(ens: Ensemble) -> TimeConvergenceResult:
    _header("Convergence with Simulation Time")
    tc = convergence_time(ens.data, ens.time, ens.dt)

    print(f"  Windows (s): {tc.windows}")
    print()

    # Show how max_mean evolves for each feature
    print("  --- Max mean vs window duration ---")
    col_w = max(len(n) for n in ens.feature_names) + 2
    header = f"  {'feature':<{col_w}}" + "".join(f" {w:>8.0f}s" for w in tc.windows)
    print(header)
    print("  " + "-" * len(header))
    for f_idx, name in enumerate(ens.feature_names):
        vals = " ".join(f" {mx.max_mean[f_idx]:>8.4f}" for mx in tc.maxima)
        print(f"  {name:<{col_w}}{vals}")

    # Show variance convergence
    print("\n  --- Variance vs window duration ---")
    print(header)
    print("  " + "-" * len(header))
    for f_idx, name in enumerate(ens.feature_names):
        vals = " ".join(f" {m.var_scalar[f_idx]:>8.4f}" for m in tc.moments)
        print(f"  {name:<{col_w}}{vals}")

    return tc


def print_sample_convergence(ens: Ensemble) -> SampleConvergenceResult:
    _header("Convergence with Number of Samples")
    sc = convergence_samples(ens.data, ens.dt)

    print(f"  Sample sizes: {sc.sample_sizes}")
    print()

    col_w = max(len(n) for n in ens.feature_names) + 2

    print("  --- Mean vs sample count ---")
    header = f"  {'feature':<{col_w}}" + "".join(
        f"  N={n:>3d}" for n in sc.sample_sizes
    )
    print(header)
    print("  " + "-" * len(header))
    for f_idx, name in enumerate(ens.feature_names):
        vals = " ".join(f" {m.mean_scalar[f_idx]:>7.4f}" for m in sc.moments)
        print(f"  {name:<{col_w}}{vals}")

    print("\n  --- Max mean vs sample count (with 95% CI) ---")
    for f_idx, name in enumerate(ens.feature_names):
        parts = []
        for i, n_sub in enumerate(sc.sample_sizes):
            val = sc.maxima[i].max_mean[f_idx]
            lo = sc.maxima_ci_low[i].max_mean[f_idx]
            hi = sc.maxima_ci_high[i].max_mean[f_idx]
            parts.append(f" {val:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  {name:<{col_w}}{'  '.join(parts)}")

    return sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_path: Path,
    n_files: int = 50,
    t_warmup: float = 600.0,
    output_dir: Path | None = None,
) -> tuple[
    Ensemble,
    MomentResult,
    EVResult,
    PSDResult,
    TimeConvergenceResult,
    SampleConvergenceResult,
] | None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "statistics_output.txt"
        tee = open(log_path, "w")  # noqa: SIM115
        redirect = contextlib.redirect_stdout(tee)
        redirect.__enter__()
    else:
        tee = None
        redirect = None

    print(f"Data path: {data_path}")
    print(f"Max files: {n_files}")
    print(f"Warm-up:   {t_warmup} s")

    files = find_parquet_files(
        data_path,
        lambda m: (
            m.get("end_time") == 10800
            and m.get("timestep") == 0.05
            and m.get("seed", 999) < n_files
        ),
    )
    print(f"Found {len(files)} matching parquet files")
    if not files:
        print("No files found. Check --data-path.")
        return None

    ens = load_ensemble(files, t_warmup=t_warmup)
    print(f"Ensemble shape: {ens.data.shape}  (N={ens.N}, T={ens.T}, F={ens.F})")
    print(f"Time range: [{ens.time[0]:.1f}, {ens.time[-1]:.1f}] s  (dt={ens.dt:.4f})")

    mom = print_moments(ens)
    mx = print_maxima(ens)
    psd = print_psd(ens)
    tc = print_time_convergence(ens)
    sc = print_sample_convergence(ens)

    _header("Done")

    if redirect is not None:
        redirect.__exit__(None, None, None)
        if tee is not None:
            tee.close()
        print(f"Output saved to {log_path}")

    return ens, mom, mx, psd, tc, sc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP ensemble statistics")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            r"/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data"
        ),
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--n-files", type=int, default=50, help="Max number of seed files"
    )
    parser.add_argument(
        "--warmup", type=float, default=600.0, help="Warm-up truncation (s)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/statistics"),
        help="Directory for output files",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        n_files=args.n_files,
        t_warmup=args.warmup,
        output_dir=args.output_dir,
    )

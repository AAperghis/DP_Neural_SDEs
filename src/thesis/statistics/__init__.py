"""
Time series ensemble analysis toolkit for DP simulations.

All core functions operate on generic (N, T, F) numpy arrays.
Any source (DP simulation, neural SDE, etc.) producing that shape works.
"""

from thesis.statistics.loading import load_ensemble
from thesis.statistics.ensemble_stats import (
    ensemble_moments,
    ensemble_psd,
    ensemble_cross_correlation,
    cross_correlation_error,
    filter_outliers,
)
from thesis.statistics.convergence import convergence_time, convergence_samples
from thesis.statistics.comparison import compare_ensembles
from thesis.statistics.training_metrics import (
    compute_training_stats,
    compute_derived_quantities,
    plot_cross_correlation,
    plot_mpm_vs_hs,
    plot_feature_wave_correlation,
    plot_moments_vs_hs,
    plot_psd_comparison,
    plot_sea_state_comparison,
)
from thesis.statistics.extreme_values import pot_extreme_values, EVResult, POTResult
from thesis.statistics.main import main as run_statistics

__all__ = [
    "load_ensemble",
    "ensemble_moments",
    "ensemble_psd",
    "ensemble_cross_correlation",
    "cross_correlation_error",
    "filter_outliers",
    "convergence_time",
    "convergence_samples",
    "compare_ensembles",
    "compute_training_stats",
    "compute_derived_quantities",
    "plot_cross_correlation",
    "plot_mpm_vs_hs",
    "plot_feature_wave_correlation",
    "plot_moments_vs_hs",
    "plot_psd_comparison",
    "plot_sea_state_comparison",
    "pot_extreme_values",
    "EVResult",
    "POTResult",
    "run_statistics",
]

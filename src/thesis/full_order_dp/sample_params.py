"""Module for sampling parameters for the MSS model.
Uses the DNV-RP-C205 North atlantic Hs/Tz scatter diagram to sample Hs and Tz values.
The scatter diagram is represented as a 2D histogram,
where the x-axis represents the significant wave height (Hs)
and the y-axis represents the zero-crossing period (Tz).
The histogram values represent the frequency of occurrence of each Hs/Tz combination.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def read_scatter_diagram(
    file_path: Path = Path("src/thesis/full_order_dp/c-2_scatter_diagram.csv"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the scatter diagram from a CSV file and returns a pandas DataFrame."""
    scatter_diagram = pd.read_csv(file_path, header=None).to_numpy()
    tzs = scatter_diagram[0, 1:]
    hss = scatter_diagram[1:, 0]
    frequencies = scatter_diagram[1:, 1:]
    frequencies = frequencies / frequencies.sum()  # Normalize to get probabilities
    return tzs, hss, frequencies


def sobol_sample(
    hs_lo: float,
    hs_hi: float,
    tp_lo_fn: Callable,
    tp_hi_fn: Callable,
    dirs_lo: float | None = None,
    dirs_hi: float | None = None,
    n: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sample Hs and Tp using Sobol QMC with Hs-dependent Tp bounds.

    The first Sobol dimension maps uniformly to [hs_lo, hs_hi].
    The second dimension maps uniformly to [dirs_lo, dirs_hi] if provided, otherwise is ignored.
    The third dimension maps to [tp_lo_fn(hs), tp_hi_fn(hs)] per sample.
    """
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=3, scramble=True)
    u = sampler.random(n=n)  # shape (n, 3), values in [0, 1]
    hs_sample = hs_lo + u[:, 0] * (hs_hi - hs_lo)
    dirs_sample = (
        dirs_lo + u[:, 1] * (dirs_hi - dirs_lo)
        if dirs_lo is not None and dirs_hi is not None
        else None
    )
    tp_lo_vals = np.asarray([float(tp_lo_fn(h)) for h in hs_sample])
    tp_hi_vals = np.asarray([float(tp_hi_fn(h)) for h in hs_sample])
    tp_sample = tp_lo_vals + u[:, 2] * (tp_hi_vals - tp_lo_vals)
    return hs_sample, tp_sample, dirs_sample


def extract_bounds(
    tps: np.ndarray,
    hss: np.ndarray,
    frequencies: np.ndarray,
    hs_tail: float = 0.05,
    tp_tail: float = 0.05,
) -> tuple[float, float, Callable, Callable]:
    """Derive Hs limits and Hs-dependent Tp limits from a scatter diagram.

    Uses cumulative marginal / conditional probabilities so the bounds
    capture the central (1 − 2·tail) mass rather than relying on an
    absolute density threshold.

    Parameters
    ----------
    tps : np.ndarray
        Peak period grid corresponding to the columns of *frequencies*.
    hss : np.ndarray
        Significant wave height grid corresponding to the rows of *frequencies*.
    frequencies : np.ndarray
        2D array of frequencies (probabilities) for each Hs-Tp bin.
    hs_tail : float
        Probability mass to trim from each tail of the Hs marginal.
    tp_tail : float
        Probability mass to trim from each tail of the Tp conditional.

    Returns
    -------
    hs_lo, hs_hi : float
        Hs bounds enclosing the central (1 − 2·hs_tail) mass.
    tp_lo_fn, tp_hi_fn : callable(hs) -> float
        Interpolators that return the Tp bounds for a given Hs.
    """
    from scipy.interpolate import interp1d

    # --- Hs marginal bounds ---
    marginal_hs = frequencies.sum(axis=1)
    marginal_hs = marginal_hs / marginal_hs.sum()
    cdf_hs = np.cumsum(marginal_hs)
    hs_lo = np.interp(hs_tail, cdf_hs, hss)
    hs_hi = np.interp(1.0 - hs_tail, cdf_hs, hss)

    # --- Tp conditional bounds at each Hs bin ---
    tp_lo_per_hs = np.full(len(hss), np.nan)
    tp_hi_per_hs = np.full(len(hss), np.nan)
    for i, row in enumerate(frequencies):
        row_sum = row.sum()
        if row_sum <= 0:
            continue
        cdf_tp = np.cumsum(row / row_sum)
        tp_lo_per_hs[i] = np.interp(tp_tail, cdf_tp, tps)
        tp_hi_per_hs[i] = np.interp(1.0 - tp_tail, cdf_tp, tps)

    # Keep only bins with valid data for interpolation
    valid = ~np.isnan(tp_lo_per_hs)
    tp_lo_fn = interp1d(
        hss[valid], tp_lo_per_hs[valid], bounds_error=False, fill_value="extrapolate"
    )
    tp_hi_fn = interp1d(
        hss[valid], tp_hi_per_hs[valid], bounds_error=False, fill_value="extrapolate"
    )

    print(f"  Hs range: [{hs_lo:.2f}, {hs_hi:.2f}] m  (tail={hs_tail})")
    sample_hs = np.linspace(hs_lo, hs_hi, 8)
    for h in sample_hs:
        print(
            f"    Hs={h:5.2f} m  →  Tp ∈ [{float(tp_lo_fn(h)):.2f}, {float(tp_hi_fn(h)):.2f}] s"
        )

    return hs_lo, hs_hi, tp_lo_fn, tp_hi_fn


def plot_all_scatter_diagrams() -> None:
    """Plot observed, DNV parametric, and trivariate scatter diagrams as subplots."""
    import matplotlib.pyplot as plt

    tzs, hss, obs_frequencies = read_scatter_diagram()
    tps = np.asarray(tz_to_tp(tzs))
    tps_imca, tps_dnv = hs_tp_relations(hss)

    # High-resolution grid for parametric models
    hss_hr = np.linspace(hss.min(), hss.max(), len(hss) * 4)
    tps_hr = np.linspace(tps.min(), tps.max(), len(tps) * 4)
    tps_imca_hr, tps_dnv_hr = hs_tp_relations(hss_hr)

    dnv_frequencies = dnv_parametric_scatter(hss_hr, tps_hr)
    bi_frequencies = bivariate_scatter_li(hss_hr, tps_hr)
    tri_frequencies = trivariate_scatter_li(hss_hr, tps_hr)

    extent_obs = [tps.min(), tps.max(), hss.min(), hss.max()]
    extent_hr = [tps_hr.min(), tps_hr.max(), hss_hr.min(), hss_hr.max()]

    panels = [
        (
            "Observed Scatter Diagram - North Atlantic",
            obs_frequencies,
            "Frequency of Occurrence",
            hss,
            tps,
            tps_imca,
            tps_dnv,
            extent_obs,
        ),
        (
            "DNV-RP-C205 Parametric — North Sea",
            dnv_frequencies,
            "Probability Density (scaled)",
            hss_hr,
            tps_hr,
            tps_imca_hr,
            tps_dnv_hr,
            extent_hr,
        ),
        (
            "Trivariate Model (Li et al. 2013) - North Sea",
            tri_frequencies,
            "Probability Density (scaled)",
            hss_hr,
            tps_hr,
            tps_imca_hr,
            tps_dnv_hr,
            extent_hr,
        ),
        (
            "Bivariate Model (Li et al. 2013) - North Sea",
            bi_frequencies,
            "Probability Density (scaled)",
            hss_hr,
            tps_hr,
            tps_imca_hr,
            tps_dnv_hr,
            extent_hr,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(22, 6))

    for ax, (title, freq, cbar_label, p_hss, p_tps, p_imca, p_dnv, ext) in zip(
        axes.flatten(), panels
    ):
        im = ax.imshow(freq, extent=ext, aspect="auto", origin="lower", cmap="viridis")
        ax.plot(p_imca, p_hss, label="IMCA Recommended", color="red", linestyle="--")
        ax.plot(p_dnv, p_hss, label="DNV Recommended", color="blue", linestyle="--")

        # Draw probability bounds as a closed polygon
        print(f"{title}:")
        hs_lo, hs_hi, tp_lo_fn, tp_hi_fn = extract_bounds(p_tps, p_hss, freq)
        hs_curve = np.linspace(hs_lo, hs_hi, 100)
        tp_lo_curve = tp_lo_fn(hs_curve)
        tp_hi_curve = tp_hi_fn(hs_curve)
        # bottom edge (lo→hi along tp_lo), top edge (hi→lo along tp_hi) → closed loop
        poly_tp = np.concatenate([tp_lo_curve, tp_hi_curve[::-1], [tp_lo_curve[0]]])
        poly_hs = np.concatenate([hs_curve, hs_curve[::-1], [hs_curve[0]]])
        ax.plot(
            poly_tp, poly_hs, color="white", linewidth=1.5, label="Probability bounds"
        )

        # Sobol QMC samples within the bounds
        hs_sample, tp_sample, dirs_sample = sobol_sample(
            hs_lo, hs_hi, tp_lo_fn, tp_hi_fn
        )
        ax.scatter(
            tp_sample,
            hs_sample,
            color="white",
            s=10,
            zorder=5,
            edgecolors="black",
            linewidths=0.3,
            label="Sobol samples",
        )

        ax.set_xlabel("Peak Period (Tp)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize="small")
        ax.set_xlim(tp_lo_fn(hs_lo) * 0.9, tp_hi_fn(hs_hi) * 1.1)
        ax.set_ylim(hs_lo * 0.9, hs_hi * 1.1)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_ylabel("Significant Wave Height (Hs)")

    fig.tight_layout()
    plt.show()


def tz_to_tp(tz: np.ndarray | float, spectrum: str = "JONSWAP") -> np.ndarray | float:
    """Converts zero-crossing period (Tz) to peak period (Tp) using a common empirical relationship.
    Uses standard conversion factor of 1.29 (https://www.orcina.com/webhelp/OrcaFlex/Content/html/Environment,Modellingdesignwaves.htm)
    For JONSWAP spectrum or 1.41 for Pierson-Moskowitz spectrum.
    """
    match spectrum.lower():
        case "jonswap":
            return tz * 1.29
        case "pm" | "pierson-moskowitz":
            return tz * 1.41
        case _:
            raise ValueError(
                f"Unsupported spectrum type: {spectrum}. Supported types are 'JONSWAP' and 'Pierson-Moskowitz'."
            )


def hs_tp_relations(hss: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the peak period (Tp) corresponding to the significant wave height (Hs) using an empirical relationship.
    Returns IMCA and DNV recommended relationships.
    """
    hs_imca = [
        0,
        1.28,
        1.78,
        2.44,
        3.21,
        4.09,
        5.07,
        6.12,
        7.26,
        8.47,
        9.75,
        11.09,
        12.5,
        13.97,
        15.49,
    ]
    tp_imca = [
        0,
        5.3,
        6.26,
        7.32,
        8.41,
        9.49,
        10.56,
        11.61,
        12.64,
        13.65,
        14.65,
        15.62,
        16.58,
        17.53,
        18.46,
    ]

    hs_dnv = [0, 0.1, 0.4, 0.8, 1.3, 2.1, 3.1, 4.2, 5.7, 7.4, 9.5, 12.1]
    tp_dnv = [0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 10.5, 11.5, 12.0]

    tps_imca = np.asarray(np.interp(hss, hs_imca, tp_imca))
    tps_dnv = np.asarray(np.interp(hss, hs_dnv, tp_dnv))
    return tps_imca, tps_dnv


def dnv_parametric_scatter(
    hss: np.ndarray,
    tps: np.ndarray,
    alpha_s: float = 2.19,
    beta_s: float = 1.26,
    a1: float = 0.935,
    a2: float = 0.222,
    b1: float = 0.1386,
    b2: float = -0.0208,
) -> np.ndarray:
    """Build a scatter diagram from DNV-RP-C205 Table C-1 parametric model.

    Parameters default to Area 11 (North Sea).

    Returns
    -------
    frequencies : NDArray, shape (len(hss), len(tps))
        Joint probability density f(Hs, Tp) evaluated at the bin centres,
        scaled to the same total as a scatter diagram (counts / 1000).
    """
    from scipy.stats import weibull_min, lognorm

    dHs = np.diff(hss).mean() if len(hss) > 1 else 1.0
    dTp = np.diff(tps).mean() if len(tps) > 1 else 1.0

    frequencies = np.zeros((len(hss), len(tps)))
    for i, hs in enumerate(hss):
        if hs <= 0:
            continue
        # f(Hs) — Weibull PDF
        f_hs = weibull_min.pdf(hs, c=beta_s, scale=alpha_s)

        # f(Tz|Hs) — log-normal, convert Tp grid back to Tz
        tzs = tps / 1.285

        # See DNV-RP-C205 3.6.3.7 for the rationale behind these parameterisations
        mu_ln = 0.7 + a1 * hs**a2
        sigma_ln = 0.07 + b1 * np.exp(b2 * hs)
        if sigma_ln <= 0:
            sigma_ln = 1e-6
        f_tz = lognorm.pdf(tzs, s=sigma_ln, scale=np.exp(mu_ln))
        # Jacobian: f(Tp) = f(Tz) * |dTz/dTp| = f(Tz) / 1.285
        f_tp = f_tz / 1.285

        frequencies[i, :] = f_hs * f_tp

    # Scale to 1 total
    total = frequencies.sum() * dHs * dTp
    if total > 0:
        frequencies *= 1.0 / total * dHs * dTp

    return frequencies


def bivariate_scatter_li(
    hss: np.ndarray,
    tps: np.ndarray,
    *,
    h0: float = 2.5,
    mu_lhm: float = 0.334,
    sigma_lhm: float = 0.615,
    alpha_hm: float = 1.369,
    beta_hm: float = 1.653,
    c1: float = 1.587,
    c2: float = 0.222,
    c3: float = 0.674,
    d1: float = 0.008,
    d2: float = 0.227,
    d3: float = -0.956,
) -> np.ndarray:
    r"""Build an Hs-Tp scatter diagram by marginalising the bivariate Hs-Tp model (Li et al. 2013) over wind speed.

    .. math::
        $ f(H_s, T_p) = \int_0^\infty f(V_w)\,f(H_s|V_w)\,f(T_p|V_w,H_s)\;dV_w. $
    Returns
    -------
    frequencies : NDArray, shape (len(hss), len(tps))
        Joint density f(Hs, Tp) scaled to ~1000 total.
    """
    from scipy.stats import weibull_min, lognorm

    dHs = np.diff(hss).mean() if len(hss) > 1 else 1.0
    dTp = np.diff(tps).mean() if len(tps) > 1 else 1.0

    frequencies = np.zeros((len(hss), len(tps)))
    for i, hs in enumerate(hss):
        if hs <= 0:
            continue
        elif hs > h0:
            # f(Hs) — Weibull PDF
            f_hs = weibull_min.pdf(hs, c=alpha_hm, scale=beta_hm)
        else:
            # f(Hs) — Log-normal PDF
            f_hs = lognorm.pdf(hs, s=sigma_lhm, scale=np.exp(mu_lhm))

        # f(Tz|Hs) — log-normal, convert Tp grid back to Tz
        mu_ln = c1 + c2 * hs**c3
        sigma_ln = np.sqrt(d1 + d2 * np.exp(d3 * hs))

        if sigma_ln <= 0:
            sigma_ln = 1e-6
        f_tp = lognorm.pdf(tps, s=sigma_ln, scale=np.exp(mu_ln))

        frequencies[i, :] = f_hs * f_tp

    # Scale to 1 total
    total = frequencies.sum() * dHs * dTp
    if total > 0:
        frequencies *= 1.0 / total * dHs * dTp

    return frequencies


def trivariate_scatter_li(
    hss: np.ndarray,
    tps: np.ndarray,
    *,
    alpha_w: float = 2.299,
    beta_w: float = 8.920,
    a1: float = 1.755,
    a2: float = 1.84,
    a3: float = 1.0,
    b1: float = 0.534,
    b2: float = 0.07,
    b3: float = 1.435,
    e1: float = 5.563,
    e2: float = 0.798,
    e3: float = 1.0,
    f1: float = 3.5,
    f2: float = 3.592,
    f3: float = 0.735,
    k1: float = 0.050,
    k2: float = 0.388,
    k3: float = -0.321,
    theta: float = -0.477,
    gamma: float = 1.0,
    n_vw: int = 200,
    vw_max: float = 35.0,
) -> np.ndarray:
    r"""Build an Hs-Tp scatter diagram by marginalising the trivariate
    Vw-Hs-Tp model (Li et al. 2013) over wind speed.

    .. math::
        $ f(H_s, T_p) = \int_0^\infty f(V_w)\,f(H_s|V_w)\,f(T_p|V_w,H_s)\;dV_w. $

    Returns
    -------
    frequencies : NDArray, shape (len(hss), len(tps))
        Joint density f(Hs, Tp) scaled to ~1000 total.
    """
    from scipy.stats import weibull_min, lognorm

    dHs = np.diff(hss).mean() if len(hss) > 1 else 1.0
    dTp = np.diff(tps).mean() if len(tps) > 1 else 1.0

    # Wind speed integration grid
    Uw_grid = np.linspace(0.01, vw_max, n_vw)
    dUw = Uw_grid[1] - Uw_grid[0]

    # f(Vw) — Weibull marginal
    f_Uw = weibull_min.pdf(Uw_grid, c=alpha_w, scale=beta_w)

    frequencies = np.zeros((len(hss), len(tps)))

    for iv, vw in enumerate(Uw_grid):
        # Conditional Weibull parameters for Hs|Vw  (Eq. 16)
        alpha_hs = a1 + a2 * vw**a3
        beta_hs = b1 + b2 * vw**b3
        if alpha_hs <= 0 or beta_hs <= 0:
            continue

        for ih, hs in enumerate(hss):
            if hs <= 0:
                continue

            # f(Hs|Vw)  (Eq. 15)
            f_hs_vw = weibull_min.pdf(hs, c=alpha_hs, scale=beta_hs)
            if f_hs_vw < 1e-30:
                continue

            tp_bar = e1 + e2 * hs**e3
            Uw_bar = f1 + f2 * hs**f3
            nu_tp = k1 + k2 * np.exp(k3 * hs)

            mu_tp = tp_bar * (1.0 + theta * ((vw - Uw_bar) / Uw_bar) ** gamma)

            if mu_tp <= 0 or nu_tp <= 0:
                continue
            sigma_tp = np.sqrt(np.log(nu_tp**2 + 1.0))
            mu_logtp = np.log(mu_tp / np.sqrt(nu_tp**2 + 1.0))

            # f(Tp|Vw,Hs)  (Eq. 13 with enhanced parameters)
            f_tp = lognorm.pdf(tps, s=sigma_tp, scale=np.exp(mu_logtp))

            frequencies[ih, :] += f_Uw[iv] * f_hs_vw * f_tp * dUw

    # Scale to 1 total
    total = frequencies.sum() * dHs * dTp
    if total > 0:
        frequencies *= 1.0 / total * dHs * dTp

    return frequencies


def sample_params(
    n: int,
    distribution: str = "dnv",
    *,
    h_hi_override: float | None = None,
    h_lo_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sample Hs and Tp values"""
    hss = np.arange(0.0, 15.0, 0.5)
    tps_grid = np.arange(0.0, 20.0, 0.25)
    dirs_lo = 0.0
    dirs_hi = 360.0
    match distribution.lower():
        case "dnv":
            frequencies = dnv_parametric_scatter(hss, tps_grid)
        case "scatter":
            tzs, hss, frequencies = read_scatter_diagram()
            tps_grid = np.asarray(tz_to_tp(tzs))
        case "bivariate":
            frequencies = bivariate_scatter_li(hss, tps_grid)
        case "trivariate":
            frequencies = trivariate_scatter_li(hss, tps_grid)
        case _:
            print("Defaulting to DNV parametric model.")
            frequencies = dnv_parametric_scatter(hss, tps_grid)

    hs_lo, hs_hi, tp_lo_fn, tp_hi_fn = extract_bounds(tps_grid, hss, frequencies)

    if h_lo_override is not None:
        hs_lo = h_lo_override
    if h_hi_override is not None:
        hs_hi = h_hi_override

    hs_samples, tp_samples, dirs_samples = sobol_sample(
        hs_lo, hs_hi, tp_lo_fn, tp_hi_fn, dirs_lo, dirs_hi, n
    )
    return hs_samples, tp_samples, dirs_samples


if __name__ == "__main__":
    plot_all_scatter_diagrams()

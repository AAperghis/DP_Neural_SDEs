import shutil
from pathlib import Path

import jax
import jax.typing as jtp
import jax.numpy as jnp
import jax.random as jr
import matplotlib


import os
import tempfile

import mlflow
import numpy as np
from thesis.full_hybrid_sde.model import FullHybridSDE
from thesis.shared.consolidated_dataset import ConsolidatedDataset
from thesis.shared.data_structures import HyperParameters
from thesis.shared.sampler import Sampler
from thesis.utils import AsyncLogger


def _render_sample_plots(
    sample_ts,
    prior_samples,
    full_ts,
    full_samples,
    vis_ts,
    vis_xs,
    posterior_samples,
    step: int,
    feature_names: list[str],
    data_mean,
    data_std,
) -> dict[str, str]:
    """Render prior/posterior sample plots to temp directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_feats = len(feature_names)
    nrows = max(1, (n_feats + 3) // 4)
    ncols = min(n_feats, 4)
    artifacts = {}
    tmpdir = tempfile.mkdtemp(prefix="mss_plots_")

    for tag, ts_arr, samples_arr in [
        ("prior", sample_ts, prior_samples),
        ("full_prior", full_ts, full_samples),
    ]:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for j in range(min(n_feats, len(axes_flat))):
            ax = axes_flat[j]
            for s in range(samples_arr.shape[0]):
                ax.plot(ts_arr, samples_arr[s, :, j], alpha=0.5, linewidth=0.5)
            ax.set_title(feature_names[j], fontsize=8)
        for j in range(n_feats, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{tag} — step {step}")
        plt.tight_layout()
        path = os.path.join(tmpdir, f"{tag}_step_{step:05d}.pdf")
        fig.savefig(path)
        plt.close(fig)
        artifacts[f"samples/{tag}/step_{step:05d}.pdf"] = path

    # Posterior
    n_posterior = posterior_samples.shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for j in range(min(n_feats, len(axes_flat))):
        ax = axes_flat[j]
        for s in range(n_posterior):
            ax.plot(vis_ts, vis_xs[s, :, j], color="black", alpha=0.3, linewidth=0.5)
            ax.plot(vis_ts, posterior_samples[s, :, j], alpha=0.5, linewidth=0.5)
        ax.set_title(feature_names[j], fontsize=8)
    for j in range(n_feats, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"posterior — step {step}")
    plt.tight_layout()
    path = os.path.join(tmpdir, f"posterior_step_{step:05d}.pdf")
    fig.savefig(path)
    plt.close(fig)
    artifacts[f"samples/posterior/step_{step:05d}.pdf"] = path

    return artifacts


def _render_sea_state_plot(
    model_data,
    ref_data,
    model_wave,
    ref_wave,
    dt,
    feature_names,
    physics_config,
    step,
) -> dict[str, str]:
    """Render sea-state comparison plot to temp directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from thesis.statistics.training_metrics import plot_sea_state_comparison

    fig = plot_sea_state_comparison(
        model_data,
        ref_data,
        model_wave,
        ref_wave,
        dt,
        feature_names,
        physics_config,
        step=step,
    )
    tmpdir = tempfile.mkdtemp(prefix="ss_plot_")
    path = os.path.join(tmpdir, f"sea_state_step_{step:05d}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {f"samples/sea_state/step_{step:05d}.pdf": path}


def _render_cross_correlation_plot(
    model_data,
    ref_data,
    feature_names,
    step,
) -> dict[str, str]:
    """Render feature cross-correlation heatmap to temp directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from thesis.statistics.training_metrics import plot_cross_correlation

    fig = plot_cross_correlation(
        model_data,
        ref_data,
        feature_names,
        step=step,
    )
    tmpdir = tempfile.mkdtemp(prefix="xcorr_plot_")
    path = os.path.join(tmpdir, f"cross_corr_step_{step:05d}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {f"samples/cross_correlation/step_{step:05d}.pdf": path}


def _render_eval_plots(
    model_data,
    ref_data,
    model_wave,
    ref_wave,
    dt,
    feature_names,
    step,
) -> dict[str, str]:
    """Render MPM-vs-Hs, feature-wave correlation, moments-vs-Hs, and PSD plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from thesis.statistics.training_metrics import (
        plot_mpm_vs_hs,
        plot_feature_wave_correlation,
        plot_moments_vs_hs,
        plot_psd_comparison,
    )

    tmpdir = tempfile.mkdtemp(prefix="eval_plots_")
    artifacts: dict[str, str] = {}

    for name, fig in [
        (
            "mpm_vs_hs",
            plot_mpm_vs_hs(
                model_data, ref_data, model_wave, ref_wave, dt, feature_names, step=step
            ),
        ),
        (
            "feature_wave_corr",
            plot_feature_wave_correlation(
                model_data, ref_data, model_wave, ref_wave, feature_names, step=step
            ),
        ),
        (
            "moments_vs_hs",
            plot_moments_vs_hs(
                model_data, ref_data, model_wave, ref_wave, feature_names, step=step
            ),
        ),
        (
            "psd",
            plot_psd_comparison(model_data, ref_data, dt, feature_names, step=step),
        ),
    ]:
        path = os.path.join(tmpdir, f"{name}_step_{step:05d}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        artifacts[f"samples/{name}/step_{step:05d}.pdf"] = path

    return artifacts


def _log_artifacts(
    artifacts: dict[str, str],
    run_id: str,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    with mlflow.start_run(run_id=run_id):
        for art_path, local_path in artifacts.items():
            mlflow.log_artifact(local_path, artifact_path=str(Path(art_path).parent))
        if metrics:
            mlflow.log_metrics(metrics, step=step)
    dirs = {os.path.dirname(p) for p in artifacts.values()}
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def _compute_stats(
    model_data: np.ndarray,
    ref_data: np.ndarray,
    model_wave: np.ndarray,
    ref_wave: np.ndarray,
    dt: float,
    feature_names: list[str],
    step: int,
    sweep_ref_cache=None,
    stats_ref_cache=None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Compute training statistics and sweep scores (runs in process pool).

    Returns:
        artifacts: dict mapping artifact paths to local file paths
        metrics: flat dict of scalar metrics for MLflow
    """
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from thesis.statistics.training_metrics import (
        compute_training_stats,
        sweep_score,
        plot_training_stats,
    )

    artifacts: dict[str, str] = {}
    tmpdir = tempfile.mkdtemp(prefix="stats_plots_")

    statistics = compute_training_stats(
        model_data,
        ref_data,
        dt,
        feature_names=feature_names,
        prefix="prior_stats",
        ref_cache=stats_ref_cache,
    )
    metrics = dict(statistics.metrics)

    # Compact η-focused sweep scores
    scores = sweep_score(
        model_data,
        ref_data,
        dt,
        model_wave=model_wave,
        ref_wave=ref_wave,
        ref_cache=sweep_ref_cache,
    )
    metrics.update({f"sweep/{k}": v for k, v in scores.items()})

    # Stats plots
    fig_moments, fig_psd, fig_maxima = plot_training_stats(
        statistics,
        feature_names=feature_names,
    )
    for name, fig in [
        ("moments", fig_moments),
        ("psd", fig_psd),
        ("maxima", fig_maxima),
    ]:
        path = os.path.join(tmpdir, f"{name}_step_{step:05d}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        artifacts[f"stats/prior_stats/{name}/step_{step:05d}.pdf"] = path

    return artifacts, metrics


class FOHybridSampler(Sampler):
    """Periodic sampler for :class:`FOHybridTrainSetup`.

    ``setup`` precomputes the reference trajectory cache once per run.
    ``sample`` generates prior/posterior samples and dispatches rendering
    and stats computation to the background pools stored by ``setup``.
    """

    def setup(
        self,
        model: FullHybridSDE,
        dataset: ConsolidatedDataset,
        hyperparams: HyperParameters,
        compute_pool,
        log_thread,
        run_id: str,
    ) -> None:
        super().setup(model, dataset, hyperparams, compute_pool, log_thread, run_id)
        self._pending = None

        from thesis.statistics.training_metrics import (
            precompute_sweep_ref,
            precompute_stats_ref,
        )

        ref_key = jr.PRNGKey(42)
        n_ref = min(128, dataset._n_series())
        _, ref_xs_std, ref_wcs_scaled = dataset.sample_full_trajectories(ref_key, n_ref)
        ref_xs_phys = np.asarray(ref_xs_std) * np.asarray(model.data_std) + np.asarray(
            model.data_mean
        )
        ref_wcs_phys = np.asarray(dataset.inverse_scale_wave_cond(ref_wcs_scaled))

        model_T = len(jnp.arange(0, 10800.0, hyperparams.data.dt))
        T_min = min(model_T, ref_xs_phys.shape[1])
        self._ref_xs_phys = ref_xs_phys[:, :T_min]
        self._ref_wcs_phys = ref_wcs_phys
        self._T_min = T_min
        self._sweep_ref_cache = precompute_sweep_ref(
            self._ref_xs_phys, ref_wave=ref_wcs_phys
        )
        self._stats_ref_cache = precompute_stats_ref(
            self._ref_xs_phys, hyperparams.data.dt
        )
        self._feature_names = [
            f.value if hasattr(f, "value") else str(f)
            for f in hyperparams.data.features
        ]

    def sample(
        self,
        model: FullHybridSDE,
        step: int,
        step_key: jtp.ArrayLike,
        *,
        logger: AsyncLogger | None = None,
    ) -> None:
        if self._pending is not None:
            self._pending.result()

        hyperparams = self.hyperparams
        dataset = self.dataset
        ref_xs_phys = self._ref_xs_phys
        ref_wcs_phys = self._ref_wcs_phys
        T_min = self._T_min
        feature_names = self._feature_names

        sample_key = jr.fold_in(step_key, 999)
        prior_key, posterior_key = jr.split(sample_key, 2)
        full_ts = jnp.arange(0, 10800.0, hyperparams.data.dt)

        # Sample prior trajectories
        n_prior = 64
        data_key, wc_key = jr.split(prior_key)
        _, _, wcs_prior = dataset.sample_full_trajectories(wc_key, n_prior)
        prior_keys = jr.split(data_key, n_prior)
        full_samples = jax.vmap(
            lambda k, wc: model.sample_prior(full_ts, key=k, wave_cond=wc, unscale=True)
        )(prior_keys, wcs_prior)

        # Sample posterior trajectories
        n_posterior = 8
        post_data_key, post_sample_key = jr.split(posterior_key)
        vis_ts, vis_xs, vis_wc, _ = dataset.sample_random(post_data_key, n_posterior)
        post_keys = jr.split(post_sample_key, n_posterior)
        posterior_samples = jax.vmap(
            lambda xs, wc, k: model.sample_posterior(
                xs, vis_ts, key=k, wave_cond=wc, unscale=False
            )
        )(vis_xs, vis_wc, post_keys)

        def _unscale(x):
            return x * model.data_std + model.data_mean

        # Convert to NumPy for plotting and stats computation
        posterior_np = np.asarray(jax.vmap(jax.vmap(_unscale))(posterior_samples))
        vis_xs_np = np.asarray(jax.vmap(jax.vmap(_unscale))(vis_xs))
        full_samples_np = np.asarray(full_samples)
        model_wcs_phys = np.asarray(dataset.inverse_scale_wave_cond(wcs_prior))
        phys_cfg = hyperparams.model.fo_physics_config
        train_len = len(vis_ts)

        # Dispatch rendering and stats computation to background pools
        plot_future = self.compute_pool.submit(
            _render_sample_plots,
            np.asarray(full_ts[:train_len]),
            full_samples_np[:4, :train_len],
            np.asarray(full_ts),
            full_samples_np[:4],
            np.asarray(vis_ts),
            vis_xs_np[:4],
            posterior_np[:4],
            step,
            feature_names,
            np.asarray(model.data_mean),
            np.asarray(model.data_std),
        )
        ss_future = self.compute_pool.submit(
            _render_sea_state_plot,
            full_samples_np[:, :T_min],
            ref_xs_phys[:, :T_min],
            model_wcs_phys,
            ref_wcs_phys,
            hyperparams.data.dt,
            feature_names,
            phys_cfg,
            step,
        )
        xcorr_future = self.compute_pool.submit(
            _render_cross_correlation_plot,
            full_samples_np[:, :T_min],
            ref_xs_phys[:, :T_min],
            feature_names,
            step,
        )
        eval_future = self.compute_pool.submit(
            _render_eval_plots,
            full_samples_np[:, :T_min],
            ref_xs_phys[:, :T_min],
            model_wcs_phys,
            ref_wcs_phys,
            hyperparams.data.dt,
            feature_names,
            step,
        )
        stats_future = self.compute_pool.submit(
            _compute_stats,
            full_samples_np[:, :T_min],
            ref_xs_phys[:, :T_min],
            model_wcs_phys,
            ref_wcs_phys,
            hyperparams.data.dt,
            feature_names,
            step,
            self._sweep_ref_cache,
            self._stats_ref_cache,
        )

        run_id = self.run_id

        # Wait for all rendering and stats computation to finish, then log everything together
        def _log_all():
            artifacts = plot_future.result()
            artifacts.update(ss_future.result())
            artifacts.update(xcorr_future.result())
            artifacts.update(eval_future.result())
            stats_artifacts, stats_metrics = stats_future.result()
            artifacts.update(stats_artifacts)
            _log_artifacts(artifacts, run_id, step=step, metrics=stats_metrics)

        self._pending = self.log_thread.submit(_log_all)

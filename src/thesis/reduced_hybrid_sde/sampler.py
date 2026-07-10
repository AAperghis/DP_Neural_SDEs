"""Sampler for the ReducedHybridSDE training pipeline.

Encapsulates periodic prior/posterior sample generation, rendering,
and MLflow logging, keeping all background-pool logic out of train.py.
"""

import os
import shutil
import tempfile
from concurrent.futures import Future
from pathlib import Path

import jax
import jax.typing as jtp
import jax.numpy as jnp
import jax.random as jr
import mlflow
import numpy as np

from thesis.reduced_hybrid_sde.model import ReducedHybridSDE
from thesis.shared.data_structures import HyperParameters
from thesis.shared.sampler import Sampler
from thesis.utils import AsyncLogger


def _render_mr_params(
    rates,
    means,
    step: int,
) -> dict[str, str]:
    """Render mean-reversion rates and means as a bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = len(rates)
    x = np.arange(d)
    artifacts = {}
    tmpdir = tempfile.mkdtemp(prefix="mr_plots_")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, rates, color="steelblue")
    ax1.set_xlabel("Latent dimension")
    ax1.set_ylabel("Rate (softplus)")
    ax1.set_title(f"MR decay rates — step {step}")
    ax1.set_xticks(x)

    ax2.bar(x, means, color="coral")
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Latent dimension")
    ax2.set_ylabel("Mean")
    ax2.set_title(f"MR target mean — step {step}")
    ax2.set_xticks(x)

    fig.tight_layout()
    path = os.path.join(tmpdir, f"mr_params_step_{step:05d}.pdf")
    fig.savefig(path)
    plt.close(fig)
    artifacts[f"samples/mr_params/step_{step:05d}.pdf"] = path
    return artifacts


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
    """Render prior and posterior figures to a temp directory (process-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = {}
    tmpdir = tempfile.mkdtemp(prefix="sample_plots_")
    prior_dir = os.path.join(tmpdir, "prior")
    os.makedirs(prior_dir)
    full_dir = os.path.join(tmpdir, "full")
    os.makedirs(full_dir)
    post_dir = os.path.join(tmpdir, "posterior")
    os.makedirs(post_dir)

    # Prior (training window)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        for s in range(prior_samples.shape[0]):
            ax.plot(sample_ts, prior_samples[s, :, j], alpha=0.6, linewidth=0.5)
        ax.set_title(feature_names[j])
        mu, sigma = float(data_mean[j]), float(data_std[j])
        ax.secondary_yaxis(
            "right",
            functions=(
                lambda x, m=mu, s=sigma: (x - m) / s,
                lambda x, m=mu, s=sigma: x * s + m,
            ),
        )
    fig.suptitle(f"Prior samples — step {step}")
    plt.tight_layout()
    prior_path = os.path.join(prior_dir, f"step_{step:05d}.pdf")
    fig.savefig(prior_path)
    plt.close(fig)
    artifacts[f"samples/prior/step_{step:05d}.pdf"] = prior_path

    # Full 3-hour prior
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        for s in range(full_samples.shape[0]):
            ax.plot(full_ts, full_samples[s, :, j], alpha=0.6, linewidth=0.5)
        ax.set_title(feature_names[j])
        mu, sigma = float(data_mean[j]), float(data_std[j])
        ax.secondary_yaxis(
            "right",
            functions=(
                lambda x, m=mu, s=sigma: (x - m) / s,
                lambda x, m=mu, s=sigma: x * s + m,
            ),
        )
    fig.suptitle(f"3h prior samples — step {step}")
    plt.tight_layout()
    full_prior_path = os.path.join(full_dir, f"step_{step:05d}.pdf")
    fig.savefig(full_prior_path)
    plt.close(fig)
    artifacts[f"samples/3h_prior/step_{step:05d}.pdf"] = full_prior_path

    # Posterior
    n_posterior = posterior_samples.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        for s in range(n_posterior):
            ax.plot(
                vis_ts,
                vis_xs[s, :, j],
                color="black",
                alpha=0.4,
                linewidth=0.5,
                label="data" if s == 0 else None,
            )
            ax.plot(
                vis_ts,
                posterior_samples[s, :, j],
                alpha=0.6,
                linewidth=0.5,
                label="posterior" if s == 0 else None,
            )
        ax.set_title(feature_names[j])
        mu, sigma = float(data_mean[j]), float(data_std[j])
        ax.secondary_yaxis(
            "right",
            functions=(
                lambda x, m=mu, s=sigma: (x - m) / s,
                lambda x, m=mu, s=sigma: x * s + m,
            ),
        )
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle(f"Posterior samples — Step {step}")
    plt.tight_layout()
    post_path = os.path.join(post_dir, f"step_{step:05d}.pdf")
    fig.savefig(post_path)
    plt.close(fig)
    artifacts[f"samples/posterior/step_{step:05d}.pdf"] = post_path

    return artifacts


def _compute_stats(
    prior_samples,
    ref_prior,
    dt: float,
    step: int,
    feature_names: list[str],
    prefix: str = "prior_stats",
) -> tuple[dict[str, str], dict[str, float]]:
    """Compute ensemble statistics and render figures (process-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from thesis.statistics.training_metrics import (
        compute_training_stats,
        plot_training_stats,
    )

    artifacts = {}
    tmpdir = tempfile.mkdtemp(prefix="stats_plots_")

    statistics = compute_training_stats(
        prior_samples,
        ref_prior,
        dt,
        feature_names=feature_names,
        prefix=prefix,
    )
    fig_moments, fig_psd, fig_maxima = plot_training_stats(
        statistics, feature_names=feature_names
    )

    for name, fig in [
        ("moments", fig_moments),
        ("psd", fig_psd),
        ("maxima", fig_maxima),
    ]:
        path = os.path.join(tmpdir, f"{name}_step_{step:05d}.pdf")
        fig.savefig(path)
        plt.close(fig)
        artifacts[f"stats/{prefix}/{name}/step_{step:05d}.pdf"] = path

    return artifacts, statistics.metrics


def _log_artifacts(
    artifacts: dict[str, str],
    run_id: str,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Log precomputed artifacts and metrics to MLflow (runs in main-process thread)."""
    with mlflow.start_run(run_id=run_id):
        for artifact_path, local_path in artifacts.items():
            mlflow.log_artifact(
                local_path, artifact_path=str(Path(artifact_path).parent)
            )
        if metrics:
            mlflow.log_metrics(metrics, step=step)
    dirs_to_remove = {os.path.dirname(p) for p in artifacts.values()}
    for d in dirs_to_remove:
        shutil.rmtree(d, ignore_errors=True)


class ROHybridSampler(Sampler):
    """Periodic sampler for :class:`ROHybridTrainSetup`.

    ``setup`` stores the dataset reference once per run.
    ``sample`` generates prior/posterior samples and dispatches rendering
    and statistics computation to the background pools.
    """

    def setup(
        self,
        model: ReducedHybridSDE,
        dataset,
        hyperparams: HyperParameters,
        compute_pool,
        log_thread,
        run_id: str,
    ) -> None:
        super().setup(model, dataset, hyperparams, compute_pool, log_thread, run_id)
        self._pending: Future | None = None
        self._feature_names = [
            f.value if hasattr(f, "value") else str(f)
            for f in hyperparams.data.features
        ]

    def sample(
        self,
        model: ReducedHybridSDE,
        step: int,
        step_key: jtp.ArrayLike,
        *,
        logger: AsyncLogger | None = None,
    ) -> None:
        if self._pending is not None:
            self._pending.result()

        hyperparams = self.hyperparams
        dataset = self.dataset
        feature_names = self._feature_names

        sample_key = jr.fold_in(step_key, 999)
        prior_key, posterior_key, stats_key = jr.split(sample_key, 3)

        full_ts = jnp.arange(0, 10800.0, hyperparams.data.dt)  # 3 hours

        # Prior samples
        n_prior = 16
        full_keys = jr.split(prior_key, n_prior)
        full_samples = jax.vmap(
            lambda k: model.sample_prior(full_ts, key=k, unscale=True)
        )(full_keys)

        # Posterior samples
        n_posterior = 16
        post_data_key, post_sample_key = jr.split(posterior_key)
        vis_ts, vis_xs, _, _ = dataset.sample_random(post_data_key, n_posterior)
        post_keys = jr.split(post_sample_key, n_posterior)
        posterior_samples = jax.vmap(
            lambda xs, k: model.sample_posterior(xs, vis_ts, key=k, unscale=False)
        )(vis_xs, post_keys)

        train_len = len(vis_ts)

        # Reference data for statistics comparison
        ref_data_key = jr.fold_in(stats_key, step)
        _, ref_xs, _ = dataset.sample_full_trajectories(ref_data_key, n_prior)

        def _unscale(x):
            return x * model.data_std + model.data_mean

        ref_xs = jax.vmap(jax.vmap(_unscale))(ref_xs)
        posterior_samples = jax.vmap(jax.vmap(_unscale))(posterior_samples)
        vis_xs = jax.vmap(jax.vmap(_unscale))(vis_xs)

        train_samples = full_samples[:, :train_len, :]
        train_ts = full_ts[:train_len]

        # Copy to CPU numpy
        train_samples_np = np.asarray(train_samples)
        full_samples_np = np.asarray(full_samples)
        posterior_np = np.asarray(posterior_samples)
        vis_ts_np = np.asarray(vis_ts)
        vis_xs_np = np.asarray(vis_xs)
        train_ts_np = np.asarray(train_ts)
        ref_xs_np = np.asarray(ref_xs)
        data_mean_np = np.asarray(model.data_mean)
        data_std_np = np.asarray(model.data_std)
        full_ts_np = np.asarray(full_ts)

        mr = getattr(model, "mean_reversion", None)
        mr_rates_np = np.asarray(mr.rates) if mr is not None else None
        mr_means_np = np.asarray(mr.mean) if mr is not None else None

        plot_future = self.compute_pool.submit(
            _render_sample_plots,
            train_ts_np,
            train_samples_np[:4],
            full_ts_np,
            full_samples_np[:4],
            vis_ts_np,
            vis_xs_np[:4],
            posterior_np[:4],
            step,
            feature_names,
            data_mean_np,
            data_std_np,
        )

        full_stats_future = self.compute_pool.submit(
            _compute_stats,
            full_samples_np,
            ref_xs_np,
            hyperparams.data.dt,
            step,
            feature_names,
            "full_prior_stats",
        )

        mr_future = None
        if mr_rates_np is not None:
            mr_future = self.compute_pool.submit(
                _render_mr_params,
                mr_rates_np,
                mr_means_np,
                step,
            )

        run_id = self.run_id

        def _log_all():
            plot_artifacts = plot_future.result()
            _log_artifacts(plot_artifacts, run_id)

            full_artifacts, full_metrics = full_stats_future.result()
            _log_artifacts(full_artifacts, run_id, step=step, metrics=full_metrics)

            if mr_future is not None:
                mr_artifacts = mr_future.result()
                _log_artifacts(mr_artifacts, run_id)

        self._pending = self.log_thread.submit(_log_all)

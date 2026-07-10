"""Shared parametric training loop for SDE models."""

import math
import subprocess
import sys
import tempfile
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow

from thesis.shared.data_handling import find_parquet_files
from thesis.shared.annealing import anneal_params
from thesis.shared.data_structures import (
    hyperparams_to_json,
)
from thesis.shared.jax_dataset import JAXParquetDataset
from thesis.shared.model import SDE
from thesis.shared.train_setup import TrainSetup
from thesis.shared.utils import get_device, log_batch, log_summary
from thesis.utils import AsyncLogger, crash_logger, log_compute_to_mlflow

mlflow.enable_system_metrics_logging()

# Persist XLA compilation cache across runs so the first training step
# doesn't trigger a full recompilation every time.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")


def train(
    train_setup: TrainSetup,
) -> None:
    """Step-based training loop for SDE models.

    Args:
        train_setup: An instance of TrainSetup containing the model, dataset, and training configuration.
    """
    # Unpack training setup
    hyperparams = train_setup.hyperparams
    data_path = train_setup.data_path
    initialised_model = train_setup.initialised_model
    model_type_name = train_setup.model_type_name
    device = train_setup.device
    run_name = train_setup.run_name
    print_batch = train_setup.print_batch
    dataset = train_setup.dataset
    test_dataset = train_setup.test_dataset
    parent_id = train_setup.parent_id
    start_step = train_setup.start_step

    logger = AsyncLogger()
    device = get_device(device, logger)

    model_key, train_key, _ = jr.split(hyperparams.training.key, 3)
    if initialised_model is not None:
        model = initialised_model
    else:
        model = train_setup.init_model_fn(hyperparams, model_key)

    # ------------------------------------------------------------------
    # Dataset setup
    # ------------------------------------------------------------------
    if dataset is not None:
        # Use pre-built dataset; data_path used only for logging.
        files = getattr(dataset, "files", [])
        if test_dataset is None:
            test_fraction = hyperparams.data.test_fraction
            if test_fraction > 0 and len(files) >= 4:
                dataset, test_dataset = dataset.split(
                    test_fraction=test_fraction, seed=42
                )
                print(
                    f"Train/test split: {len(dataset.files)} train, "
                    f"{len(test_dataset.files)} test files"
                )
    else:
        files = find_parquet_files(
            data_path,
            lambda m: (
                m["end_time"] == 10800
                and m["timestep"] == 0.05
                and (
                    hyperparams.data.n_files is None
                    or m["seed"] < hyperparams.data.n_files
                )
            ),
        )
        dataset = JAXParquetDataset(
            files,
            columns=hyperparams.data.features,
            sample_length=int(
                hyperparams.data.sample_length * (hyperparams.data.dt / 0.05)
            ),
            resample_every=int(hyperparams.data.dt // 0.05),
            standardise=True,
            cache_size=5000,
            truncate_seconds=hyperparams.data.truncate_seconds,
            group_scaling=hyperparams.data.group_scaling,
        )

        # Train/test split — standardisation is computed from train files only
        test_fraction = hyperparams.data.test_fraction
        if test_fraction > 0 and len(files) >= 4:
            dataset, test_dataset = dataset.split(test_fraction=test_fraction, seed=42)
            print(
                f"Train/test split: {len(dataset.files)} train, "
                f"{len(test_dataset.files)} test files"
            )

    print(f"Loaded {len(files)} files, {len(dataset)} train samples")

    # Store standardisation stats on the model (if it supports it)
    if hasattr(model, "data_mean"):
        model: SDE = eqx.tree_at(
            lambda m: (m.data_mean, m.data_std),
            model,
            (
                jnp.array(dataset.standardise["mean"], dtype=jnp.float32),
                jnp.array(dataset.standardise["std"], dtype=jnp.float32),
            ),
        )

    schedule = train_setup.make_schedule_fn(hyperparams)
    if schedule is None:
        raise ValueError("make_schedule_fn returned None; a schedule is required.")
    optimizer = train_setup.make_optimizer_fn(
        schedule, hyperparams.training.max_grad_norm, hyperparams.training.weight_decay
    )

    filter_spec = model.trainable_filter()
    model_params, model_static = eqx.partition(model, filter_spec)

    opt_state = optimizer.init(model_params)

    model_params = jax.device_put(model_params, device)
    opt_state = jax.device_put(opt_state, device)

    train_step = train_setup.make_train_step_fn(
        optimizer, model_static, dataset, hyperparams
    )

    # Build test loss function (forward pass only, no gradients)
    test_loss_fn = train_setup.make_test_loss_fn(model_static, hyperparams)

    compute_pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn" if sys.platform == "win32" else "forkserver"),
    )
    log_thread = ThreadPoolExecutor(max_workers=1)

    with (
        mlflow.start_run(
            run_name=f"{run_name}_bs{hyperparams.training.batch_size}_lr{hyperparams.training.lr_init}_{device.platform}",
            nested=True,
            parent_run_id=parent_id,
        ),
        crash_logger() as crash_state,
    ):
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run.")
        train_setup.sampler.setup(
            model,
            dataset,
            hyperparams,
            compute_pool,
            log_thread,
            active_run.info.run_id,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            hyperparams_to_json(hyperparams, Path(tmpdir) / "hyperparams.json")
            mlflow.log_artifact(str(Path(tmpdir) / "hyperparams.json"))

        log_compute_to_mlflow()
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            mlflow.set_tag("commit_hash", commit_hash)
        except Exception:
            pass
        mlflow.log_params(
            {
                "model_type": model_type_name,
                "n_files": len(files),
                "device": device.platform,
                "start_step": start_step,
            }
        )

        def _flatten(d: dict, prefix: str = "") -> dict:
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    out.update(_flatten(v, key))
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            out.update(_flatten(item, f"{key}.{i}"))
                        else:
                            out[f"{key}.{i}"] = item
                else:
                    out[key] = v
            return out

        mlflow.log_params(_flatten(asdict(hyperparams)))

        # Log annealing summary params for easy run comparison
        def _anneal_summary(cfg, prefix: str) -> dict:
            # Per-channel config: list[list[AnnealConfig]] → summarise first channel
            if isinstance(cfg, list) and len(cfg) > 0 and isinstance(cfg[0], list):
                phases = cfg[0]
                n_channels = len(cfg)
            else:
                phases = cfg if isinstance(cfg, list) else [cfg]
                n_channels = 1
            summary = {
                f"{prefix}_n_phases": len(phases),
                f"{prefix}_initial_weight": phases[0].initial_weight,
                f"{prefix}_final_weight": phases[-1].final_weight,
                f"{prefix}_total_steps": int(phases[-1].end - phases[0].start),
                f"{prefix}_n_channels": n_channels,
            }
            strategies = {p.annealing_strategy.value for p in phases}
            if len(strategies) == 1:
                summary[f"{prefix}_strategy"] = strategies.pop()
            else:
                summary[f"{prefix}_strategy"] = "mixed"
            return summary

        mlflow.log_params(
            {
                **_anneal_summary(hyperparams.training.kl_annealing, "kl"),
                **_anneal_summary(hyperparams.training.noise_annealing, "noise"),
            }
        )

        best_loss = float("inf")
        best_test_loss = float("inf")
        prev_test_loss = float("inf")
        test_loss_no_improve = 0
        test_loss_patience = 400

        train_start_time = time.perf_counter()
        interval_start_time = time.perf_counter()
        window_metrics: list[dict] = []
        step_logs: list[tuple[int, dict[str, float]]] = []

        def curriculum_length(step: int) -> int:
            """Return the sample_length for this step per the curriculum schedule.

            The curriculum stages are specified in seconds.  For datasets
            built on a 0.05 s base timestep with resampling via
            ``resample_every``, the sample_length is in base-dt units.
            For :class:`ConsolidatedDataset` where resampling is
            already applied, sample_length is in resampled-dt units.
            We detect which convention to use from the dataset type.
            """
            # Determine the timestep that sample_length is counted in.
            if hasattr(dataset, "resample_dt") and dataset.resample_dt is not None:
                # ConsolidatedDataset — sample_length in resampled steps
                sl_dt = dataset.resample_dt
            else:
                # JAXParquetDataset — sample_length in base-dt steps (0.05 s)
                sl_dt = 0.05

            curriculum = hyperparams.training.curriculum
            if not curriculum:
                return int(hyperparams.data.sample_length / sl_dt)
            length = int(hyperparams.data.sample_length / sl_dt)
            for threshold_step, seg_length in sorted(curriculum):
                if step >= threshold_step:
                    length = int(seg_length / sl_dt)
            return length

        current_curriculum_length = curriculum_length(start_step)
        dataset.sample_length = current_curriculum_length
        logger.log(
            f"[Curriculum] step={start_step} → sample_length={current_curriculum_length}"
        )

        remaining_steps = hyperparams.training.num_steps - start_step
        if remaining_steps <= 0:
            logger.log(
                f"[SKIP] start_step={start_step} >= num_steps={hyperparams.training.num_steps}"
            )
            return

        for step_offset, batch in enumerate(
            dataset.iter_steps(
                hyperparams.training.batch_size,
                num_steps=remaining_steps,
                key=train_key,
                device=device,
                prefetch=4,
            )
        ):
            step = start_step + step_offset
            # Normalise batch: (ts, xs, meta) or (ts, xs, cond, meta)
            if len(batch) == 4:
                ts, xs, cond, _meta = batch
            else:
                ts, xs, _meta = batch
                cond = None

            kl_weight, log_sigma = anneal_params(hyperparams, step)
            step_key = jr.fold_in(train_key, step)

            # Curriculum: update dataset sample_length when stage changes
            new_curriculum_length = curriculum_length(step)
            if new_curriculum_length != current_curriculum_length:
                current_curriculum_length = new_curriculum_length
                dataset.sample_length = current_curriculum_length
                logger.log(
                    f"[Curriculum] step={step} → sample_length={current_curriculum_length}"
                )
                mlflow.log_metric(
                    "curriculum_sample_length", current_curriculum_length, step=step
                )

            _step_kwargs = dict(
                key=step_key,
                kl_weight=kl_weight,
                log_sigma=log_sigma,
                cond=cond,
            )
            model_params, opt_state, step_metrics = train_step(
                model_params,
                opt_state,
                xs,
                ts,
                **_step_kwargs,
            )

            crash_state.update(
                step=step,
                best_loss=best_loss,
                **step_metrics,
            )

            window_metrics.append(step_metrics)

            # Log epoch (number of full passes through the training data)
            mlflow.log_metric("epoch", dataset.current_epoch, step=step)

            interval_start_time = log_batch(
                logger=logger,
                hyperparams=hyperparams,
                step=step,
                metrics=step_metrics,
                schedule=schedule,
                step_logs=step_logs,
                interval_start_time=interval_start_time,
                print=print_batch,
            )

            if (
                step % hyperparams.training.log_every == 0
                and step > 0
                and window_metrics
            ) or step == hyperparams.training.num_steps - 1:
                current_model = eqx.combine(model_params, model_static)
                best_loss = log_summary(
                    logger=logger,
                    step=step,
                    model=current_model,
                    best_loss=best_loss,
                    window_metrics=window_metrics,
                    step_logs=step_logs,
                    train_start_time=train_start_time,
                )
                window_metrics.clear()
                step_logs.clear()

            if step % hyperparams.training.checkpoint_every == 0 and step > 0:
                current_model = eqx.combine(model_params, model_static)
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / f"checkpoint_{step:07d}.eqx"
                    eqx.tree_serialise_leaves(tmp_path, current_model)
                    mlflow.log_artifact(str(tmp_path), artifact_path="checkpoint")

            if (
                step % hyperparams.training.sample_every == 0
                and step > 0
                or step == hyperparams.training.num_steps - 1
            ):
                # --- Test loss on held-out data ---
                # Skip first sample_every step to avoid overlapping JIT
                # compilation memory with sample_fn's first compilation.
                first_sample_step = step == hyperparams.training.sample_every
                if (
                    test_loss_fn is not None
                    and test_dataset is not None
                    and not first_sample_step
                ):
                    test_key = jr.fold_in(train_key, step + 2**30)
                    n_test = min(hyperparams.training.batch_size, len(test_dataset))
                    test_batch = test_dataset.sample_random(test_key, n_test)
                    if len(test_batch) == 4:
                        test_ts, test_xs, test_cond, _ = test_batch
                    else:
                        test_ts, test_xs, _ = test_batch
                        test_cond = None
                    test_ts = jax.device_put(test_ts, device)
                    test_xs = jax.device_put(test_xs, device)
                    if test_cond is not None:
                        test_cond = jax.device_put(test_cond, device)
                    _test_kwargs = dict(
                        key=test_key,
                        kl_weight=kl_weight,
                        log_sigma=log_sigma,
                        cond=test_cond,
                    )
                    test_metrics = test_loss_fn(
                        model_params,
                        test_xs,
                        test_ts,
                        **_test_kwargs,
                    )
                    test_metrics = {
                        f"test/{k}": float(v)
                        for k, v in test_metrics.items()
                        if math.isfinite(float(v))
                    }
                    mlflow.log_metrics(test_metrics, step=step)
                    logger.log(
                        f"[Test] step={step} "
                        + " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
                    )

                    # Early stopping on test loss
                    current_test_loss = test_metrics.get("test/loss", float("inf"))
                    if current_test_loss < best_test_loss:
                        best_test_loss = current_test_loss
                    # Reset patience if loss decreased vs previous eval
                    # (tolerates spikes from curriculum changes)
                    if current_test_loss < prev_test_loss:
                        test_loss_no_improve = 0
                    else:
                        test_loss_no_improve += 1
                    prev_test_loss = current_test_loss
                    mlflow.log_metric(
                        "test_loss_no_improve", test_loss_no_improve, step=step
                    )
                    # Free GPU memory before sampling
                    del test_ts, test_xs, test_cond

                current_model = eqx.combine(model_params, model_static)
                train_setup.sampler.sample(
                    current_model,
                    step,
                    step_key,
                    logger=logger,
                )

            if not math.isfinite(step_metrics["loss"]):
                reason = f"NaN/Inf window loss at step {step}"
                logger.log(f"[COLLAPSE] {reason}. Stopping.")
                mlflow.log_param("early_stop_reason", reason)
                break

            if test_loss_patience > 0 and test_loss_no_improve >= test_loss_patience:
                reason = f"Test loss did not improve for {test_loss_patience} evaluations (best={best_test_loss:.4f})"
                logger.log(f"[EARLY STOP] {reason}")
                mlflow.log_param("early_stop_reason", reason)
                break

        compute_pool.shutdown(wait=True)
        log_thread.shutdown(wait=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "final.eqx"
            final_model = eqx.combine(model_params, model_static)
            eqx.tree_serialise_leaves(tmp_path, final_model)
            mlflow.log_artifact(str(tmp_path), artifact_path="checkpoint")

        wall_time = time.perf_counter() - train_start_time
        mlflow.log_metric("wall_time", wall_time)
        print(f"Training complete. Best loss: {best_loss:.4f}")

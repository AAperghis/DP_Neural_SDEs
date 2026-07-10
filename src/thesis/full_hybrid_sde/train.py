"""Training pipeline for the FullHybridSDE model.

Provides model initialisation, loss functions, train-step construction,
sample generation, and an entry-point ``main()`` that delegates to the
shared :func:`thesis.shared.train.train` loop.

The pipeline mirrors :mod:`thesis.reduced_hybrid_sde.train` and uses:

* **Consolidated parquet dataset** — single-file I/O with per-run wave
  metadata.
* **Wave-conditioned SDE** — ``(Hs, Tp, beta_w)`` is passed through to
  the model on every forward pass via the ``cond`` keyword.
* **Azimuth thruster physics** — via
  :class:`~thesis.full_order_dp.reduced_order_model.NominalDynamicsFO`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.typing as jtp
import jax.numpy as jnp
import jax.random as jr
import mlflow
import optax

from thesis.full_hybrid_sde.model import FullHybridSDE
from thesis.full_hybrid_sde.nominal_dynamics import default_mss_physics_config
from thesis.full_hybrid_sde.sampler import FOHybridSampler
from thesis.shared.consolidated_dataset import ConsolidatedDataset
from thesis.shared.data_structures import (
    AnnealConfig,
    AnnealStrategy,
    DataConfig,
    FieldConfig,
    FieldType,
    HyperParameters,
    FO_3DOF_FEATURES,
    ModelConfig,
    TrainingConfig,
    hyperparams_from_json,
)
from thesis.shared.train import train
from thesis.shared.train_setup import TrainSetup


# Indices of eta channels in the standard FO feature layout (eta_0, eta_1, eta_5).
_ETA_IDX = (0, 1, 2)


# ======================================================================
# Loss functions
# ======================================================================


def loss_fn(
    sde: FullHybridSDE,
    xs: jax.Array,
    ts: jax.Array,
    wave_cond: jax.Array,
    *,
    key,
    kl_weight: jax.Array = jnp.array(1.0),
    log_sigma: jax.Array = jnp.array(-5.0),
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """ELBO loss for a single trajectory."""
    zs, xs_hat, log_pxs, kl_initial, kl_path = sde(
        ts,
        xs,
        key=key,
        log_sigma=log_sigma,
        wave_cond=wave_cond,
    )
    loss = -log_pxs + (kl_initial + kl_path) * kl_weight
    return loss, (kl_initial, kl_path, log_pxs)


@eqx.filter_value_and_grad(has_aux=True)
def batch_loss_fn(
    sde: FullHybridSDE,
    xs_batch: jax.Array,
    ts: jax.Array,
    wave_cond_batch: jax.Array,
    *,
    key,
    kl_weight: jax.Array = jnp.array(1.0),
    log_sigma: jax.Array = jnp.array(-5.0),
    sens_weight: jax.Array = jnp.array(0.0),
    moment_weight: jax.Array = jnp.array(0.0),
    aux_active: jax.Array = jnp.array(False),
    aux_subbatch: int = 32,
) -> tuple[
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array],
]:
    """Batch-mean ELBO + optional prior-based aux losses, with gradients.

    Aux losses fire only when ``aux_active`` is True (a traced scalar so
    JIT cost amortises). They use the first ``aux_subbatch`` items of
    the batch.
    """
    batch_size = xs_batch.shape[0]
    elbo_key, aux_key = jr.split(key)
    keys = jr.split(elbo_key, batch_size)

    losses, (kl_initials, kl_paths, log_pxs) = jax.vmap(
        lambda x, wc, k: loss_fn(
            sde,
            x,
            ts,
            wc,
            key=k,
            kl_weight=kl_weight,
            log_sigma=log_sigma,
        )
    )(xs_batch, wave_cond_batch, keys)
    elbo_mean = jnp.mean(losses)

    total = elbo_mean
    return (
        total,
        (
            jnp.mean(kl_initials),
            jnp.mean(kl_paths),
            jnp.mean(log_pxs),
        ),
    )


# ======================================================================
# Gradient scaling
# ======================================================================


def increase_update_initial(
    updates: optax.Updates, sde: FullHybridSDE
) -> optax.Updates:
    """Scale initial-condition parameter updates by 10."""
    initial_leaves = lambda u: [
        u.pz0_mean,
        u.pz0_logvar,
        u.qz0_posterior.weight,
        u.qz0_posterior.bias,
    ]
    return eqx.tree_at(initial_leaves, updates, replace_fn=lambda x: x * 10)


def init_model(hyperparams: HyperParameters, key: jax.Array) -> FullHybridSDE:
    model_key, _, _ = jr.split(key, 3)
    f_key, h_key, g_key, model_key = jr.split(model_key, 4)
    mc = hyperparams.model

    f_config = replace(mc.f_config, key=f_key)
    h_config = replace(mc.h_config, key=h_key)
    g_config = replace(mc.g_config, key=g_key)

    return FullHybridSDE(
        hyperparams.data.data_size,
        mc.latent_size,
        mc.ctx_size,
        mc.hidden_size,
        f_config,
        h_config,
        g_config,
        phys_config=mc.fo_physics_config,
        n_wave_params=hyperparams.data.n_wave_params,
        dt=hyperparams.data.dt,
        indirect_eta=mc.indirect_eta,
        kl_eps=mc.kl_eps,
        key=model_key,
    )


class FOHybridTrainSetup(TrainSetup):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sampler", FOHybridSampler())
        super().__init__(*args, **kwargs)

    def make_schedule_fn(self, hyperparams: HyperParameters) -> optax.Schedule:
        tc = hyperparams.training
        warmup_steps = max(1, int(tc.lr_warmup_fraction * tc.num_steps))
        decay_steps = tc.num_steps - warmup_steps
        warmup = optax.linear_schedule(0.0, tc.lr_init, warmup_steps)
        decay = optax.exponential_decay(
            tc.lr_init,
            decay_steps,
            tc.lr_end / tc.lr_init,
            end_value=tc.lr_end,
        )
        return optax.join_schedules([warmup, decay], [warmup_steps])

    def make_train_step_fn(self, optimizer, model_static, dataset, hyperparams):
        """Build a JIT-compiled training step that accepts wave conditioning.

        Maintains an internal Python-side step counter so the shared training
        loop's contract (no ``step`` kwarg) is preserved while still feeding a
        traced scalar to the JIT body for the aux-loss skipping branch.
        """

        @eqx.filter_jit
        def train(
            sde_params,
            opt_state: optax.OptState,
            xs_batch: jax.Array,
            ts: jax.Array,
            *,
            key: jax.Array,
            kl_weight: jax.Array,
            log_sigma: jax.Array,
            cond: jax.Array,
        ):
            sde = eqx.combine(sde_params, model_static)

            (total_loss, (kl_initial, kl_path, log_pxs)), grads = batch_loss_fn(
                sde,
                xs_batch,
                ts,
                cond,
                key=key,
                kl_weight=kl_weight,
                log_sigma=log_sigma,
            )

            grads = increase_update_initial(grads, sde)
            grads = jax.tree_util.tree_map(
                lambda p, g: g if p is not None else None,
                sde_params,
                grads,
                is_leaf=lambda x: x is None,
            )
            updates, opt_state_new = optimizer.update(grads, opt_state, sde_params)
            sde = eqx.apply_updates(sde, updates)
            all_arrays = eqx.filter(sde, eqx.is_inexact_array)
            sde_params_new = jax.tree_util.tree_map(
                lambda p, a: a if p is not None else None,
                sde_params,
                all_arrays,
                is_leaf=lambda x: x is None,
            )

            grad_norm = optax.global_norm(grads)
            elbo = total_loss
            metrics = {
                "loss": total_loss,
                "elbo": elbo,
                "kl_initial": kl_initial,
                "kl_path": kl_path,
                "log_pxs": -log_pxs,
                "kl_weight": kl_weight,
                "grad_norm": grad_norm,
                "kl_path_weighted": kl_path * kl_weight,
                "kl_initial_weighted": kl_initial * kl_weight,
            }
            if log_sigma.shape == ():
                metrics["log_sigma"] = log_sigma
            else:
                metrics["log_sigma_mean"] = jnp.mean(log_sigma)
            return sde_params_new, opt_state_new, metrics

        return train

    def make_test_loss_fn(self, model_static, hyperparams):
        @eqx.filter_jit
        def _test(
            sde_params,
            xs_batch,
            ts,
            *,
            key,
            kl_weight,
            log_sigma,
            cond,
        ):
            sde = eqx.combine(sde_params, model_static)
            batch_size = xs_batch.shape[0]
            keys = jr.split(key, batch_size)
            losses, (kl_initials, kl_paths, log_pxs) = jax.vmap(
                lambda x, wc, k: loss_fn(
                    sde,
                    x,
                    ts,
                    wc,
                    key=k,
                    kl_weight=kl_weight,
                    log_sigma=log_sigma,
                )
            )(xs_batch, cond, keys)
            return {
                "loss": jnp.mean(losses),
                "kl_initial": jnp.mean(kl_initials),
                "kl_path": jnp.mean(kl_paths),
                "log_pxs": -jnp.mean(log_pxs),
            }

        return _test
    
    def init_model_fn(self, hyperparams: HyperParameters, key: jax.Array) -> FullHybridSDE:
        return init_model(hyperparams, key)


# ======================================================================
# Entry point
# ======================================================================


def main(
    data_path: Path,
    hyperparams: HyperParameters | None = None,
    device: str = "gpu",
    run_name: str = "full_hybrid_sde",
    parent_id: str | None = None,
) -> None:
    """Convenience entry point with default MSS hyper-parameters.

    Builds a :class:`ConsolidatedDataset` and delegates to the
    shared :func:`thesis.shared.train.train` loop.
    """
    if hyperparams is None:
        phys_config = default_mss_physics_config()
        hyperparams = HyperParameters(
            model=ModelConfig(
                f_config=FieldConfig(
                    field_type=FieldType.CONTEXT_STATE,
                    latent_size=64,
                    hidden_layer_width=128,
                    depth=3,
                    context_size=64,
                ),
                h_config=FieldConfig(
                    field_type=FieldType.STATE,
                    latent_size=64,
                    hidden_layer_width=128,
                    depth=6,
                    mean_reversion=True,
                ),
                g_config=FieldConfig(
                    field_type=FieldType.STATE,
                    latent_size=64,
                    hidden_layer_width=128,
                    depth=3,
                    control_size=64,
                    final_activation="sigmoid",
                    diagonal=True,
                ),
                fo_physics_config=phys_config,
                indirect_eta=True,
            ),
            training=TrainingConfig(
                lr_init=1e-3,
                lr_end=1e-6,
                lr_warmup_fraction=0.1,
                num_steps=5000,
                batch_size=128,
                log_every=1,
                sample_every=100,
                checkpoint_every=100,
                kl_annealing=[
                    AnnealConfig(
                        start=0,
                        end=3000,
                        warmup=0.0,
                        initial_weight=1.0,
                        final_weight=5.0,
                        annealing_strategy=AnnealStrategy.COSINE,
                    )
                ],
                noise_annealing=[
                    AnnealConfig(
                        start=0,
                        end=3000,
                        warmup=0.0,
                        initial_weight=-1.0,
                        final_weight=-3.5,
                        annealing_strategy=AnnealStrategy.COSINE,
                    )
                ],
                curriculum=(
                    (0, 50),
                    (500, 200),
                    (1000, 500),
                    (1500, 2000),
                ),
                max_grad_norm=5000.0,
            ),
            data=DataConfig(
                features=FO_3DOF_FEATURES,
                dt=0.5,
                sample_length=2000,
                n_files=128,
                truncate_seconds=600.0,
                group_scaling=True,
            ),
        )

    # Build consolidated dataset with wave conditioning
    hs_filter = None
    if hyperparams.data.hs_max is not None:
        hs_limit = hyperparams.data.hs_max
        hs_filter = lambda m: float(m.get("Hs", 0.0)) < hs_limit

    dataset = ConsolidatedDataset(
        data_path,
        columns=hyperparams.data.features,
        wave_keys=hyperparams.data.wave_keys,
        angular_wave_keys=hyperparams.data.angular_wave_keys,
        sample_length=int(hyperparams.data.sample_length),
        resample_dt=hyperparams.data.dt,
        standardise=True,
        truncate_seconds=hyperparams.data.truncate_seconds,
        group_scaling=hyperparams.data.group_scaling,
        n_runs=hyperparams.data.n_files,
        filter_fn=hs_filter,
    )

    setup = FOHybridTrainSetup(
        hyperparams,
        str(data_path),
        model_type_name="FullHybridSDE",
        device=device,
        run_name=run_name,
        parent_id=parent_id,
        dataset=dataset,
    )
    train(setup)


def resume(
    data_path: Path,
    source_run_id: str,
    checkpoint: str = "final",
    *,
    device: str = "gpu",
    run_name: str = "full_hybrid_sde_resume",
    parent_id: str | None = None,
    extra_steps: int | None = None,
    hyperparams_override: HyperParameters | None = None,
) -> None:
    """Resume training a FullHybridSDE from an MLflow checkpoint.

    Downloads the hyperparameters and checkpoint from *source_run_id*,
    reconstructs the model, and continues training.  The optimizer
    state is re-initialised (momentum is lost).

    Args:
        data_path: Path to consolidated parquet file.
        source_run_id: MLflow run ID to load checkpoint from.
        checkpoint: Checkpoint name — ``"final"`` or step number
            (e.g. ``"1000"``).
        device: ``'gpu'`` or ``'cpu'``.
        run_name: MLflow run name prefix for the new run.
        parent_id: Parent MLflow run ID for nesting.
        extra_steps: If set, overrides ``num_steps`` to
            ``start_step + extra_steps`` so the model trains for
            exactly this many additional steps.
        hyperparams_override: If set, use these hyperparams instead of
            the ones from the source run.  Useful for changing schedules.
    """

    # Download artifacts from source run
    hp_path = mlflow.artifacts.download_artifacts(
        run_id=source_run_id, artifact_path="hyperparams.json"
    )
    hyperparams = hyperparams_override or hyperparams_from_json(hp_path)

    if checkpoint == "final":
        artifact_name = "checkpoint/final.eqx"
        start_step = hyperparams.training.num_steps
    else:
        step_num = int(checkpoint)
        artifact_name = f"checkpoint/checkpoint_{step_num:07d}.eqx"
        start_step = step_num

    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=source_run_id,
        artifact_path=artifact_name,
    )

    # Reconstruct model and load checkpoint weights
    key = jr.key(0)
    model = FOHybridTrainSetup(hyperparams, str(data_path)).init_model_fn(
        hyperparams, key
    )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)

    # Adjust num_steps if extra_steps is given
    if extra_steps is not None:
        hyperparams = replace(
            hyperparams,
            training=replace(
                hyperparams.training,
                num_steps=start_step + extra_steps,
            ),
        )

    # Build dataset
    hs_filter = None
    if hyperparams.data.hs_max is not None:
        hs_limit = hyperparams.data.hs_max
        hs_filter = lambda m: float(m.get("Hs", 0.0)) < hs_limit

    dataset = ConsolidatedDataset(
        data_path,
        columns=hyperparams.data.features,
        wave_keys=hyperparams.data.wave_keys,
        angular_wave_keys=hyperparams.data.angular_wave_keys,
        sample_length=int(hyperparams.data.sample_length),
        resample_dt=hyperparams.data.dt,
        standardise=True,
        truncate_seconds=hyperparams.data.truncate_seconds,
        group_scaling=hyperparams.data.group_scaling,
        n_runs=hyperparams.data.n_files,
        filter_fn=hs_filter,
    )

    # Re-apply dataset standardisation stats to the model
    model = eqx.tree_at(
        lambda m: (m.data_mean, m.data_std),
        model,
        (
            jnp.array(dataset.standardise["mean"], dtype=jnp.float32),
            jnp.array(dataset.standardise["std"], dtype=jnp.float32),
        ),
    )

    print(
        f"Resuming from run {source_run_id}, checkpoint={checkpoint} "
        f"(step {start_step}), training to step {hyperparams.training.num_steps}"
    )

    setup = FOHybridTrainSetup(
        hyperparams,
        str(data_path),
        initialised_model=model,
        model_type_name="FullHybridSDE",
        device=device,
        run_name=run_name,
        parent_id=parent_id,
        dataset=dataset,
        start_step=start_step,
    )
    train(setup)


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train FullHybridSDE")
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Consolidated parquet file",
        default=Path(
            "/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/consolidated/fo_full_state.parquet"
        ),
    )
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--run_name", default="full_hybrid_sde")
    parser.add_argument(
        "--hyperparams", default=None, help="Path to JSON file with hyper-parameters"
    )
    parser.add_argument(
        "--run_id", default=None, help="Parent run ID for nested experiment"
    )
    args = parser.parse_args()

    if args.hyperparams is not None:
        hyperparams = hyperparams_from_json(args.hyperparams)

    mlflow.set_experiment(experiment_id="3853466280664740")
    main(
        args.data_path,
        device=args.device,
        run_name=args.run_name,
        hyperparams=hyperparams,
        parent_id=args.run_id,
    )


def resume_cli():
    """CLI entry point for resuming training from an MLflow checkpoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Resume FullHybridSDE training")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source_run_id", default=None, help="MLflow run ID to resume from"
    )
    source_group.add_argument(
        "--source_run_name",
        default=None,
        help="Name of the source child run to resume from; resolved under --run_id (parent)",
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="Checkpoint: 'final' or step number (e.g. '1000')",
    )
    parser.add_argument(
        "--extra_steps",
        type=int,
        default=None,
        help="Number of additional steps to train",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Consolidated parquet file",
        default=Path(
            "/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/consolidated/fo_full_state.parquet"
        ),
    )
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--run_name", default="full_hybrid_sde_resume")
    parser.add_argument(
        "--hyperparams",
        default=None,
        help="Override hyperparams JSON (uses source run's if omitted)",
    )
    parser.add_argument(
        "--run_id", default=None, help="Parent run ID for nested experiment"
    )
    args = parser.parse_args()

    if args.source_run_id:
        source_run_id = args.source_run_id
    else:
        if not args.run_id:
            parser.error(
                "--source_run_name requires --run_id (parent run ID) to search under"
            )
        mlflow.set_tracking_uri("databricks://dev")
        client = mlflow.MlflowClient()
        parent_run = client.get_run(args.run_id)
        experiment_id = parent_run.info.experiment_id
        found = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.`mlflow.parentRunId` = '{args.run_id}' "
                f"AND attributes.run_name = '{args.source_run_name}'"
            ),
        )
        if not found:
            raise ValueError(
                f"No child run named '{args.source_run_name}' found under parent {args.run_id}"
            )
        source_run_id = found[0].info.run_id
        print(f"Resolved source run '{args.source_run_name}' → {source_run_id}")

    hp_override = None
    if args.hyperparams is not None:
        hp_override = hyperparams_from_json(args.hyperparams)

    mlflow.set_experiment(experiment_id="3853466280664740")
    resume(
        args.data_path,
        source_run_id=source_run_id,
        checkpoint=args.checkpoint,
        device=args.device,
        run_name=args.run_name,
        parent_id=args.run_id,
        extra_steps=args.extra_steps,
        hyperparams_override=hp_override,
    )


if __name__ == "__main__":
    cli()

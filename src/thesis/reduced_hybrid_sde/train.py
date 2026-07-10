"""Training pipeline for the ReducedHybridSDE model.

Orchestrates model initialisation, ELBO loss computation, optimiser
construction, sample generation, and the main training loop backed by
MLflow logging.  Rendering and statistics are offloaded to a
:class:`~concurrent.futures.ProcessPoolExecutor` so the GPU is not
blocked by matplotlib work.
"""

from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.typing as jtp
import mlflow
import optax

from thesis.reduced_hybrid_sde.model import ReducedHybridSDE
from thesis.shared.data_structures import (
    AnnealConfig,
    AnnealStrategy,
    DataConfig,
    Features,
    FieldConfig,
    FieldType,
    HyperParameters,
    ModelConfig,
    PhysicsConfig,
    TrainingConfig,
    hyperparams_from_json,
)
from thesis.shared.train import train
from thesis.shared.train_setup import TrainSetup
from thesis.reduced_hybrid_sde.sampler import ROHybridSampler
from thesis.reduced_order_dp.supply import SupplyVessel

mlflow.enable_system_metrics_logging()


def loss_fn(
    sde: ReducedHybridSDE,
    xs: jax.Array,
    ts: jax.Array,
    *,
    key,
    kl_weight: jax.Array = jnp.array(1.0),
    log_sigma: jax.Array = jnp.array(-5.0),
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """ELBO loss: -log p(x|z) + kl_weight * KL + physics constraints.

    Args:
        sde: ReducedHybridSDE model (single sample, no batch dim).
        xs: Observations, shape (T, data_size).
        ts: Timestamps, shape (T,).
        key: PRNG key.
        kl_weight: Annealing weight for KL term (0 → 1 over training).
        log_sigma: Log observation noise standard deviation.

    Returns:
        A tuple ``(loss, aux)`` where *loss* is the scalar negative ELBO
        (including physics constraints) and *aux* is a tuple of
        ``(kl_initial, kl_path, log_pxs)``.
    """
    print("[DEBUG]: Compiled loss")
    zs, xs_hat, log_pxs, kl_initial, kl_path = sde(ts, xs, key=key, log_sigma=log_sigma)
    loss = -log_pxs + (kl_initial + kl_path) * kl_weight
    return loss, (kl_initial, kl_path, log_pxs)


@eqx.filter_value_and_grad(has_aux=True)
def batch_loss_fn(
    sde: ReducedHybridSDE,
    xs_batch: jax.Array,
    ts: jax.Array,
    *,
    key,
    kl_weight: jax.Array = jnp.array(1.0),
    log_sigma: jax.Array = jnp.array(-5.0),
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Compute the mean ELBO loss over a batch and return gradients.

    Wrapped by :func:`eqx.filter_value_and_grad` so it returns both the
    loss value (with aux) and the parameter gradients.

    Args:
        sde: ReducedHybridSDE model.
        xs_batch: Observation batch, shape ``(batch, T, data_size)``.
        ts: Shared timestamps, shape ``(T,)``.
        key: PRNG key.
        kl_weight: Annealing weight for the KL term.
        log_sigma: Log observation noise standard deviation.

    Returns:
        A tuple ``(loss, aux)`` with the batch-mean scalar loss and
        ``(kl_initial, kl_path, log_pxs)``.
    """
    print("[DEBUG]: Compiled batch loss")
    batch_size = xs_batch.shape[0]
    keys = jr.split(key, batch_size)

    losses, (kl_initials, kl_paths, log_pxs) = jax.vmap(
        lambda x, k: loss_fn(
            sde,
            x,
            ts,
            key=k,
            kl_weight=kl_weight,
            log_sigma=log_sigma,
        )
    )(xs_batch, keys)

    return (
        jnp.mean(losses),
        (
            jnp.mean(kl_initials),
            jnp.mean(kl_paths),
            jnp.mean(log_pxs),
        ),
    )


# def make_mask(sde_example: ReducedHybridSDE) -> eqx.Module:
#     """Create a boolean mask that selects initial-condition parameters.

#     Returns a pytree matching *sde_example* where leaves corresponding to
#     ``pz0_mean``, ``pz0_logvar``, ``px0_mean``, and ``qz0_posterior`` are ``True`` and
#     all other leaves are ``False``.

#     Args:
#         sde_example: An instance of :class:`ReducedHybridSDE` used as a
#             structural template.

#     Returns:
#         A pytree of booleans with the same structure as the trainable
#         parameters of *sde_example*.
#     """
#     params, _ = eqx.partition(sde_example, eqx.is_inexact_array)
#     mask = jax.tree_util.tree_map(lambda _: False, params)
#     mask = eqx.tree_at(
#         lambda m: (
#             m.pz0_mean,
#             m.pz0_logvar,
#             m.px0_mean,
#             m.qz0_posterior.weight,
#             m.qz0_posterior.bias,
#         ),
#         mask,
#         replace=(True, True, True, True),
#     )
#     return mask


def increase_update_initial(updates: optax.Updates, sde: ReducedHybridSDE) -> optax.Updates:
    """Scale gradient updates for initial-condition parameters by 10x.

    Applies a larger learning rate to ``pz0_mean``, ``pz0_logvar``, and
    ``qz0_posterior`` so the latent initial-state distribution converges
    faster than the dynamics networks.

    Args:
        updates: Pytree of parameter updates from the optimiser.
        sde: Model instance (used only for tree structure reference).

    Returns:
        Updated pytree with the selected leaves scaled by 10.
    """
    initial_leaves = lambda u: [
        u.pz0_mean,
        u.pz0_logvar,
        u.qz0_posterior.weight,
        u.qz0_posterior.bias,
    ]
    return eqx.tree_at(initial_leaves, updates, replace_fn=lambda x: x * 10)


def make_schedule_exp(hyperparams: HyperParameters) -> optax.Schedule:
    """Construct a learning rate schedule from hyper-parameters.

    Uses a linear warmup followed by exponential decay to ``lr_end``.

    Args:
        hyperparams: Training hyper-parameters that define the schedule.

    Returns:
        An Optax learning rate schedule function that maps step numbers to
        learning rates.
    """
    tc = hyperparams.training
    warmup_steps = max(1, int(tc.lr_warmup_fraction * tc.num_steps))
    decay_steps = tc.num_steps - warmup_steps
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=tc.lr_init,
        transition_steps=warmup_steps,
    )
    decay = optax.exponential_decay(
        init_value=tc.lr_init,
        transition_steps=decay_steps,
        decay_rate=tc.lr_end / tc.lr_init,
        end_value=tc.lr_end,
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps],
    )


def make_schedule_one_cycle(hyperparams: HyperParameters) -> optax.Schedule:
    """Construct a one-cycle learning rate schedule from hyper-parameters.

    Linearly increases the learning rate from 0 to ``lr_init`` over the
    first half of training, then linearly decreases back to 0.

    Args:
        hyperparams: Training hyper-parameters that define the schedule.

    Returns:
        An Optax learning rate schedule function that maps step numbers to
        learning rates.
    """
    tc = hyperparams.training
    return optax.cosine_onecycle_schedule(
        transition_steps=int(tc.num_steps * 0.6),
        peak_value=tc.lr_init,
        pct_start=tc.lr_warmup_fraction,
        div_factor=10,
        final_div_factor=tc.lr_init / tc.lr_end,
    )


def make_train_step(
    optimizer: optax.GradientTransformation,
    model_static: eqx.Module,
    dataset,
    hyperparams: HyperParameters,
):
    """Build a JIT-compiled training step closure.

    Captures the optimizer, static model leaves, and physics-constraint
    constants so the returned function only receives mutable state.

    Args:
        optimizer: Optax optimizer (e.g. AdamW + scheduler).
        model_static: Non-trainable (static) partition of the model.
        dataset: Training dataset; must expose ``standardise`` with
            ``"mean"`` and ``"std"`` keys.
        hyperparams: Full set of training hyper-parameters.

    Returns:
        A JIT-compiled ``_step`` function with signature::

            _step(sde_params, opt_state, xs_batch, ts, *,
                  key, kl_weight, log_sigma)
            -> (sde_params_new, opt_state_new, metrics_dict)
    """

    @eqx.filter_jit
    def _step(
        sde_params,
        opt_state: optax.OptState,
        xs_batch: jax.Array,
        ts: jax.Array,
        *,
        key: jax.Array,
        kl_weight: jax.Array,
        log_sigma: jax.Array,
        cond: jax.Array | None = None,
    ):
        sde = cast(ReducedHybridSDE, eqx.combine(sde_params, model_static))

        (elbo, (kl_initial, kl_path, log_pxs)), grads = batch_loss_fn(
            sde,
            xs_batch,
            ts,
            key=key,
            kl_weight=kl_weight,
            log_sigma=log_sigma,
        )

        grads = increase_update_initial(grads, sde)
        # Filter grads to match sde_params (exclude frozen ODE parameters)
        grads = jax.tree_util.tree_map(
            lambda p, g: g if p is not None else None,
            sde_params,
            grads,
            is_leaf=lambda x: x is None,
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, sde_params)
        sde = eqx.apply_updates(sde, updates)
        # Preserve partition structure (keep frozen ODE params as None)
        all_arrays = eqx.filter(sde, eqx.is_inexact_array)
        sde_params_new = jax.tree_util.tree_map(
            lambda p, a: a if p is not None else None,
            sde_params,
            all_arrays,
            is_leaf=lambda x: x is None,
        )
        grad_norm = optax.global_norm(grads)
        metrics = {
            "loss": elbo,
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
            metrics["log_sigma_eta"] = jnp.mean(log_sigma[:3])
            metrics["log_sigma_nu"] = jnp.mean(log_sigma[3:6])
            metrics["log_sigma_rpm"] = jnp.mean(log_sigma[6:])

        return sde_params_new, opt_state_new, metrics

    return _step


def init_model(hyperparams: HyperParameters, key: jax.Array) -> ReducedHybridSDE:
    """Instantiate a :class:`ReducedHybridSDE` from hyper-parameters.

    Uses the :class:`FieldConfig` objects stored in
    ``hyperparams.model`` directly, injecting fresh PRNG keys.

    Args:
        hyperparams: Training hyper-parameters that define network
            architecture and field types.
        key: PRNG key used to initialise all sub-networks.

    Returns:
        A freshly initialised :class:`ReducedHybridSDE` instance.
    """
    from dataclasses import replace

    model_key, _, _ = jr.split(key, 3)
    f_key, h_key, g_key, model_key = jr.split(model_key, 4)
    mc = hyperparams.model

    f_config = replace(mc.f_config, key=f_key)
    h_config = replace(mc.h_config, key=h_key)
    g_config = replace(mc.g_config, key=g_key)

    sde = ReducedHybridSDE(
        hyperparams.data.data_size,
        mc.latent_size,
        mc.ctx_size,
        mc.hidden_size,
        f_config,
        h_config,
        g_config,
        key=model_key,
        dt=hyperparams.data.dt,
        phys_config=hyperparams.model.physics_config,
        indirect_eta=hyperparams.model.indirect_eta,
        kl_eps=mc.kl_eps,
    )

    return sde


def make_test_loss(
    model_static: eqx.Module,
    hyperparams: HyperParameters,
):
    """Build a JIT-compiled test loss function (forward pass only).

    Returns the same metrics as training but without computing gradients
    or updating parameters.

    Args:
        model_static: Non-trainable (static) partition of the model.
        hyperparams: Training hyper-parameters.

    Returns:
        A JIT-compiled function with signature::

            test_fn(sde_params, xs_batch, ts, *, key, kl_weight, log_sigma)
            -> dict[str, scalar]
    """

    @eqx.filter_jit
    def _test(
        sde_params,
        xs_batch: jax.Array,
        ts: jax.Array,
        *,
        key: jax.Array,
        kl_weight: jax.Array,
        log_sigma: jax.Array,
        cond: jax.Array | None = None,
    ):
        del cond  # RO model has no wave conditioning
        sde = cast(ReducedHybridSDE, eqx.combine(sde_params, model_static))
        batch_size = xs_batch.shape[0]
        keys = jr.split(key, batch_size)

        losses, (kl_initials, kl_paths, log_pxs) = jax.vmap(
            lambda x, k: loss_fn(
                sde,
                x,
                ts,
                key=k,
                kl_weight=kl_weight,
                log_sigma=log_sigma,
            )
        )(xs_batch, keys)

        return {
            "loss": jnp.mean(losses),
            "kl_initial": jnp.mean(kl_initials),
            "kl_path": jnp.mean(kl_paths),
            "log_pxs": -jnp.mean(log_pxs),
        }

    return _test


class ROHybridTrainSetup(TrainSetup):
    """TrainSetup for the ReducedHybridSDE model."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sampler", ROHybridSampler())
        kwargs.setdefault("model_type_name", "ReducedHybridSDE")
        super().__init__(*args, **kwargs)

    def make_schedule_fn(self, hyperparams: HyperParameters) -> optax.Schedule:
        return make_schedule_exp(hyperparams)

    def make_train_step_fn(self, optimizer, model_static, dataset, hyperparams):
        return make_train_step(optimizer, model_static, dataset, hyperparams)

    def make_test_loss_fn(self, model_static, hyperparams):
        return make_test_loss(model_static, hyperparams)

    def init_model_fn(self, hyperparams: HyperParameters, key) -> ReducedHybridSDE:
        return init_model(hyperparams, key)


def main(
    data_path: Path,
    hyperparams: HyperParameters,
    device: str = "gpu",
    run_name: str = "reduced_hybrid_sde",
    parent_id: str | None = None,
) -> None:
    """Run the full training pipeline for the ReducedHybridSDE.

    Delegates to :func:`thesis.shared.train.train`, passing model
    initialisation, train-step construction, and sampling callbacks.

    Args:
        data_path: Root directory containing the training data files.
        hyperparams: Full set of training hyper-parameters.
        device: Accelerator to use (``"gpu"`` or ``"cpu"``).
        run_name: MLflow run name.
        parent_id: Optional MLflow run ID for nested experiment organization.
    """
    setup = ROHybridTrainSetup(
        hyperparams,
        str(data_path),
        device=device,
        run_name=run_name,
        parent_id=parent_id,
    )
    train(setup)


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train ReducedHybridSDE")
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Consolidated parquet file",
        default=Path(
            "/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/directional_ou"
        ),
    )
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--run_name", default="reduced_hybrid_sde")
    parser.add_argument(
        "--hyperparams", default=None, help="Path to JSON file with hyper-parameters"
    )
    parser.add_argument(
        "--run_id", default=None, help="Parent run ID for nested experiment"
    )
    args = parser.parse_args()

    hyperparams = (
        hyperparams_from_json(args.hyperparams)
        if args.hyperparams is not None
        else None
    )

    if args.run_id:
        parent_run = mlflow.get_run(args.run_id)
        experiment_id = parent_run.info.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_id="2062398295866591")
    if hyperparams is None:
        raise SystemExit("--hyperparams is required.")
    main(
        args.data_path,
        device=args.device,
        run_name=args.run_name,
        hyperparams=hyperparams,
        parent_id=args.run_id,
    )


if __name__ == "__main__":
    mlflow.set_experiment("physics_sde_local")
    data_path = Path(
        r"/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/directional_ou"
    )
    vessel = SupplyVessel()

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
            physics_config=PhysicsConfig(
                M=jnp.array(vessel.M3),
                D=jnp.array(vessel.D3),
                n_max=jnp.array(vessel.n_max),
                thrust_matrix=jnp.array(vessel.B),
                w0=jnp.array(vessel.wn),
                zeta=jnp.array(vessel.zeta),
                T_n=vessel.T_n,
            ),
            indirect_eta=True,
        ),
        training=TrainingConfig(
            lr_init=1e-3,
            lr_end=1e-5,
            lr_warmup_fraction=0.01,
            num_steps=5000,
            batch_size=128,
            log_every=1,
            sample_every=1,
            checkpoint_every=100,
            kl_annealing=[
                AnnealConfig(
                    start=0,
                    end=1500,
                    warmup=0.3,
                    initial_weight=0.01,
                    final_weight=1.0,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
                AnnealConfig(
                    start=1500,
                    end=3000,
                    warmup=0.1,
                    initial_weight=1.0,
                    final_weight=3.0,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
                AnnealConfig(
                    start=3000,
                    end=5000,
                    warmup=0.1,
                    initial_weight=3.0,
                    final_weight=5.0,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
            ],
            noise_annealing=[
                AnnealConfig(
                    start=0,
                    end=1666,
                    warmup=0.0,
                    initial_weight=0,
                    final_weight=-1,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
                AnnealConfig(
                    start=1666,
                    end=3332,
                    warmup=0.0,
                    initial_weight=-1.0,
                    final_weight=-2.0,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
                AnnealConfig(
                    start=3332,
                    end=5000,
                    warmup=0.0,
                    initial_weight=-2.0,
                    final_weight=-3.0,
                    annealing_strategy=AnnealStrategy.COSINE,
                ),
            ],
            curriculum=(
                (0, 50),  # short windows to learn local dynamics
                (1500, 200),  # medium
                (3000, 400),  # approaching T_z0
            ),
            max_grad_norm=5000.0,
        ),
        data=DataConfig(
            features=[
                Features.POS_ETA_X,
                Features.POS_ETA_Y,
                Features.POS_ETA_MZ,
                Features.POS_NU_X,
                Features.POS_NU_Y,
                Features.POS_NU_MZ,
                Features.RPM_BOW_FORE,
                Features.RPM_BOW_AFT,
                Features.RPM_STERN_FORE,
                Features.RPM_STERN_AFT,
                Features.RPM_FIXED_PS,
                Features.RPM_FIXED_SB,
            ],
            dt=0.5,
            sample_length=400,
            n_files=50,
            truncate_seconds=600,
        ),
    )

    main(data_path, hyperparams=hyperparams, device="cpu", run_name="short_25s")

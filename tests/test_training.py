"""Smoke tests for the full-order hybrid SDE training pipeline."""
from mlflow.entities import Run
from pandas import DataFrame

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow
import pytest

from thesis.full_hybrid_sde.model import FullHybridSDE
from thesis.full_hybrid_sde.nominal_dynamics import default_mss_physics_config
from thesis.full_hybrid_sde.train import FOHybridTrainSetup
from thesis.shared.consolidated_dataset import ConsolidatedDataset
from thesis.shared.data_structures import (
    AnnealConfig,
    DataConfig,
    FieldConfig,
    FieldType,
    FO_3DOF_FEATURES,
    HyperParameters,
    ModelConfig,
    TrainingConfig,
    hyperparams_from_json,
    hyperparams_to_json,
)
from thesis.shared.sampler import NullSampler
from thesis.shared.train import train

SAMPLE_LENGTH = 20  # steps at dt = 0.5 s → 10 s windows
LATENT = 4


def tiny_hyperparams(num_steps: int = 3) -> HyperParameters:
    key = jr.key(0)
    f_key, h_key, g_key, t_key = jr.split(key, 4)
    return HyperParameters(
        model=ModelConfig(
            f_config=FieldConfig(
                field_type=FieldType.CONTEXT_STATE,
                latent_size=LATENT,
                hidden_layer_width=8,
                depth=1,
                context_size=4,
                key=f_key,
            ),
            h_config=FieldConfig(
                field_type=FieldType.STATE,
                latent_size=LATENT,
                hidden_layer_width=8,
                depth=1,
                mean_reversion=True,
                key=h_key,
            ),
            g_config=FieldConfig(
                field_type=FieldType.STATE,
                latent_size=LATENT,
                hidden_layer_width=8,
                depth=1,
                control_size=LATENT,
                final_activation="sigmoid",
                diagonal=True,
                key=g_key,
            ),
            fo_physics_config=default_mss_physics_config(),
            indirect_eta=True,
        ),
        training=TrainingConfig(
            lr_init=1e-3,
            lr_end=1e-4,
            num_steps=num_steps,
            batch_size=2,
            log_every=1,
            sample_every=1000,
            checkpoint_every=1000,
            max_grad_norm=1000.0,
            kl_annealing=AnnealConfig(
                start=0, end=100, initial_weight=1.0, final_weight=1.0
            ),
            noise_annealing=AnnealConfig(
                start=0, end=100, initial_weight=-2.0, final_weight=-2.0
            ),
            curriculum=((0, 10),),  # 10 s → 20 steps at dt 0.5
            key=t_key,
        ),
        data=DataConfig(
            features=FO_3DOF_FEATURES,
            dt=0.5,
            sample_length=SAMPLE_LENGTH,
            n_files=None,
            truncate_seconds=0.0,
            group_scaling=True,
            test_fraction=0.0,
        ),
    )


@pytest.fixture(scope="module")
def dataset(consolidated_path) -> ConsolidatedDataset:
    hp = tiny_hyperparams()
    return ConsolidatedDataset(
        consolidated_path,
        columns=hp.data.features,
        wave_keys=hp.data.wave_keys,
        sample_length=SAMPLE_LENGTH,
        resample_dt=hp.data.dt,
        standardise=True,
        group_scaling=hp.data.group_scaling,
    )


def make_setup(hp, consolidated_path, dataset, **kwargs) -> FOHybridTrainSetup:
    return FOHybridTrainSetup(
        hp,
        str(consolidated_path),
        model_type_name="FullHybridSDE",
        device="cpu",
        print_batch=False,
        dataset=dataset,
        sampler=NullSampler(),
        **kwargs,
    )


def test_init_model(consolidated_path, dataset):
    hp = tiny_hyperparams()
    setup = make_setup(hp, consolidated_path, dataset)
    model = setup.init_model_fn(hp, jr.key(0))
    assert isinstance(model, FullHybridSDE)
    assert model.data_size == len(FO_3DOF_FEATURES)
    assert model.latent_size == LATENT


def test_single_train_step_updates_params(consolidated_path, dataset):
    hp = tiny_hyperparams()
    setup = make_setup(hp, consolidated_path, dataset)
    model = setup.init_model_fn(hp, jr.key(0))

    schedule = setup.make_schedule_fn(hp)
    optimizer = setup.make_optimizer_fn(
        schedule, hp.training.max_grad_norm, hp.training.weight_decay
    )
    params, static = eqx.partition(model, model.trainable_filter())
    opt_state = optimizer.init(params)
    train_step = setup.make_train_step_fn(optimizer, static, dataset, hp)

    ts, xs, wave_cond, _ = dataset.sample_random(jr.key(1), hp.training.batch_size)
    # Two steps: the LR warmup schedule is zero at step 0, so parameters
    # only move from the second step onwards.
    new_params, new_opt_state, metrics = params, opt_state, {}
    for step in range(2):
        new_params, new_opt_state, metrics = train_step(
            new_params,
            new_opt_state,
            xs,
            ts,
            key=jr.fold_in(jr.key(2), step),
            kl_weight=jnp.array(1.0),
            log_sigma=jnp.array(-2.0),
            cond=wave_cond,
        )

    for name in ("loss", "kl_initial", "kl_path", "log_pxs", "grad_norm"):
        assert name in metrics
        assert bool(jnp.isfinite(metrics[name])), f"{name} is not finite"

    # At least one trainable parameter must have moved
    diffs = jax.tree_util.tree_map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))) if a is not None else 0.0,
        params,
        new_params,
        is_leaf=lambda x: x is None,
    )
    assert max(jax.tree_util.tree_leaves(diffs), default=0.0) > 0.0


def test_test_loss_fn_finite(consolidated_path, dataset):
    hp = tiny_hyperparams()
    setup = make_setup(hp, consolidated_path, dataset)
    model = setup.init_model_fn(hp, jr.key(0))
    params, static = eqx.partition(model, model.trainable_filter())
    test_loss_fn = setup.make_test_loss_fn(static, hp)

    ts, xs, wave_cond, _ = dataset.sample_random(jr.key(3), 2)
    metrics = test_loss_fn(
        params,
        xs,
        ts,
        key=jr.key(4),
        kl_weight=jnp.array(1.0),
        log_sigma=jnp.array(-2.0),
        cond=wave_cond,
    )
    assert bool(jnp.isfinite(metrics["loss"]))


def test_sample_prior_generates_trajectory(consolidated_path, dataset):
    hp = tiny_hyperparams()
    setup = make_setup(hp, consolidated_path, dataset)
    model = setup.init_model_fn(hp, jr.key(0))

    ts = jnp.arange(SAMPLE_LENGTH, dtype=jnp.float32) * 0.5
    wave_cond = jnp.array([0.5, 0.5, 0.5])
    xs = model.sample_prior(ts, key=jr.key(5), wave_cond=wave_cond)
    assert xs.shape == (SAMPLE_LENGTH, len(FO_3DOF_FEATURES))
    assert bool(jnp.all(jnp.isfinite(xs)))


def test_hyperparams_json_roundtrip(tmp_path):
    hp = tiny_hyperparams()
    path = tmp_path / "hyperparams.json"
    hyperparams_to_json(hp, path)
    hp2 = hyperparams_from_json(path)
    assert hp2.training.num_steps == hp.training.num_steps
    # JSON turns tuples into lists; compare values only
    assert hp.training.curriculum is not None
    assert hp2.training.curriculum is not None
    assert [list(c) for c in hp2.training.curriculum] == [
        list(c) for c in hp.training.curriculum
    ]
    assert hp2.data.features == hp.data.features
    assert hp2.model.f_config.latent_size == hp.model.f_config.latent_size


def test_train_end_to_end_smoke(consolidated_path, dataset, tmp_path):
    """Run the real training loop for a few steps against a local MLflow store."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("pipeline_smoke_test")

    hp = tiny_hyperparams(num_steps=3)
    setup = make_setup(
        hp, consolidated_path, dataset, run_name="smoke_test"
    )
    train(setup)

    runs = mlflow.search_runs(experiment_names=["pipeline_smoke_test"], output_format="pandas")
    assert len(runs) == 1
    if isinstance(runs, DataFrame) and not runs.empty:
        run = runs.iloc[0]
    else:
        raise RuntimeError("No runs found for the experiment.")
    assert run["status"] == "FINISHED"
    assert jnp.isfinite(run["metrics.loss"])
    assert run["metrics.wall_time"] > 0.0

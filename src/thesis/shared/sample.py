import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import pandas as pd

from pathlib import Path

from thesis.full_hybrid_sde.train import init_model as init_full_hybrid
from thesis.reduced_hybrid_sde.train import init_model as init_reduced_hybrid
from thesis.shared.data_handling import find_parquet_files
from thesis.shared.data_structures import hyperparams_from_json
from thesis.shared.model import SDE


def sample_trained(
    ts: jax.Array,
    run_id: str,
    experiment_id: str,
    artifact_path: str,
    store_path: Path = Path("tmp/samples"),
    n_samples: int = 50,
    data_path: Path | None = None,
    *,
    key,
    **kwargs,
) -> tuple[SDE, Path] | None:
    """Sample from the trained SDE model and return path to saved data."""
    import mlflow

    os.environ["DATABRICKS_CONFIG_PROFILE"] = "dev"
    mlflow.set_tracking_uri("databricks://dev")
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    hyperparams_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="hyperparams.json"
    )

    hyperparams = hyperparams_from_json(hyperparams_path)
    model_type = mlflow.get_run(run_id).data.params["model_type"]
    store_path = store_path / model_type

    if model_type == "FullHybridSDE":
        model: SDE = init_full_hybrid(hyperparams, key)
    elif model_type == "ReducedHybridSDE":
        model = init_reduced_hybrid(hyperparams, key)
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    try:
        model = eqx.tree_deserialise_leaves(model_path, model)
    except eqx._serialisation.TreePathError:
        # Old checkpoint without data_mean/data_std — load into a
        # version of the tree that has those leaves filtered out.
        filter_spec = jax.tree.map(lambda _: True, model)
        filter_spec = eqx.tree_at(
            lambda m: (m.data_mean, m.data_std), filter_spec, (False, False)
        )
        partial = eqx.filter(model, filter_spec)
        partial = eqx.tree_deserialise_leaves(model_path, partial)
        model = eqx.combine(
            partial, eqx.filter(model, jax.tree.map(lambda x: not x, filter_spec))
        )

    # Backwards compatibility: old checkpoints lack data_mean/data_std.
    # Detect identity scaling and reconstruct stats from training data.
    _has_scaling = not (jnp.all(model.data_mean == 0) and jnp.all(model.data_std == 1))
    if not _has_scaling:
        if data_path is None:
            raise ValueError(
                "This checkpoint has no stored scaling stats. "
                "Pass data_path to reconstruct them from the training data."
            )
        from thesis.shared.jax_dataset import JAXParquetDataset

        train_files = find_parquet_files(
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
            train_files,
            columns=hyperparams.data.features,
            sample_length=hyperparams.data.sample_length,
            resample_every=int(hyperparams.data.dt // 0.05),
            standardise=True,
            truncate_seconds=hyperparams.data.truncate_seconds,
            group_scaling=hyperparams.data.group_scaling,
        )
        model = eqx.tree_at(
            lambda m: (m.data_mean, m.data_std),
            model,
            (
                jnp.array(dataset.standardise["mean"], dtype=jnp.float32),
                jnp.array(dataset.standardise["std"], dtype=jnp.float32),
            ),
        )

    sample_dir = (
        store_path
        / f"{artifact_path.split('/')[-1].split('.')[0]}_{run_id}_{experiment_id}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(sample_dir.glob("*.parquet"))
    if len(existing) >= n_samples:
        print(f"Sample data already exists at {sample_dir}. Skipping sampling.")
        return model, sample_dir

    keys = jr.split(key, n_samples)
    # sample_prior unscales to physical units by default
    data = jax.vmap(lambda key: model.sample_prior(ts, key=key, **kwargs))(keys)
    # data shape: (n_samples, T, F)

    feature_names = [f.value for f in hyperparams.data.features]
    time_np = np.asarray(ts)
    data_np = np.asarray(data)

    for i in range(n_samples):
        trajectory = data_np[i]  # (T, F)
        df = pd.DataFrame(trajectory, columns=pd.Index(feature_names))
        df.insert(0, "time", time_np)
        df.to_parquet(sample_dir / f"sample_{i:04d}.parquet", index=False)

    return model, sample_dir

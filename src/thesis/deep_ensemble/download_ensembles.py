from pathlib import Path
import os
import mlflow

def _strip_name(name: str) -> str:
    """
    Strip the run name to a more manageable format.
    """
    return "_".join(name.split("_")[:-3])

def _empty_dir(folder: Path) -> None:
    """
    Empty the contents of a directory.
    """
    for f in folder.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)
        elif f.is_dir():
            _empty_dir(f)
            f.rmdir()

def download_ensembles(
    experiment_id: str, 
    parent_id: str, 
    folder: Path = Path(__file__).parent / "ensembles",
    checkpoints: dict[str, str | int | None] = {},
    exclude: list[str] = []) -> None:
    """
    Download the ensemble of models from the remote repository.
    """
    os.environ["DATABRICKS_CONFIG_PROFILE"] = "dev"
    mlflow.set_tracking_uri("databricks://dev")
    mlflow.set_experiment(experiment_id=experiment_id)
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")

    folder.mkdir(parents=True, exist_ok=True)

    _empty_dir(folder)


    for run in runs:
        run_id = run.info.run_id
        if run_id in exclude:
            print(f"Skipping excluded run {run.info.run_name} (ID: {run_id})")
            continue
        run_name = run.info.run_name
        hp = Path(mlflow.artifacts.download_artifacts(artifact_path="hyperparams.json", run_id=run_id))
        hp.rename(folder / f"{_strip_name(run_name)}_hyperparams.json")

        cp = checkpoints.get(run_id, None)
        if cp is None:
            artifact_path = "checkpoint/final.eqx"
        else:
            artifact_path = f"checkpoint/checkpoint_{cp}.eqx"
        cp = Path(mlflow.artifacts.download_artifacts(artifact_path=artifact_path, run_id=run_id))
        cp.rename(folder / f"{_strip_name(run_name)}.eqx")
        print(f"Downloaded artifacts for run {run_name} (ID: {run_id}) to {folder}")


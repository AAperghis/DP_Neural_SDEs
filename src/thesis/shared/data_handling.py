from dataclasses import asdict, dataclass, field
import datetime
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd

import json
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = [
    "time",
    "pos_eta_x",
    "pos_eta_y",
    "pos_eta_mz",
    "pos_nu_x",
    "pos_nu_y",
    "pos_nu_mz",
    "tau_control_x",
    "tau_control_y",
    "tau_control_mz",
    "tau_actual_x",
    "tau_actual_y",
    "tau_actual_mz",
    "tau_ext_x",
    "tau_ext_y",
    "tau_ext_mz",
    "gain_P_x",
    "gain_P_y",
    "gain_P_mz",
    "gain_I_x",
    "gain_I_y",
    "gain_I_mz",
    "gain_D_x",
    "gain_D_y",
    "gain_D_mz",
    "rpm_bow_fore",
    "rpm_bow_aft",
    "rpm_stern_fore",
    "rpm_stern_aft",
    "rpm_fixed_ps",
    "rpm_fixed_sb",
]


@dataclass(frozen=True)
class ParquetMetadata:
    model: str
    version: str
    seed: int
    timestep: float
    end_time: float
    n_steps: int
    mean_force: list[float]
    var_force: list[float]
    inital_pos: tuple[float, float, float]
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    )


def make_df(N: int) -> np.ndarray:
    d = len(COLUMNS)
    return np.empty((N + 1, d))


def finalize_df(data: np.ndarray) -> pd.DataFrame:
    """Convert the raw NumPy buffer into a DataFrame (call once after sim loop)."""
    return pd.DataFrame(data, columns=np.array(COLUMNS))


def update_df(
    data: np.ndarray,
    idx: int,
    t: float,
    eta: np.ndarray,
    nu: np.ndarray,
    tau_control: np.ndarray,
    tau_actual: np.ndarray,
    f_ext: np.ndarray,
    pid_gains: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    rpms: np.ndarray,
) -> None:
    """Function to update data storage array"""
    row = data[idx]
    row[0] = t

    # Save eta and nu, only 3DOF
    row[1] = eta[0]
    row[2] = eta[1]
    row[3] = eta[5]
    row[4] = nu[0]
    row[5] = nu[1]
    row[6] = nu[5]

    # Forces
    row[7:10] = tau_control
    row[10:13] = tau_actual
    row[13:16] = f_ext

    # PID gains
    row[16:19] = pid_gains[0]
    row[19:22] = pid_gains[1]
    row[22:25] = pid_gains[2]

    # Thrusters
    row[25:] = rpms


def save_df_to_parquet(
    df: pd.DataFrame,
    metadata: ParquetMetadata,
    base_name: str = "dp_sim",
    path: Path | None = None,
) -> None:
    """
    Function to save a dataframe as a Parquet file. Encodes metadata into the file for efficient indexing.

    File is saved with format: "{base_name}_{metadata.end_time}_{metadata.timestep}_{metadata.seed}.parquet"

    :param df: Dataframe to be saved
    :type df: pd.DataFrame
    :param metadata: Metadata dataclass containing information about the simulation to be saved
    :type metadata: ParquetMetadata
    :param base_name: Base name used to save the file
    :type base_name: str
    :param path: Folder to save the Parquet file in
    :type path: Path | None
    """
    table = pa.Table.from_pandas(df)
    meta = dict(table.schema.metadata or {})
    meta_key = b"run_params"
    meta[meta_key] = json.dumps(asdict(metadata)).encode("utf-8")

    table = table.replace_schema_metadata(meta)

    filename = (
        f"{base_name}_{metadata.end_time}_{metadata.timestep}_{metadata.seed}.parquet"
    )
    path.mkdir(parents=True, exist_ok=True) if path else None
    if path:
        path = path / filename
    else:
        path = Path(filename)

    pq.write_table(table, path)


def extract_metadatas(
    folder: Path, meta_key: str = "run_params"
) -> list[dict[str, Any]]:
    """Extract metadata dictionaries from a list of parquet files."""
    folder = Path(folder)
    if folder.is_file() and folder.suffix == ".parquet":
        return list(read_consolidated_metadata(folder, meta_key).values())
    return [read_metadata(path, meta_key) for path in sorted(folder.glob("*.parquet"))]


def read_metadata(path: Path, meta_key: str = "run_params") -> dict[str, Any]:
    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    key_b = meta_key.encode("utf-8")
    if key_b in meta:
        return json.loads(meta[key_b].decode("utf-8"))
    return {}


def find_parquet_files(
    folder: str | Path,
    filter_fn: Callable[[dict[str, Any]], bool] = lambda _: True,
    meta_key: str = "run_params",
) -> list[Path]:
    """Search for parquet files whose embedded metadata satisfies *filter_fn*.

    Works for both a **directory of individual parquet files** and a
    **single consolidated parquet file**.  When *folder* points to a
    consolidated file, each run's metadata (stored under the
    ``run_metadata`` schema key) is checked and a list containing the
    single file path is returned when at least one run matches.
    """
    folder = Path(folder)

    # Single consolidated file
    if folder.is_file() and folder.suffix == ".parquet":
        run_meta = read_consolidated_metadata(folder)
        if any(filter_fn(m) for m in run_meta.values()):
            return [folder]
        return []

    matches = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            params = read_metadata(path, meta_key)
            if filter_fn(params):
                matches.append(path)
        except Exception:
            pass
    return matches


# ---------------------------------------------------------------------------
# Consolidated parquet helpers
# ---------------------------------------------------------------------------


def read_consolidated_metadata(
    path: Path,
    meta_key: str = "run_metadata",
) -> dict[str, dict[str, Any]]:
    """Read per-run metadata from a consolidated parquet file.

    Returns a dict mapping ``run_id`` (str) to metadata dict.
    """
    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(meta_key.encode("utf-8"), b"{}")
    return json.loads(raw.decode("utf-8"))


def filter_consolidated_runs(
    path: Path,
    filter_fn: Callable[[dict[str, Any]], bool] = lambda _: True,
    meta_key: str = "run_metadata",
) -> list[int]:
    """Return run IDs from a consolidated parquet file that match *filter_fn*."""
    run_meta = read_consolidated_metadata(path, meta_key)
    return [int(rid) for rid, m in run_meta.items() if filter_fn(m)]

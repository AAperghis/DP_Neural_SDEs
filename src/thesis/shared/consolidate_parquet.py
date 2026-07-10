"""Consolidate individual MSS parquet files into a single file.

Merges per-run parquet files produced by ``mss_model.gen_training_data``
into one consolidated file with a ``run_id`` column.  Per-run metadata
(Hs, Tp, beta_wave, seed, …) is stored as JSON in the Parquet schema
metadata under the key ``run_metadata``.

Usage::

    python -m thesis.shared.consolidate_parquet \\
        --input-dir /path/to/individual_parquets \\
        --output /path/to/consolidated.parquet

The consolidated file can then be loaded efficiently by
:class:`~thesis.shared.consolidated_dataset.ConsolidatedDataset`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def consolidate(
    input_dir: Path,
    output_path: Path,
    meta_key: str = "sim_config",
    columns: list[str] | None = None,
) -> Path:
    """Merge individual parquet files into a single consolidated file.

    Parameters
    ----------
    input_dir : Path
        Directory containing individual per-run parquet files.
    output_path : Path
        Destination path for the consolidated file.
    meta_key : str
        Schema metadata key containing the per-run JSON config.
    columns : list[str] | None
        If given, only keep these columns (plus ``t``).  By default all
        columns are preserved.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    run_metadata: dict[str, dict] = {}
    all_tables: list[pa.Table] = []

    for run_id, fpath in enumerate(tqdm(files, desc="Reading parquets")):
        table = pq.read_table(fpath, columns=columns)

        # Extract per-file metadata
        schema_meta = table.schema.metadata or {}
        key_b = meta_key.encode("utf-8")
        meta = (
            json.loads(schema_meta[key_b].decode("utf-8"))
            if key_b in schema_meta
            else {}
        )
        run_metadata[str(run_id)] = meta

        # Add run_id column
        run_id_col = pa.array(np.full(len(table), run_id, dtype=np.int32))
        table = table.append_column("run_id", run_id_col)
        all_tables.append(table)

    combined = pa.concat_tables(all_tables, promote_options="default")

    # Store per-run metadata at file level
    file_meta = dict(combined.schema.metadata or {})
    file_meta[b"run_metadata"] = json.dumps(run_metadata).encode("utf-8")
    file_meta[b"source_meta_key"] = meta_key.encode("utf-8")
    combined = combined.replace_schema_metadata(file_meta)

    pq.write_table(combined, output_path, compression="zstd")
    n_rows = len(combined)
    size_mb = output_path.stat().st_size / 1e6
    print(
        f"Consolidated {len(files)} files → {output_path} "
        f"({n_rows:,} rows, {size_mb:.1f} MB)"
    )
    return output_path


def main(input_dir: Path | None = None, output: Path | None = None):
    parser = argparse.ArgumentParser(description="Consolidate parquet files")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meta-key", default="sim_config")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to keep (default: all)",
    )
    args = parser.parse_args()
    if input_dir is None:
        input_dir = args.input_dir
    if output is None:
        output = args.output
    consolidate(input_dir, output, args.meta_key, args.columns)


if __name__ == "__main__":
    main(
        Path(
            "/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/fo_data_full_state_v4"
        ),
        Path(
            "/mnt/c/Users/AAg/OneDrive - Allseas Engineering BV/Documents/Thesis/data/consolidated/fo_data_v4.parquet"
        ),
    )

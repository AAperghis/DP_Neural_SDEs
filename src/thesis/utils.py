from __future__ import annotations
from contextlib import contextmanager
import os
import time
import traceback

import queue
import threading
import platform
import sys
from typing import Any

import jax
import jax.typing as jtp
import mlflow


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def databricks_test_func() -> None:
    print("Test v2")


class AsyncLogger:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background thread writes logs without blocking training"""
        while self.running:
            try:
                msg = self.queue.get(timeout=0.5)
                if msg is None:  # Shutdown signal
                    break
                print(msg)
                self.queue.task_done()
            except queue.Empty:
                continue

    def log(self, msg: str) -> None:
        """Non-blocking log - returns immediately"""
        self.queue.put_nowait(msg)

    def flush(self) -> None:
        """Wait for all logs to finish (call at epoch end)"""
        self.queue.join()

    def shutdown(self) -> None:
        self.running = False
        self.queue.put(None)


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _collect_base_env() -> dict[str, Any]:
    return {
        "os.system": platform.system(),
        "os.release": platform.release(),
        "os.version": platform.version(),
        "python.version": sys.version.split()[0],
    }


def _collect_jax_info() -> dict[str, Any]:
    info = {}
    jax = _safe_import("jax")
    if jax is None:
        return info
    try:
        info["jax.version"] = getattr(jax, "__version__", "unknown")
        # List devices and backend in a portable way.
        devs = jax.devices()  # list of Device objects (cpu/gpu/tpu)  [3](https://docs.jax.dev/en/latest/_autosummary/jax.devices.html)
        info["jax.devices"] = ", ".join([f"{d.platform}:{d.id}" for d in devs])
        info["jax.backend"] = devs[0].platform if devs else "none"
    except Exception as e:
        info["jax.error"] = repr(e)
    return info


def _collect_torch_info() -> dict[str, Any]:
    info = {}
    torch = _safe_import("torch")
    if torch is None:
        return info
    try:
        info["torch.version"] = getattr(torch, "__version__", "unknown")
        if (
            hasattr(torch, "cuda") and torch.cuda.is_available()
        ):  # CUDA presence  [4](https://docs.pytorch.org/docs/stable/cuda.html)
            n = torch.cuda.device_count()
            names = []
            for i in range(n):
                try:
                    names.append(torch.cuda.get_device_name(i))
                except Exception:
                    names.append(f"cuda:{i}")
            info["torch.cuda.available"] = True
            info["torch.cuda.count"] = n
            info["torch.cuda.names"] = ", ".join(names)
        else:
            info["torch.cuda.available"] = False
    except Exception as e:
        info["torch.error"] = repr(e)
    return info


def _collect_databricks_env() -> dict[str, Any]:
    """
    Collect Databricks-specific signals. Works only if running on Databricks.
    Falls back silently elsewhere.
    """
    d = {}
    # Common Databricks-provided env vars on clusters / jobs (driver & workers)  [5](https://docs.databricks.com/aws/en/init-scripts/environment-variables)[6](https://learn.microsoft.com/en-us/azure/databricks/init-scripts/environment-variables)
    for key in [
        "DB_CLUSTER_ID",
        "DB_CLUSTER_NAME",
        "DB_IS_DRIVER",
        "DB_INSTANCE_TYPE",
        "DATABRICKS_RUNTIME_VERSION",  # frequently present on many runtimes
    ]:
        val = os.environ.get(key)
        if val is not None:
            d[f"db.{key}"] = val

    # Try to pull richer info from Spark conf if Spark is available.
    # Many Databricks tags are exposed via spark.conf.* clusterUsageTags.*  (runtime, node type, region, etc.)  [7](https://jdhao.github.io/2023/05/13/databricks-spark-get-set-conf/)
    try:
        from pyspark.sql import SparkSession  # pyright: ignore[reportMissingImports]  # ty:ignore[unresolved-import]

        spark = SparkSession.builder.getOrCreate()
        # A few useful keys (presence may vary by runtime/policy):
        keys = [
            "spark.databricks.clusterUsageTags.sparkVersion",
            "spark.databricks.clusterUsageTags.node_type_id",
            "spark.databricks.clusterUsageTags.driver_node_type_id",
            "spark.databricks.clusterUsageTags.clusterName",
            "spark.databricks.clusterUsageTags.clusterId",
            "spark.databricks.clusterUsageTags.cloudProvider",
            "spark.databricks.clusterUsageTags.region",
        ]
        for k in keys:
            try:
                v = spark.conf.get(k)
                if v is not None:
                    d[f"db.{k}"] = v
            except Exception:
                pass
    except Exception:
        pass

    return d


def collect_compute_facts() -> dict[str, Any]:
    facts = {}
    facts.update(_collect_base_env())
    facts.update(_collect_jax_info())
    facts.update(_collect_torch_info())
    facts.update(_collect_databricks_env())
    return facts


def log_compute_to_mlflow(
    extra_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Logs a curated set of compute facts to the active MLflow run as tags/params.
    Returns the dict for your own use as well.

    Strategy:
      - Put "where/how I ran" into TAGS (searchable, not part of the model signature).
      - Put a couple of tiny, stable things as PARAMS (e.g., device backend).
    """

    facts = collect_compute_facts()

    # Separate "small params" from "tags". Use params sparingly to avoid quotas.  [2](https://docs.databricks.com/aws/en/mlflow/tracking)
    params = {}
    if "jax.backend" in facts:
        params["compute.jax.backend"] = facts["jax.backend"]
    if "torch.cuda.available" in facts:
        params["compute.torch.cuda"] = str(facts["torch.cuda.available"])

    if params:
        mlflow.log_params(
            params
        )  # log tiny, stable params first  [1](https://mlflow.org/docs/latest/ml/tracking/)

    # Everything else as tags (strings only).
    tags = {}
    for k, v in facts.items():
        try:
            tags[f"compute.{k}"] = str(v)
        except Exception:
            continue

    if extra_tags:
        tags.update({str(k): str(v) for k, v in extra_tags.items()})

    # Set tags for this run. (Tags are intended for run metadata like environment.)  [1](https://mlflow.org/docs/latest/ml/tracking/)
    mlflow.set_tags(tags)
    return facts


@contextmanager
def crash_logger():
    """Context manager that captures training state on crash and writes a diagnostic report.

    Usage::

        with crash_logger("tmp/crash_logs") as state:
            state["epoch"] = 0
            state["global_step"] = 0
            for epoch in range(n_epochs):
                state["epoch"] = epoch
                ...

    On unhandled exception the report is written to ``<log_dir>/crash_<timestamp>.log``.
    ``KeyboardInterrupt`` is logged as a clean interruption and re-raised.
    """

    state: dict[str, Any] = {}

    # noinspection PyBroadException
    try:
        yield state
    except BaseException as exc:
        is_interrupt = isinstance(exc, KeyboardInterrupt)

        lines: list[str] = []
        lines.append(f"{'INTERRUPTED' if is_interrupt else 'CRASH REPORT'}")
        lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Python: {sys.version}")
        lines.append("")

        # Exception info
        lines.append("== Exception ==")
        lines.append(f"Type : {type(exc).__qualname__}")
        lines.append(f"Value: {exc}")
        if not is_interrupt:
            lines.append("")
            lines.append("== Traceback ==")
            lines.append(traceback.format_exc())

        # Training state snapshot
        if state:
            lines.append("")
            lines.append("== Training State ==")
            for k, v in state.items():
                try:
                    if isinstance(v, jtp.ArrayLike):
                        v_str = (
                            f"{float(v):.6g}"
                            if v.ndim == 0
                            else f"shape={v.shape} dtype={v.dtype}"
                        )
                    else:
                        v_str = repr(v)
                    # Truncate very long repr strings
                    if len(v_str) > 500:
                        v_str = v_str[:500] + "..."
                    lines.append(f"  {k}: {v_str}")
                except Exception:
                    lines.append(f"  {k}: <failed to repr>")

        # JAX device / memory info
        lines.append("")
        lines.append("== Device Info ==")
        try:
            for dev in jax.local_devices():
                lines.append(f"  {dev}")
                mem = dev.memory_stats()
                if mem:
                    peak = mem.get("peak_bytes_in_use", 0)
                    current = mem.get("bytes_in_use", 0)
                    limit = mem.get("bytes_limit", 0)
                    lines.append(
                        f"    memory: {current / 1e6:.0f} MB / {limit / 1e6:.0f} MB (peak {peak / 1e6:.0f} MB)"
                    )
        except Exception:
            lines.append("  <failed to query devices>")

        report = "\n".join(lines)

        print(f"\n{'=' * 60}")
        print(report)
        print(f"{'=' * 60}")

        raise

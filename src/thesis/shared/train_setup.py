from abc import ABC, abstractmethod

from pathlib import Path
from typing import Callable

import jax
import optax
from thesis.shared.base_dataset import BaseParquetDataset
from thesis.shared.data_structures import HyperParameters
from thesis.shared.model import SDE
from thesis.shared.sampler import NullSampler, Sampler


class TrainSetup(ABC):
    """Encapsulates the setup for a training run, including model initialisation,
    dataset construction, and train/test step functions.

    This class is used to ensure the signatures of the training setup components are consistent and to
    provide a clear interface for the training loop.  It also allows us to keep the training loop
    clean and focused on the training logic, while the setup details are handled separately.
    """

    _data_path: str
    hyperparams: HyperParameters
    initialised_model: SDE | None = None
    model_type_name: str = "Model"
    device: str = "gpu"
    run_name: str = "train"
    print_batch: bool = True
    dataset: BaseParquetDataset | None = None
    test_dataset: BaseParquetDataset | None = None
    parent_id: str | None = None
    start_step: int = 0

    def __init__(
        self,
        hyperparams: HyperParameters,
        data_path: str,
        *,
        initialised_model: SDE | None = None,
        model_type_name: str = "Model",
        device: str = "gpu",
        run_name: str = "train",
        print_batch: bool = True,
        parent_id: str | None = None,
        start_step: int = 0,
        dataset: BaseParquetDataset | None = None,
        test_dataset: BaseParquetDataset | None = None,
        sampler: Sampler | None = None,
    ):
        self.hyperparams = hyperparams
        self.data_path = data_path
        self.initialised_model = initialised_model
        self.model_type_name = model_type_name
        self.device = device
        self.run_name = run_name
        self.print_batch = print_batch
        self.parent_id = parent_id
        self.start_step = start_step
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.sampler = sampler if sampler is not None else NullSampler()

    @property
    def data_path(self) -> str:
        """Return the path to the dataset for this training setup."""
        return self._data_path

    @data_path.setter
    def data_path(self, path: str | Path):
        """Set the path to the dataset for this training setup."""
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f"Data path must be a string or Path, got {type(path)}")

        if not path.exists():
            raise FileNotFoundError(f"Data path {path} does not exist.")
        if path.is_dir() and list(path.glob("*.parquet")) == []:
            raise FileNotFoundError(
                f"Data path {path} is a directory but contains no parquet files."
            )
        elif not path.is_dir() and path.suffix != ".parquet":
            raise FileNotFoundError(
                f"Data path {path} is not a directory and does not have a .parquet extension."
            )
        self._data_path = str(path)

    @abstractmethod
    def make_train_step_fn(
        self, optimizer, model_static, dataset, hyperparams: HyperParameters
    ) -> Callable:
        """Return a JIT-compiled function that performs a single training step.

        The returned function must have the signature:
            step_fn(params, opt_state, xs, ts, *, key, kl_weight, log_sigma, cond) -> (new_params, new_opt_state, metrics_dict)
        """
        raise NotImplementedError

    def make_test_loss_fn(
        self, model_static, hyperparams: HyperParameters
    ) -> Callable | None:
        """Return a function that computes the test loss, or None to skip test evaluation.

        The returned function must have the signature:
            test_fn(params, xs, ts, *, key, kl_weight, log_sigma, cond) -> metrics_dict
        """
        return None

    @abstractmethod
    def init_model_fn(self, hyperparams: HyperParameters, key: jax.Array) -> SDE:
        """Return an initialised model instance.

        The returned model should be an instance of SDE (or a subclass) and should be initialised with the given hyperparameters and random key.

        This method can be overridden to provide custom model initialisation logic.  If not overridden, it will raise NotImplementedError.
        """
        raise NotImplementedError

    def make_schedule_fn(self, hyperparams: HyperParameters) -> Callable | None:
        """Return a function that computes the schedule for a given training step.

        The returned function should have the signature:
            schedule_fn(step: int) -> dict

        Defaults to one-cycle cosine schedule for learning rate, but can be overridden to provide custom scheduling logic (e.g., for KL weight annealing).
        """
        return optax.schedules.cosine_onecycle_schedule(
            transition_steps=hyperparams.training.num_steps,
            peak_value=hyperparams.training.lr_init,
            pct_start=0.1,
        )

    def make_optimizer_fn(
        self, schedule: Callable, max_grad_norm: float = 10.0, weight_decay: float = 0.0
    ) -> optax.GradientTransformation:
        """Return a function that creates the optimizer given the schedule function and hyperparameters.

        The returned function should have the signature:
            optimizer_fn() -> optax.GradientTransformation

        Defaults to Adam(W) with the given schedule for learning rate, but can be overridden to provide custom optimizer logic (e.g., for different optimizers or additional transformations).
            if weight_decay > 0:
        """
        if weight_decay > 0:
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(schedule, weight_decay=weight_decay),
            )

        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )

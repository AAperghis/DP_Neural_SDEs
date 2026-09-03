from dataclasses import dataclass, field, fields
from enum import Enum
import json
from pathlib import Path
import time

import jax.random as jr
import jax.numpy as jnp
import jax.typing as jtp

class Features(Enum):
    POS_ETA_X = "pos_eta_x"
    POS_ETA_Y = "pos_eta_y"
    POS_ETA_MZ = "pos_eta_mz"
    POS_NU_X = "pos_nu_x"
    POS_NU_Y = "pos_nu_y"
    POS_NU_MZ = "pos_nu_mz"
    RPM_BOW_FORE = "rpm_bow_fore"
    RPM_BOW_AFT = "rpm_bow_aft"
    RPM_STERN_FORE = "rpm_stern_fore"
    RPM_STERN_AFT = "rpm_stern_aft"
    RPM_FIXED_PS = "rpm_fixed_ps"
    RPM_FIXED_SB = "rpm_fixed_sb"
    TAU_EXT_X = "tau_ext_x"
    TAU_EXT_Y = "tau_ext_y"
    TAU_EXT_MZ = "tau_ext_mz"


class FieldType(Enum):
    TIME_STATE = "time_state"
    STATE = "state"
    CONTEXT_TIME_STATE = "context_time_state"
    CONTEXT_STATE = "context_state"
    CONSTANT = "constant"
    FILM_STATE = "film_state"


class AnnealStrategy(Enum):
    LINEAR = "linear"
    COSINE = "cosine"


@dataclass
class FieldConfig:
    field_type: FieldType = FieldType.CONTEXT_TIME_STATE
    latent_size: int = 32
    hidden_layer_width: int = 64
    depth: int = 2
    scale: bool = True
    context_size: int = 32
    control_size: int = 1
    diagonal: bool = True
    key: jtp.ArrayLike = field(default_factory=lambda: jr.key(time.time_ns()))
    mean_reversion: bool = True
    input_size: int | None = (
        None  # If set, overrides latent_size for MLP input dimension
    )
    hidden_activation: str = "lipswish"
    final_activation: str = "tanh"
    film_size: int | None = None  # FiLM conditioning input size (e.g. n_wave_params)


@dataclass
class AnnealConfig:
    start: int = 0
    end: float = jnp.inf
    annealing: float = 1.0
    warmup: float = 0
    initial_weight: float = 0.0
    final_weight: float = 1.0
    annealing_strategy: AnnealStrategy = AnnealStrategy.COSINE


@dataclass
class PhysicsConfig:
    M: jtp.ArrayLike = field(
        default_factory=lambda: jnp.zeros((3, 3), dtype=jnp.float32)
    )  # mass matrix
    D: jtp.ArrayLike = field(
        default_factory=lambda: jnp.zeros((3, 3), dtype=jnp.float32)
    )  # damping matrix
    n_max: jtp.ArrayLike = field(
        default_factory=lambda: jnp.zeros((6,), dtype=jnp.float32)
    )  # max RPM
    thrust_matrix: jtp.ArrayLike = field(
        default_factory=lambda: jnp.zeros((3, 6), dtype=jnp.float32)
    )  # thrust allocation matrix
    w0: jtp.ArrayLike = field(
        default_factory=lambda: jnp.diag(jnp.array([0.1, 0.1, 0.1], dtype=jnp.float32))
    )  # base process noise scale
    zeta: jtp.ArrayLike = field(
        default_factory=lambda: jnp.diag(jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32))
    )  # control noise scaling
    T_n: float = 1.0
    n_rate: jtp.ArrayLike | None = None
    restore_n: bool = False
    disable_controller: bool = False


@dataclass
class FullOrderPhysicsConfig:
    """Physics configuration for FO vessel with azimuth thrusters (3-DOF)."""

    M: jtp.ArrayLike = field(default_factory=lambda: jnp.zeros((3, 3), dtype=jnp.float32))
    D: jtp.ArrayLike = field(default_factory=lambda: jnp.zeros((3, 3), dtype=jnp.float32))
    n_max: jtp.ArrayLike = field(
        default_factory=lambda: jnp.array([140.0, 140.0, 150.0, 150.0])
    )
    K_thr: jtp.ArrayLike = field(default_factory=lambda: jnp.eye(4, dtype=jnp.float32))
    l_x: jtp.ArrayLike = field(
        default_factory=lambda: jnp.array([37.0, 35.0, -42.0, -42.0])
    )
    l_y: jtp.ArrayLike = field(default_factory=lambda: jnp.array([0.0, 0.0, 7.0, -7.0]))
    n_tunnel: int = 2  # First n_tunnel thrusters are tunnel type
    w0: jtp.ArrayLike = field(
        default_factory=lambda: jnp.diag(jnp.array([0.1, 0.1, 0.3], dtype=jnp.float32))
    )
    zeta: jtp.ArrayLike = field(default_factory=lambda: jnp.eye(3, dtype=jnp.float32))
    T_n: float = 1.0
    T_alpha: float = 2.0
    alpha_max: float = 1.047  # pi/3 radians
    disable_controller: bool = True
    # Index mappings into the state vector
    eta_idx: tuple = (0, 1, 2)
    nu_idx: tuple = (3, 4, 5)
    n_idx: tuple = (6, 7, 8, 9)
    alpha_idx: tuple = (10, 11)


@dataclass
class ModelConfig:
    f_config: FieldConfig = field(
        default_factory=lambda: FieldConfig(field_type=FieldType.CONTEXT_STATE)
    )
    h_config: FieldConfig = field(
        default_factory=lambda: FieldConfig(field_type=FieldType.STATE)
    )
    g_config: FieldConfig = field(
        default_factory=lambda: FieldConfig(field_type=FieldType.STATE)
    )
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    fo_physics_config: FullOrderPhysicsConfig = field(
        default_factory=FullOrderPhysicsConfig
    )
    indirect_eta: bool = True  # If True, SDE forcing does not affect eta directly (only through kinematics)
    kl_eps: float = 1e-6  # Floor added to g² in KL denominator; raise to prevent KL blow-up when g is small

    @property
    def latent_size(self) -> int:
        return self.f_config.latent_size

    @property
    def hidden_size(self) -> int:
        return self.f_config.hidden_layer_width

    @property
    def ctx_size(self) -> int:
        return self.f_config.context_size

    @property
    def depth(self) -> int:
        return self.f_config.depth

    @property
    def control_size(self) -> int:
        return self.g_config.control_size

    @property
    def diagonal(self) -> bool:
        return self.g_config.diagonal


@dataclass
class TrainingConfig:
    lr_init: float = 1e-2
    lr_end: float = 1e-5
    lr_warmup_fraction: float = 0.1
    num_steps: int = 10000
    batch_size: int = 256
    log_every: int = 10
    sample_every: int = 100
    checkpoint_every: int = 100
    max_grad_norm: float = 1e7
    kl_annealing: list[AnnealConfig] | AnnealConfig = field(
        default_factory=AnnealConfig
    )
    noise_annealing: list[AnnealConfig] | AnnealConfig = field(
        default_factory=AnnealConfig
    )
    key: jtp.ArrayLike = field(default_factory=lambda: jr.key(time.time_ns()))
    curriculum: tuple[tuple[int, int], ...] | None = (
        None  # List of (train_len, n_prior) pairs for curriculum learning. train_len in seconds.
    )
    weight_decay: float = 0.0  # AdamW decoupled weight decay (0 = plain Adam)


@dataclass
class DataConfig:
    features: "list[Features] | list[FullOrderFeatures]" = field(default_factory=list)
    dt: float = 0.1
    sample_length: int = 2000
    # None (or a value <= 0) uses all available runs; a smaller value draws a
    # random subset (see ConsolidatedDataset / MultiFileDataset).
    n_files: int | None = 25
    truncate_seconds: float = 0.0
    group_scaling: bool = True
    test_fraction: float = 0.2
    hs_max: float | None = None  # If set, exclude runs with Hs >= this value
    wave_keys: list[str] = field(default_factory=lambda: ["Hs", "Tp", "beta_wave"])
    # Subset of wave_keys to encode as (cos, sin) pairs instead of min-max
    # scaling.  Used for circular quantities (e.g. wave direction in radians)
    # so the encoding is wrap-safe and continuous.  Each angular key widens
    # the conditioning vector by one extra column.
    angular_wave_keys: list[str] = field(default_factory=list)

    @property
    def data_size(self) -> int:
        return len(self.features)

    @property
    def n_wave_params(self) -> int:
        """Width of the model-facing wave conditioning vector.

        Angular keys are expanded to ``(cos, sin)`` pairs (two columns);
        all other keys contribute a single min-max scaled column.
        """
        angular = set(self.angular_wave_keys)
        return sum(2 if k in angular else 1 for k in self.wave_keys)


@dataclass
class HyperParameters:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

_ENUM_TYPES = {
    "Features": Features,
    "FieldType": FieldType,
    "AnnealStrategy": AnnealStrategy,
}

_DATACLASS_TYPES = {
    "AnnealConfig": AnnealConfig,
    "PhysicsConfig": PhysicsConfig,
    "FieldConfig": FieldConfig,
    "ModelConfig": ModelConfig,
    "TrainingConfig": TrainingConfig,
    "DataConfig": DataConfig,
    "HyperParameters": HyperParameters,
    "FullOrderPhysicsConfig": FullOrderPhysicsConfig,
}


def _serialize(obj):
    """Recursively convert dataclass / enum / jtp.ArrayLike to JSON-safe types."""
    if isinstance(obj, Enum):
        return {"__enum__": type(obj).__name__, "value": obj.value}
    if isinstance(obj, jtp.ArrayLike):
        if jnp.issubdtype(obj.dtype, jnp.integer) and obj.ndim == 0:
            return None  # Drop scalar integer arrays (legacy PRNG keys)
        try:
            import numpy as np

            return {"__ndarray__": np.asarray(obj).tolist(), "dtype": str(obj.dtype)}
        except TypeError:
            return None  # Drop PRNG keys that can't be converted
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {
            "__dataclass__": type(obj).__name__,
            **{f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)},
        }
    return obj


def _deserialize(obj, cls=None):
    """Recursively reconstruct dataclasses and enums from dicts."""
    if isinstance(obj, dict):
        if "__enum__" in obj:
            return _ENUM_TYPES[obj["__enum__"]](obj["value"])
        if "__ndarray__" in obj:
            return jnp.array(obj["__ndarray__"], dtype=obj.get("dtype", "float32"))
        if "__dataclass__" in obj:
            dc_cls = _DATACLASS_TYPES[obj["__dataclass__"]]

            # --- Backwards compatibility: fo_physics_config → fo_physics_config ---
            if (
                dc_cls is ModelConfig
                and "fo_physics_config" in obj
                and "fo_physics_config" not in obj
            ):
                obj["fo_physics_config"] = obj.pop("fo_physics_config")

            kwargs = {}
            for f in fields(dc_cls):
                if f.name in obj:
                    val = obj[f.name]
                    if val is None:
                        continue  # Skip None (PRNG keys etc.); default_factory regenerates
                    kwargs[f.name] = _deserialize(val)
            return dc_cls(**kwargs)
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj

def hyperparams_to_json(hp: HyperParameters, path: str | Path) -> None:
    """Save HyperParameters to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialize(hp), f, indent=2)


def hyperparams_from_json(path: str | Path) -> HyperParameters:
    """Load HyperParameters from a JSON file."""
    with open(path) as f:
        return _deserialize(json.load(f))


class FeatureGroup(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    RPM = "rpm"
    CONTROL = "control"


class ScalingGroup(Enum):
    """Groups of features that share the same standardisation scale."""

    ETA_XY = "eta_xy"
    ETA_MZ = "eta_mz"
    NU_XY = "nu_xy"
    NU_MZ = "nu_mz"
    TUNNEL_THRUSTER = "tunnel_thruster"
    FIXED_THRUSTER = "fixed_thruster"

    # FO 6-DOF model scaling groups
    FO_ETA_XY = "fo_eta_xy"
    FO_ETA_Z = "fo_eta_z"
    FO_ETA_RP = "fo_eta_rp"
    FO_ETA_YAW = "fo_eta_yaw"
    FO_NU_XY = "fo_nu_xy"
    FO_NU_Z = "fo_nu_z"
    FO_NU_RP = "fo_nu_rp"
    FO_NU_YAW = "fo_nu_yaw"
    FO_THRUSTER = "fo_thruster"
    FO_AZIMUTH = "fo_azimuth"
    FO_CONTROL_XY = "fo_control_xy"
    FO_CONTROL_Z = "fo_control_z"


# Register ScalingGroup for JSON serialization (defined after _ENUM_TYPES)
_ENUM_TYPES["ScalingGroup"] = ScalingGroup


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    group: FeatureGroup
    unit: str
    label: str  # human-readable axis label
    scaling_group: ScalingGroup = (
        ScalingGroup.TUNNEL_THRUSTER
    )  # default; overridden per feature


FEATURE_REGISTRY: dict[str, FeatureInfo] = {
    "pos_eta_x": FeatureInfo(
        "pos_eta_x", FeatureGroup.POSITION, "m", r"$\eta_x$", ScalingGroup.ETA_XY
    ),
    "pos_eta_y": FeatureInfo(
        "pos_eta_y", FeatureGroup.POSITION, "m", r"$\eta_y$", ScalingGroup.ETA_XY
    ),
    "pos_eta_mz": FeatureInfo(
        "pos_eta_mz", FeatureGroup.POSITION, "rad", r"$\psi$", ScalingGroup.ETA_MZ
    ),
    "pos_nu_x": FeatureInfo(
        "pos_nu_x", FeatureGroup.VELOCITY, "m/s", r"$\nu_x$", ScalingGroup.NU_XY
    ),
    "pos_nu_y": FeatureInfo(
        "pos_nu_y", FeatureGroup.VELOCITY, "m/s", r"$\nu_y$", ScalingGroup.NU_XY
    ),
    "pos_nu_mz": FeatureInfo(
        "pos_nu_mz", FeatureGroup.VELOCITY, "rad/s", r"$r$", ScalingGroup.NU_MZ
    ),
    "rpm_bow_fore": FeatureInfo(
        "rpm_bow_fore",
        FeatureGroup.RPM,
        "RPM",
        "Bow fore",
        ScalingGroup.TUNNEL_THRUSTER,
    ),
    "rpm_bow_aft": FeatureInfo(
        "rpm_bow_aft", FeatureGroup.RPM, "RPM", "Bow aft", ScalingGroup.TUNNEL_THRUSTER
    ),
    "rpm_stern_fore": FeatureInfo(
        "rpm_stern_fore",
        FeatureGroup.RPM,
        "RPM",
        "Stern fore",
        ScalingGroup.TUNNEL_THRUSTER,
    ),
    "rpm_stern_aft": FeatureInfo(
        "rpm_stern_aft",
        FeatureGroup.RPM,
        "RPM",
        "Stern aft",
        ScalingGroup.TUNNEL_THRUSTER,
    ),
    "rpm_fixed_ps": FeatureInfo(
        "rpm_fixed_ps", FeatureGroup.RPM, "RPM", "Fixed PS", ScalingGroup.FIXED_THRUSTER
    ),
    "rpm_fixed_sb": FeatureInfo(
        "rpm_fixed_sb", FeatureGroup.RPM, "RPM", "Fixed SB", ScalingGroup.FIXED_THRUSTER
    ),
}

DEFAULT_FEATURES: list[str] = list(FEATURE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# FO (Marine Systems Simulator) features — 6-DOF vessel with azimuth thrusters
# ---------------------------------------------------------------------------


class FullOrderFeatures(Enum):
    """Feature column names for FO simulation data."""

    ETA_0 = "eta_0"  # North position (m)
    ETA_1 = "eta_1"  # East position (m)
    ETA_2 = "eta_2"  # Down position (m)
    ETA_3 = "eta_3"  # Roll angle (rad)
    ETA_4 = "eta_4"  # Pitch angle (rad)
    ETA_5 = "eta_5"  # Yaw angle (rad)
    NU_0 = "nu_0"  # Surge velocity (m/s)
    NU_1 = "nu_1"  # Sway velocity (m/s)
    NU_2 = "nu_2"  # Heave velocity (m/s)
    NU_3 = "nu_3"  # Roll rate (rad/s)
    NU_4 = "nu_4"  # Pitch rate (rad/s)
    NU_5 = "nu_5"  # Yaw rate (rad/s)
    N_ACT_0 = "n_actual_0"  # Thruster 0 RPM (bow tunnel 1)
    N_ACT_1 = "n_actual_1"  # Thruster 1 RPM (bow tunnel 2)
    N_ACT_2 = "n_actual_2"  # Thruster 2 RPM (stern azimuth PS)
    N_ACT_3 = "n_actual_3"  # Thruster 3 RPM (stern azimuth SB)
    ALPHA_ACT_0 = "alpha_actual_0"  # Azimuth angle 0 (rad)
    ALPHA_ACT_1 = "alpha_actual_1"  # Azimuth angle 1 (rad)
    N_CMD_0 = "n_cmd_0"  # Demanded RPM for thruster 0 (bow tunnel 1)
    N_CMD_1 = "n_cmd_1"  # Demanded RPM for thruster 1 (bow tunnel 2)
    N_CMD_2 = "n_cmd_2"  # Demanded RPM for thruster 2 (stern azimuth PS)
    N_CMD_3 = "n_cmd_3"  # Demanded RPM for thruster 3 (stern azimuth SB)
    ALPHA_CMD_0 = "alpha_cmd_0"  # Demanded azimuth angle 0 (rad)
    ALPHA_CMD_1 = "alpha_cmd_1"  # Demanded azimuth angle 1 (rad)
    CONTR_TOTAL_0 = "tau_cmd_0"  # Total control input
    CONTR_P_0 = "pid_tau_p_0"  # Proportional control input
    CONTR_D_0 = "pid_tau_d_0"  # Derivative control input
    CONTR_I_0 = "pid_tau_i_0"  # Integral control input
    CONTR_I_ERR_0 = "pid_z_int_0"  # Integral state
    CONTR_TOTAL_1 = "tau_cmd_1"  # Total control input
    CONTR_P_1 = "pid_tau_p_1"  # Proportional control input
    CONTR_D_1 = "pid_tau_d_1"  # Derivative control input
    CONTR_I_1 = "pid_tau_i_1"  # Integral control input
    CONTR_I_ERR_1 = "pid_z_int_1"  # Integral state
    CONTR_TOTAL_2 = "tau_cmd_2"  # Total control input
    CONTR_P_2 = "pid_tau_p_2"  # Proportional control input
    CONTR_D_2 = "pid_tau_d_2"  # Derivative control input
    CONTR_I_2 = "pid_tau_i_2"  # Integral control input
    CONTR_I_ERR_2 = "pid_z_int_2"  # Integral state


FO_3DOF_FEATURES: list[FullOrderFeatures] = [
    FullOrderFeatures.ETA_0,
    FullOrderFeatures.ETA_1,
    FullOrderFeatures.ETA_5,
    FullOrderFeatures.NU_0,
    FullOrderFeatures.NU_1,
    FullOrderFeatures.NU_5,
    FullOrderFeatures.N_ACT_0,
    FullOrderFeatures.N_ACT_1,
    FullOrderFeatures.N_ACT_2,
    FullOrderFeatures.N_ACT_3,
    FullOrderFeatures.ALPHA_ACT_0,
    FullOrderFeatures.ALPHA_ACT_1,
    FullOrderFeatures.N_CMD_0,
    FullOrderFeatures.N_CMD_1,
    FullOrderFeatures.N_CMD_2,
    FullOrderFeatures.N_CMD_3,
    FullOrderFeatures.ALPHA_CMD_0,
    FullOrderFeatures.ALPHA_CMD_1,
    FullOrderFeatures.CONTR_TOTAL_0,
    FullOrderFeatures.CONTR_TOTAL_1,
    FullOrderFeatures.CONTR_TOTAL_2,
    FullOrderFeatures.CONTR_P_0,
    FullOrderFeatures.CONTR_P_1,
    FullOrderFeatures.CONTR_P_2,
    FullOrderFeatures.CONTR_D_0,
    FullOrderFeatures.CONTR_D_1,
    FullOrderFeatures.CONTR_D_2,
    FullOrderFeatures.CONTR_I_0,
    FullOrderFeatures.CONTR_I_1,
    FullOrderFeatures.CONTR_I_2,
    FullOrderFeatures.CONTR_I_ERR_0,
    FullOrderFeatures.CONTR_I_ERR_1,
    FullOrderFeatures.CONTR_I_ERR_2,
]

FO_6DOF_FEATURES: list[FullOrderFeatures] = list(FullOrderFeatures)


FO_FEATURE_REGISTRY: dict[str, FeatureInfo] = {
    "eta_0": FeatureInfo(
        "eta_0", FeatureGroup.POSITION, "m", r"$\eta_x$", ScalingGroup.FO_ETA_XY
    ),
    "eta_1": FeatureInfo(
        "eta_1", FeatureGroup.POSITION, "m", r"$\eta_y$", ScalingGroup.FO_ETA_XY
    ),
    "eta_2": FeatureInfo(
        "eta_2", FeatureGroup.POSITION, "m", r"$\eta_z$", ScalingGroup.FO_ETA_Z
    ),
    "eta_3": FeatureInfo(
        "eta_3", FeatureGroup.POSITION, "rad", r"$\phi$", ScalingGroup.FO_ETA_RP
    ),
    "eta_4": FeatureInfo(
        "eta_4", FeatureGroup.POSITION, "rad", r"$\theta$", ScalingGroup.FO_ETA_RP
    ),
    "eta_5": FeatureInfo(
        "eta_5", FeatureGroup.POSITION, "rad", r"$\psi$", ScalingGroup.FO_ETA_YAW
    ),
    "nu_0": FeatureInfo(
        "nu_0", FeatureGroup.VELOCITY, "m/s", r"$u$", ScalingGroup.FO_NU_XY
    ),
    "nu_1": FeatureInfo(
        "nu_1", FeatureGroup.VELOCITY, "m/s", r"$v$", ScalingGroup.FO_NU_XY
    ),
    "nu_2": FeatureInfo(
        "nu_2", FeatureGroup.VELOCITY, "m/s", r"$w$", ScalingGroup.FO_NU_Z
    ),
    "nu_3": FeatureInfo(
        "nu_3", FeatureGroup.VELOCITY, "rad/s", r"$p$", ScalingGroup.FO_NU_RP
    ),
    "nu_4": FeatureInfo(
        "nu_4", FeatureGroup.VELOCITY, "rad/s", r"$q$", ScalingGroup.FO_NU_RP
    ),
    "nu_5": FeatureInfo(
        "nu_5", FeatureGroup.VELOCITY, "rad/s", r"$r$", ScalingGroup.FO_NU_YAW
    ),
    "n_0": FeatureInfo(
        "n_actual_0", FeatureGroup.RPM, "RPM", "Bow tunnel 1", ScalingGroup.FO_THRUSTER
    ),
    "n_1": FeatureInfo(
        "n_actual_1", FeatureGroup.RPM, "RPM", "Bow tunnel 2", ScalingGroup.FO_THRUSTER
    ),
    "n_2": FeatureInfo(
        "n_actual_2",
        FeatureGroup.RPM,
        "RPM",
        "Stern azimuth PS",
        ScalingGroup.FO_THRUSTER,
    ),
    "n_3": FeatureInfo(
        "n_actual_3",
        FeatureGroup.RPM,
        "RPM",
        "Stern azimuth SB",
        ScalingGroup.FO_THRUSTER,
    ),
    "alpha_0": FeatureInfo(
        "alpha_actual_0",
        FeatureGroup.RPM,
        "rad",
        r"$\alpha_0$",
        ScalingGroup.FO_AZIMUTH,
    ),
    "alpha_1": FeatureInfo(
        "alpha_actual_1",
        FeatureGroup.RPM,
        "rad",
        r"$\alpha_1$",
        ScalingGroup.FO_AZIMUTH,
    ),
    "n_cmd_0": FeatureInfo(
        "n_cmd_0",
        FeatureGroup.RPM,
        "RPM",
        "Demanded RPM for thruster 0 (bow tunnel 1)",
        ScalingGroup.FO_THRUSTER,
    ),
    "n_cmd_1": FeatureInfo(
        "n_cmd_1",
        FeatureGroup.RPM,
        "RPM",
        "Demanded RPM for thruster 1 (bow tunnel 2)",
        ScalingGroup.FO_THRUSTER,
    ),
    "n_cmd_2": FeatureInfo(
        "n_cmd_2",
        FeatureGroup.RPM,
        "RPM",
        "Demanded RPM for thruster 2 (stern azimuth PS)",
        ScalingGroup.FO_THRUSTER,
    ),
    "n_cmd_3": FeatureInfo(
        "n_cmd_3",
        FeatureGroup.RPM,
        "RPM",
        "Demanded RPM for thruster 3 (stern azimuth SB)",
        ScalingGroup.FO_THRUSTER,
    ),
    "alpha_cmd_0": FeatureInfo(
        "alpha_cmd_0",
        FeatureGroup.RPM,
        "rad",
        "Demanded azimuth angle 0 (rad)",
        ScalingGroup.FO_AZIMUTH,
    ),
    "alpha_cmd_1": FeatureInfo(
        "alpha_cmd_1",
        FeatureGroup.RPM,
        "rad",
        "Demanded azimuth angle 1 (rad)",
        ScalingGroup.FO_AZIMUTH,
    ),
    "tau_cmd_0": FeatureInfo(
        "tau_cmd_0",
        FeatureGroup.CONTROL,
        "unit",
        "Total control input (Surge)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "tau_cmd_1": FeatureInfo(
        "tau_cmd_1",
        FeatureGroup.CONTROL,
        "unit",
        "Total control input (Sway)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "tau_cmd_2": FeatureInfo(
        "tau_cmd_2",
        FeatureGroup.CONTROL,
        "unit",
        "Total control input (Yaw)",
        ScalingGroup.FO_CONTROL_Z,
    ),
    "pid_tau_p_0": FeatureInfo(
        "pid_tau_p_0",
        FeatureGroup.CONTROL,
        "unit",
        "Proportional control input (Surge)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_p_1": FeatureInfo(
        "pid_tau_p_1",
        FeatureGroup.CONTROL,
        "unit",
        "Proportional control input (Sway)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_p_2": FeatureInfo(
        "pid_tau_p_2",
        FeatureGroup.CONTROL,
        "unit",
        "Proportional control input (Yaw)",
        ScalingGroup.FO_CONTROL_Z,
    ),
    "pid_tau_d_0": FeatureInfo(
        "pid_tau_d_0",
        FeatureGroup.CONTROL,
        "unit",
        "Derivative control input (Surge)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_d_1": FeatureInfo(
        "pid_tau_d_1",
        FeatureGroup.CONTROL,
        "unit",
        "Derivative control input (Sway)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_d_2": FeatureInfo(
        "pid_tau_d_2",
        FeatureGroup.CONTROL,
        "unit",
        "Derivative control input (Yaw)",
        ScalingGroup.FO_CONTROL_Z,
    ),
    "pid_tau_i_0": FeatureInfo(
        "pid_tau_i_0",
        FeatureGroup.CONTROL,
        "unit",
        "Integral control input (Surge)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_i_1": FeatureInfo(
        "pid_tau_i_1",
        FeatureGroup.CONTROL,
        "unit",
        "Integral control input (Sway)",
        ScalingGroup.FO_CONTROL_XY,
    ),
    "pid_tau_i_2": FeatureInfo(
        "pid_tau_i_2",
        FeatureGroup.CONTROL,
        "unit",
        "Integral control input (Yaw)",
        ScalingGroup.FO_CONTROL_Z,
    ),
    "pid_z_int_0": FeatureInfo(
        "pid_z_int_0",
        FeatureGroup.CONTROL,
        "unit",
        "Integral state (Surge)",
        ScalingGroup.FO_ETA_XY,
    ),
    "pid_z_int_1": FeatureInfo(
        "pid_z_int_1",
        FeatureGroup.CONTROL,
        "unit",
        "Integral state (Sway)",
        ScalingGroup.FO_ETA_XY,
    ),
    "pid_z_int_2": FeatureInfo(
        "pid_z_int_2",
        FeatureGroup.CONTROL,
        "unit",
        "Integral state (Yaw)",
        ScalingGroup.FO_ETA_Z,
    ),
}

# Merge FO features into the global registry for scaling_groups_for_features()
FEATURE_REGISTRY.update(FO_FEATURE_REGISTRY)

# Register FullOrderFeatures for JSON serialization (defined after _ENUM_TYPES)
_ENUM_TYPES["FullOrderFeatures"] = FullOrderFeatures


def features_by_group(group: FeatureGroup) -> list[str]:
    """Return feature names belonging to *group*."""
    return [k for k, v in FEATURE_REGISTRY.items() if v.group == group]


def scaling_groups_for_features(feat_names: list[str]) -> dict[str, list[int]]:
    """Map each ScalingGroup to its column indices within *feat_names*.

    Only groups with >=2 members are returned (singletons need no pooling).
    Features not in FEATURE_REGISTRY are left with per-feature scaling.
    """
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(feat_names):
        info = FEATURE_REGISTRY.get(name)
        if info is not None:
            groups[info.scaling_group.value].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# Backward-compat aliases (old names → new names)
# ---------------------------------------------------------------------------
FOPhysicsConfig = FullOrderPhysicsConfig
FOFeatures = FullOrderFeatures
FO_3DOF_FEATURES = FO_3DOF_FEATURES
FO_6DOF_FEATURES = FO_6DOF_FEATURES
_ENUM_TYPES["FOFeatures"] = FullOrderFeatures

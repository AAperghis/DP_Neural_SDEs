# Efficient Probabilistic Forecasting of Dynamic Positioning Behaviour

MSc thesis, Marine Technology, Delft University of Technology — in collaboration with Allseas Engineering BV.

## Abstract

Dynamic positioning (DP) operability is assessed using quasi-static force balances, which estimate the environmental
conditions a vessel can hold but cannot resolve the transient excursions and footprint that often set the true
operational limit. Time-domain simulation captures these effects and is increasingly used to predict DP behaviour more
accurately. Both approaches are deterministic. They do not account for the uncertainty introduced by the DP system
model and by the stochastic sea state, so a single run cannot characterise the distribution of the response.
Probabilistic methods can, but quantifying uncertainty with deterministic simulators requires Monte Carlo sampling, and even then only parameter uncertainty can be captured; model uncertainty as a result of simulator inaccuracy remains out of reach. The resulting computational cost is too high for routine operational use. This work investigates whether machine-learned
surrogate models can feasibly produce probabilistic short-term DP forecasts at a cost low enough for operational
screening, and identifies the challenges of implementing such a model.

A physics-informed hybrid stochastic differential equation (SDE) is developed for probabilistic short-term DP
forecasting. The nominal DP dynamics are modelled in a deterministic Newtonian term, and the remaining unmodelled behaviour
is captured by a latent neural SDE trained from simulated data. A deep ensemble of independently trained members quantifies the
epistemic uncertainty. The method is evaluated against a full-order reference model on simulation data covering 128 sea
states across the operational scatter diagram of a realistic DP vessel.

The hybrid SDE reproduces the statistical properties of the vessel motions and thrust activity across the sampled sea
states and recovers the closed-loop coupling structure of the DP system, though the strength of the coupling is
underestimated. A batch of three-hour trajectories is generated in an average of 11.04s, with parallelisation resulting in a per trajectory cost of 0.015s. This replaces a full
time-domain solve per realisation. The deep ensemble separates the aleatoric and epistemic contributions and yields a
P90 (90th-percentile) excursion estimate that is conservative for most sea states. A posterior warm-start, which
initialises the forecast from the inferred latent state, reduces the forecast median error to below 0.15m
at all horizons. The main limitations are the weaker-than-reference motion-to-thrust coupling in the generated
trajectories, the overconfident ensemble, and the practical barriers of training cost and simulation-only validation.
Unlike existing quasi-static probabilistic methods, the approach delivers time-domain DP forecasts with a separated
aleatoric and epistemic budget, at a fraction of the cost of the reference simulation. The method is
demonstrated to be feasible. The limitations identified are in the training and data preparation pipeline, not in the
fundamental approach.

## Repository structure

The code lives in a single package, `thesis`, under [src/thesis](src/thesis):

| Module | Description |
| --- | --- |
| [full_order_dp](src/thesis/full_order_dp) | Full-order 6-DOF DP vessel simulator (MSS supply vessel): hydrodynamics, wave environment and drift loads, thruster dynamics and allocation, GNC. Used to generate training data ([gen_training_data.py](src/thesis/full_order_dp/gen_training_data.py)). |
| [reduced_order_dp](src/thesis/reduced_order_dp) | Reduced-order 3-DOF DP simulator with Ornstein–Uhlenbeck disturbance modelling ([mainLoop.py](src/thesis/reduced_order_dp/mainLoop.py)). |
| [full_hybrid_sde](src/thesis/full_hybrid_sde) | `FullHybridSDE` — physics-informed latent neural SDE trained on full-order simulation data, with wave conditioning $(H_s, T_p, \beta_w)$ and azimuth-thruster nominal dynamics. |
| [reduced_hybrid_sde](src/thesis/reduced_hybrid_sde) | `ReducedHybridSDE` — the equivalent hybrid neural SDE for the reduced-order model. |
| [shared](src/thesis/shared) | Common infrastructure: training loop, hyper-parameter dataclasses and JSON (de)serialisation, parquet datasets, neural vector fields, KL/noise annealing schedules. |
| [deep_ensemble](src/thesis/deep_ensemble) | Loading, sampling, and trajectory-feasibility filtering for deep ensembles of trained models; downloading ensembles from MLflow. |
| [statistics](src/thesis/statistics) | Ensemble statistics: moments, spectral distributions, POT-based extreme value analysis, plotting, and online training metrics. |

Training runs, checkpoints, and figures are tracked with [MLflow](https://mlflow.org/).

## Installation

Requires Python ≥ 3.11, < 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AAperghis/DP_Neural_SDEs
cd DP_Neural_SDEs
uv sync            # default: JAX with CUDA 12 support
```

JAX backends are managed through mutually exclusive dependency groups. To use a different backend:

```bash
uv sync --no-default-groups --group cpu   # CPU-only
uv sync --no-default-groups --group amd   # ROCm (local install)
```

## Running

All commands run inside the project environment via `uv run`.

### 1. Generate training data

Run an ensemble of full-order DP simulations across sampled sea states and write the results to parquet:

```bash
uv run python -m thesis.full_order_dp.gen_training_data
```

### 2. Consolidate parquet files

Merge the per-run parquet files into a single consolidated file with per-run wave metadata:

```bash
uv run python -m thesis.shared.consolidate_parquet \
    --input-dir <dir-with-run-parquets> \
    --output <path/to/consolidated.parquet>
```

### 3. Train a model

Train the full-order hybrid SDE (hyper-parameters are supplied as JSON; defaults are used if omitted):

```bash
uv run python -m thesis.full_hybrid_sde.train \
    --data_path <path/to/consolidated.parquet> \
    --hyperparams <path/to/hyperparams.json> \
    --device gpu \
    --run_name my_run
```

The reduced-order pipeline is analogous (here `--hyperparams` is required):

```bash
uv run python -m thesis.reduced_hybrid_sde.train \
    --data_path <path/to/data> \
    --hyperparams <path/to/hyperparams.json>
```

Training progress, metrics, checkpoints, and sample figures are logged to the configured MLflow experiment.

### 4. Resume training

Continue a run from an MLflow checkpoint:

```bash
uv run python -c "from thesis.full_hybrid_sde.train import resume_cli; resume_cli()" \
    --source_run_id <mlflow-run-id> \
    --checkpoint final \
    --extra_steps 1000 \
    --data_path <path/to/consolidated.parquet>
```

### 5. Evaluate

Ensemble sampling and statistical evaluation (moments, PSDs, extreme value analysis) are provided by [thesis.deep_ensemble](src/thesis/deep_ensemble) and [thesis.statistics](src/thesis/statistics), intended for use from notebooks or scripts:

```python
from thesis.deep_ensemble import ...
from thesis.statistics import ensemble_moments, plot_moments_vs_hs
```

## Licence

See [LICENSE](LICENSE).

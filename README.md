# Federated Mixture of Experts Scaffold

This repository now includes a modular PyTorch-based scaffold for developing a
federated Mixture of Experts (MoE) research pipeline. The project structure is
organised to keep configuration, utilities, models, and experiment scripts
isolated and composable.

## Project Structure

```
├── blockchain/        # Placeholder for consensus, incentives, and audit tools
├── config/            # Central configuration utilities
├── experiments/       # Experiment entry points and notebooks
├── federated/         # Federated orchestration logic
├── models/            # Model definitions and gating networks
├── privacy/           # Differential privacy and secure aggregation code
├── utils/             # Shared helpers (data loading, metrics, logging, ...)
```

Each package currently ships with an `__init__.py` stub so modules can be added
incrementally without breaking imports.

## Configuration

Hyperparameters, logging behaviour, and component toggles are centralised in
`config/config.py`. The module exposes `load_config`, which supports overrides
through:

- Environment variables prefixed with `FED_MOE_` (use double underscores to
  denote nesting, e.g. `FED_MOE_TRAINING__BATCH_SIZE=64`).
- Command-line arguments such as `--training-learning-rate 5e-4`
- Programmatic overrides via keyword arguments to `load_config`.

`load_config` also normalises key paths (e.g. expanding `~/` in logging
directories) so downstream modules can rely on consistent file locations.

## Utilities

- `utils/data_loader.py` implements dataset loading with automatic synthetic
  fallbacks, IID/non-IID client partitioning, and federated `DataLoader`
  creation.
- `utils/metrics.py` provides reusable accuracy computation, running-average
  tracking, privacy budget reporting, and helpers to aggregate metrics from
  multiple clients.
- `utils/logging_utils.py` offers structured JSON logging with timestamped
  experiment directories for checkpoints, logs, and artefacts.

These utilities are intentionally framework-agnostic and ready to plug into
future training scripts or notebooks.

## Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` presently tracks the core dependencies used by the scaffold
(Pytorch and NumPy). Additional libraries (TensorFlow, TensorFlow Probability,
scikit-learn, etc.) remain available within the notebooks but can be added to
`requirements.txt` as the Python package surface grows.

## Next Steps

- Implement model definitions under `models/` and federated orchestration code
  under `federated/`.
- Populate `experiments/` with training/evaluation pipelines that consume the
  shared utilities.
- Extend `privacy/` and `blockchain/` as differential privacy and auditability
  features are integrated.

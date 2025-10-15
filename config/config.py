"""Central configuration utilities for the federated Mixture of Experts project.

This module exposes a configurable dictionary-based configuration system with
sensible defaults for training, data management, federated aggregation,
privacy, and logging. Configuration values can be overridden hierarchically via
command-line arguments, direct overrides, or environment variables prefixed
with ``FED_MOE_``.

Example usage
-------------

>>> from config.config import load_config
>>> config = load_config()
>>> batch_size = config["training"]["batch_size"]

Environment overrides use double underscores to express nesting. For example,
``FED_MOE_TRAINING__BATCH_SIZE=128`` will override the default training batch
size. Command-line overrides follow the same convention via argument names such
as ``--training-batch-size 128``.

Transformer + MoE Configuration
--------------------------------

The ``transformer`` configuration section controls the architecture of the
Transformer backbone with optional Mixture-of-Experts integration:

- ``embedding_dim``: Dimensionality of token embeddings and all intermediate
  representations throughout the network (default: 256).
- ``num_layers``: Number of Transformer encoder blocks to stack (default: 6).
- ``num_heads``: Number of attention heads in multi-head attention (default: 8).
  Must evenly divide embedding_dim.
- ``ff_dim``: Dimensionality of feed-forward hidden layer. If None, defaults to
  4 * embedding_dim (default: None).
- ``dropout``: Dropout probability applied after attention and feed-forward
  sublayers (default: 0.1).
- ``positional_encoding``: Type of positional encoding - one of "sinusoidal"
  (fixed sine/cosine), "learned" (trainable embeddings), or "none" (no
  positional information added) (default: "sinusoidal").
- ``max_seq_length``: Maximum sequence length for positional encoding
  (default: 512).
- ``use_moe``: Global flag to enable MoE blocks. When True, MoE replaces
  feed-forward layers in all or selected layers (default: False).
- ``moe_layer_indices``: Optional list of 0-based layer indices where MoE
  should be inserted (e.g., [2, 4] for layers 2 and 4 only). If None and
  use_moe=True, all layers use MoE (default: None).

The ``moe`` section interacts with transformer layers when ``use_moe`` is
enabled:

- ``num_experts``: Number of expert networks in each MoE layer (default: 4).
- ``top_k``: Number of experts to activate per token (default: 2). Must be
  <= num_experts.
- ``use_gating_network``: Whether to use learned gating (True) or random
  routing (False) (default: True).
- ``temperature``: Temperature for gating softmax. Lower values produce more
  discrete routing, higher values produce smoother blending (default: 1.0).
- ``load_balance_weight``: Weight for auxiliary load-balancing loss that
  encourages uniform expert utilization. Higher values enforce stronger
  balancing (default: 0.01).

Example Transformer+MoE configuration:

>>> config = load_config(overrides={
...     "transformer": {
...         "embedding_dim": 512,
...         "num_layers": 8,
...         "num_heads": 8,
...         "use_moe": True,
...         "moe_layer_indices": [3, 5, 7],  # MoE only in layers 3, 5, 7
...     },
...     "moe": {
...         "num_experts": 8,
...         "top_k": 2,
...         "temperature": 0.5,
...     },
... })
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

ENV_PREFIX = "FED_MOE_"

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "training": {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "scheduler": None,
    },
    "federated": {
        "num_clients": 5,
        "participation_rate": 0.6,
        "aggregation": "fedavg",
        "non_iid_alpha": 0.5,
        "local_epochs": 1,
    },
    "moe": {
        "num_experts": 4,
        "top_k": 2,
        "use_gating_network": True,
        "temperature": 1.0,
        "load_balance_weight": 0.01,
    },
    "transformer": {
        "embedding_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ff_dim": None,
        "dropout": 0.1,
        "positional_encoding": "sinusoidal",
        "max_seq_length": 512,
        "use_moe": False,
        "moe_layer_indices": None,
    },
    "privacy": {
        "enable_dp": False,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
        "target_epsilon": None,
        "target_delta": 1e-5,
    },
    "blockchain": {
        "enabled": False,
        "provider": "local",
        "endpoint": "http://localhost:8545",
        "wallet_key": None,
    },
    "data": {
        "dataset": "synthetic",
        "path": "./data",
        "iid": True,
        "validation_split": 0.1,
        "input_dim": 20,
        "num_classes": 2,
        "num_samples": 1024,
    },
    "logging": {
        "output_dir": "./results",
        "experiment_name": "federated_moe",
        "log_level": "INFO",
        "logger_name": "fed_moe",
    },
    "experiment": {
        "resume_from_checkpoint": None,
        "checkpoint_interval": 1,
        "use_wandb": False,
        "notes": "",
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default configuration dictionary."""

    return copy.deepcopy(DEFAULT_CONFIG)


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base``.

    Args:
        base: The configuration dictionary to update in-place.
        updates: A mapping containing updates to apply.

    Returns:
        The updated ``base`` mapping for convenience.
    """

    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _assign_nested(mapping: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    current = mapping
    *parents, leaf = list(path)
    for part in parents:
        current = current.setdefault(part, {})  # type: ignore[assignment]
    current[leaf] = value


def _parse_env_value(value: str) -> Any:
    """Attempt to parse an environment value into a native Python object."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        # Handle numbers, lists, dictionaries, etc.
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _collect_env_overrides(env_prefix: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    prefix_length = len(env_prefix)
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        path = key[prefix_length:].lower().split("__")
        if not path:
            continue
        parsed_value = _parse_env_value(value)
        _assign_nested(overrides, path, parsed_value)
    return overrides


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser exposing common configuration overrides."""

    parser = argparse.ArgumentParser(
        description="Federated Mixture of Experts configuration overrides",
        add_help=False,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")

    # Training arguments
    parser.add_argument("--training-batch-size", type=int, dest="training__batch_size")
    parser.add_argument("--training-epochs", type=int, dest="training__epochs")
    parser.add_argument("--training-learning-rate", type=float, dest="training__learning_rate")
    parser.add_argument("--training-weight-decay", type=float, dest="training__weight_decay")
    parser.add_argument("--training-optimizer", dest="training__optimizer")

    # Federated learning arguments
    parser.add_argument("--federated-num-clients", type=int, dest="federated__num_clients")
    parser.add_argument("--federated-participation-rate", type=float, dest="federated__participation_rate")
    parser.add_argument("--federated-aggregation", dest="federated__aggregation")
    parser.add_argument("--federated-non-iid-alpha", type=float, dest="federated__non_iid_alpha")
    parser.add_argument("--federated-local-epochs", type=int, dest="federated__local_epochs")

    # Mixture of Experts arguments
    parser.add_argument("--moe-num-experts", type=int, dest="moe__num_experts")
    parser.add_argument("--moe-top-k", type=int, dest="moe__top_k")
    parser.add_argument("--moe-use-gating", type=_parse_bool, dest="moe__use_gating_network")
    parser.add_argument("--moe-temperature", type=float, dest="moe__temperature")
    parser.add_argument("--moe-load-balance-weight", type=float, dest="moe__load_balance_weight")

    # Transformer arguments
    parser.add_argument("--transformer-embedding-dim", type=int, dest="transformer__embedding_dim")
    parser.add_argument("--transformer-num-layers", type=int, dest="transformer__num_layers")
    parser.add_argument("--transformer-num-heads", type=int, dest="transformer__num_heads")
    parser.add_argument("--transformer-ff-dim", type=int, dest="transformer__ff_dim")
    parser.add_argument("--transformer-dropout", type=float, dest="transformer__dropout")
    parser.add_argument("--transformer-positional-encoding", dest="transformer__positional_encoding")
    parser.add_argument("--transformer-max-seq-length", type=int, dest="transformer__max_seq_length")
    parser.add_argument("--transformer-use-moe", type=_parse_bool, dest="transformer__use_moe")

    # Privacy arguments
    parser.add_argument("--privacy-enable-dp", type=_parse_bool, dest="privacy__enable_dp")
    parser.add_argument("--privacy-noise-multiplier", type=float, dest="privacy__noise_multiplier")
    parser.add_argument("--privacy-max-grad-norm", type=float, dest="privacy__max_grad_norm")
    parser.add_argument("--privacy-target-epsilon", type=float, dest="privacy__target_epsilon")
    parser.add_argument("--privacy-target-delta", type=float, dest="privacy__target_delta")

    # Data arguments
    parser.add_argument("--data-dataset", dest="data__dataset")
    parser.add_argument("--data-path", dest="data__path")
    parser.add_argument("--data-iid", type=_parse_bool, dest="data__iid")
    parser.add_argument("--data-validation-split", type=float, dest="data__validation_split")
    parser.add_argument("--data-input-dim", type=int, dest="data__input_dim")
    parser.add_argument("--data-num-classes", type=int, dest="data__num_classes")
    parser.add_argument("--data-num-samples", type=int, dest="data__num_samples")

    # Logging arguments
    parser.add_argument("--logging-output-dir", dest="logging__output_dir")
    parser.add_argument("--logging-experiment-name", dest="logging__experiment_name")
    parser.add_argument("--logging-log-level", dest="logging__log_level")

    # Experiment management
    parser.add_argument("--experiment-resume", dest="experiment__resume_from_checkpoint")
    parser.add_argument("--experiment-checkpoint-interval", type=int, dest="experiment__checkpoint_interval")
    parser.add_argument("--experiment-use-wandb", type=_parse_bool, dest="experiment__use_wandb")

    return parser


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'.")


def _namespace_to_config_dict(namespace: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if value is None:
            continue
        path = key.split("__")
        _assign_nested(overrides, path, value)
    return overrides


def load_config(
    cli_args: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    env_prefix: str = ENV_PREFIX,
) -> Dict[str, Any]:
    """Load the runtime configuration, applying overrides as necessary.

    Args:
        cli_args: Optional sequence of command-line arguments (excluding the
            program name). When ``None``, ``argparse`` will read from
            ``sys.argv``. When an empty list is provided, CLI parsing is
            skipped.
        overrides: Arbitrary mapping of configuration overrides. These take
            precedence over defaults and environment variables.
        env_prefix: Prefix used to look up environment variable overrides.

    Returns:
        A nested configuration dictionary ready to be consumed by the rest of
        the codebase.
    """

    config = get_default_config()

    env_overrides = _collect_env_overrides(env_prefix)
    deep_update(config, env_overrides)

    if overrides:
        deep_update(config, overrides)

    parser = build_arg_parser()
    namespace, _unknown = parser.parse_known_args(args=cli_args)
    cli_overrides = _namespace_to_config_dict(namespace)
    deep_update(config, cli_overrides)

    _normalize_paths(config)
    return config


def _normalize_paths(config: MutableMapping[str, Any]) -> None:
    """Expand user directories and resolve relative paths where appropriate."""

    logging_cfg = config.get("logging", {})
    output_dir = logging_cfg.get("output_dir")
    if output_dir:
        logging_cfg["output_dir"] = str(Path(output_dir).expanduser().resolve())

    data_cfg = config.get("data", {})
    data_path = data_cfg.get("path")
    if data_path:
        data_cfg["path"] = str(Path(data_path).expanduser())


__all__ = [
    "ENV_PREFIX",
    "DEFAULT_CONFIG",
    "build_arg_parser",
    "deep_update",
    "get_default_config",
    "load_config",
]

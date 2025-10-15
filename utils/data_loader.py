"""Federated data loading utilities for the Mixture of Experts pipeline.

This module provides helpers to load datasets, partition them across clients
according to IID or non-IID strategies, and create PyTorch ``DataLoader``
instances. When local datasets are unavailable, configurable synthetic data is
used as a fallback to keep experiments reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

Config = Mapping[str, Mapping[str, object]]


@dataclass
class DatasetMetadata:
    """Metadata describing a dataset and its federated partitioning."""

    num_samples: int
    num_classes: int
    input_dim: int
    client_sizes: Dict[int, int]
    source: str


def load_dataset(config: Config) -> Tuple[Dataset, DatasetMetadata]:
    """Load the dataset specified by ``config`` or fall back to synthetic data."""

    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "synthetic"))
    dataset_path = Path(str(data_cfg.get("path", "./data"))).expanduser()
    num_samples = int(data_cfg.get("num_samples", 1024))
    input_dim = int(data_cfg.get("input_dim", 20))
    num_classes = int(data_cfg.get("num_classes", 2))
    seed = int(config.get("seed", 42))

    if dataset_name.lower() == "synthetic":
        dataset = _generate_synthetic_dataset(num_samples, input_dim, num_classes, seed)
        metadata = DatasetMetadata(
            num_samples=len(dataset),
            num_classes=num_classes,
            input_dim=input_dim,
            client_sizes={},
            source="synthetic",
        )
        return dataset, metadata

    try:
        dataset = _load_from_disk(dataset_path)
        inferred_classes = _infer_num_classes(dataset, default=num_classes)
        metadata = DatasetMetadata(
            num_samples=len(dataset),
            num_classes=inferred_classes,
            input_dim=_infer_input_dim(dataset, default=input_dim),
            client_sizes={},
            source=str(dataset_path),
        )
        return dataset, metadata
    except FileNotFoundError:
        dataset = _generate_synthetic_dataset(num_samples, input_dim, num_classes, seed)
        metadata = DatasetMetadata(
            num_samples=len(dataset),
            num_classes=num_classes,
            input_dim=input_dim,
            client_sizes={},
            source=f"synthetic-fallback({dataset_path})",
        )
        return dataset, metadata


def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    iid: bool = True,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, Subset]:
    """Partition ``dataset`` into ``num_clients`` subsets."""

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))

    if iid:
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return {client_id: Subset(dataset, split.tolist()) for client_id, split in enumerate(splits)}

    targets = _get_targets(dataset)
    unique_classes = np.unique(targets)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in unique_classes:
        cls_indices = indices[targets == cls]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = np.floor(proportions * len(cls_indices)).astype(int)
        remainder = len(cls_indices) - counts.sum()
        if remainder > 0:
            extra = rng.choice(num_clients, remainder, replace=True)
            for client_id in extra:
                counts[client_id] += 1
        start = 0
        for client_id, count in enumerate(counts):
            if count == 0:
                continue
            end = start + count
            client_indices[client_id].extend(cls_indices[start:end].tolist())
            start = end

    for idx_list in client_indices:
        rng.shuffle(idx_list)

    return {client_id: Subset(dataset, indices) for client_id, indices in enumerate(client_indices)}


def create_client_dataloaders(
    partitions: Mapping[int, Subset],
    batch_size: int,
    drop_last: bool = False,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Dict[int, DataLoader]:
    """Instantiate PyTorch ``DataLoader`` objects for each client partition."""

    loaders: Dict[int, DataLoader] = {}
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    for client_id, subset in partitions.items():
        loaders[client_id] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            generator=generator if shuffle else None,
        )
    return loaders


def prepare_federated_data(
    config: Config,
    dataset: Optional[Dataset] = None,
    iid: Optional[bool] = None,
    alpha: Optional[float] = None,
) -> Tuple[Dict[int, DataLoader], DatasetMetadata]:
    """Load, partition, and wrap data loaders according to ``config``."""

    seed = int(config.get("seed", 42))
    data_cfg = config.get("data", {})
    fed_cfg = config.get("federated", {})

    if dataset is None:
        dataset, metadata = load_dataset(config)
    else:
        metadata = DatasetMetadata(
            num_samples=len(dataset),
            num_classes=_infer_num_classes(dataset, default=int(data_cfg.get("num_classes", 2))),
            input_dim=_infer_input_dim(dataset, default=int(data_cfg.get("input_dim", 20))),
            client_sizes={},
            source="provided",
        )

    is_iid = bool(data_cfg.get("iid", True) if iid is None else iid)
    non_iid_alpha = float(fed_cfg.get("non_iid_alpha", 0.5) if alpha is None else alpha)
    num_clients = int(fed_cfg.get("num_clients", 1))

    partitions = partition_dataset(
        dataset,
        num_clients=num_clients,
        iid=is_iid,
        alpha=non_iid_alpha,
        seed=seed,
    )

    batch_size = int(config.get("training", {}).get("batch_size", 32))
    loaders = create_client_dataloaders(
        partitions,
        batch_size=batch_size,
        drop_last=False,
        seed=seed,
    )

    metadata.client_sizes = {client_id: len(partition) for client_id, partition in partitions.items()}
    return loaders, metadata


def _load_from_disk(path: Path) -> Dataset:
    if path.is_dir():
        candidates = list(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(path)
        path = candidates[0]

    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(path)
    if isinstance(data, Dataset):
        return data
    if isinstance(data, TensorDataset):
        return data
    if isinstance(data, tuple) and len(data) >= 2:
        features, labels = data[:2]
        return TensorDataset(torch.as_tensor(features), torch.as_tensor(labels))
    if isinstance(data, dict) and {"features", "labels"}.issubset(data.keys()):
        return TensorDataset(torch.as_tensor(data["features"]), torch.as_tensor(data["labels"]))
    raise ValueError(f"Unsupported dataset format at {path}")


def _generate_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, input_dim, generator=generator)
    weights = torch.randn(input_dim, num_classes, generator=generator)
    logits = features @ weights
    probabilities = torch.softmax(logits, dim=1)
    labels = probabilities.multinomial(num_samples=1).squeeze(1)
    return TensorDataset(features, labels)


def _get_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].detach().cpu().numpy()
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return np.asarray(targets)
    if hasattr(dataset, "labels"):
        targets = getattr(dataset, "labels")
        return np.asarray(targets)
    raise ValueError("Cannot extract targets from the provided dataset.")


def _infer_num_classes(dataset: Dataset, default: int) -> int:
    try:
        targets = _get_targets(dataset)
    except ValueError:
        return default
    return int(np.unique(targets).size)


def _infer_input_dim(dataset: Dataset, default: int) -> int:
    sample, *_ = dataset[0]
    if isinstance(sample, torch.Tensor):
        return int(sample.numel())
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], torch.Tensor):
        return int(sample[0].numel())
    return default


__all__ = [
    "DatasetMetadata",
    "create_client_dataloaders",
    "load_dataset",
    "partition_dataset",
    "prepare_federated_data",
]

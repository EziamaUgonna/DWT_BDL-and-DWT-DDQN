"""Evaluation metrics and tracking utilities for federated experiments."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy given predictions and targets."""

    if predictions.ndim > 1:
        predicted_labels = predictions.argmax(dim=1)
    else:
        predicted_labels = (predictions > 0.5).long()
    correct = (predicted_labels == targets).float().mean().item()
    return float(correct)


@dataclass
class MetricsTracker:
    """Utility class for maintaining running averages of metrics."""

    _totals: MutableMapping[str, float] = field(default_factory=lambda: defaultdict(float))
    _counts: MutableMapping[str, float] = field(default_factory=lambda: defaultdict(float))

    def update(self, name: str, value: float, weight: float = 1.0) -> None:
        self._totals[name] += float(value) * weight
        self._counts[name] += weight

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, total in self._totals.items():
            count = self._counts.get(name, 0.0)
            metrics[name] = total / count if count else 0.0
        return metrics

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()


def aggregate_client_metrics(
    client_metrics: Mapping[str, Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Aggregate metrics reported by multiple clients.

    By default, the number of samples per client is used as the weighting
    factor when a ``num_samples`` entry is present in each client's metric
    dictionary.
    """

    if not client_metrics:
        return {}

    if weights is None:
        inferred_weights = {}
        for client_id, metrics in client_metrics.items():
            inferred_weights[client_id] = float(metrics.get("num_samples", 1.0))
        weights = inferred_weights

    total_weight = float(sum(weights.values()))
    if total_weight == 0:
        total_weight = float(len(weights))

    aggregated: Dict[str, float] = {}
    for client_id, metrics in client_metrics.items():
        weight = float(weights.get(client_id, 1.0))
        for metric_name, metric_value in metrics.items():
            if metric_name == "num_samples":
                continue
            aggregated.setdefault(metric_name, 0.0)
            aggregated[metric_name] += float(metric_value) * weight

    for metric_name, total in aggregated.items():
        aggregated[metric_name] = total / total_weight

    aggregated["total_weight"] = total_weight
    return aggregated


def privacy_budget_report(
    epsilon_budget: float,
    delta: float,
    epsilon_spent: Optional[float] = None,
) -> Dict[str, float]:
    """Return a structured privacy budget report."""

    spent = float(epsilon_spent or 0.0)
    remaining = max(epsilon_budget - spent, 0.0) if epsilon_budget else None
    report = {
        "epsilon_budget": float(epsilon_budget) if epsilon_budget else None,
        "epsilon_spent": spent,
        "epsilon_remaining": remaining,
        "delta": float(delta),
    }
    return report


def loss_to_numpy(loss: torch.Tensor) -> float:
    """Safely convert a PyTorch loss tensor into a Python float."""

    if isinstance(loss, torch.Tensor):
        return float(loss.detach().cpu().item())
    return float(loss)


__all__ = [
    "MetricsTracker",
    "accuracy",
    "aggregate_client_metrics",
    "loss_to_numpy",
    "privacy_budget_report",
]

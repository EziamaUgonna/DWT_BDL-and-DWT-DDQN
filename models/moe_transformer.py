"""Mixture-of-Experts Transformer implementation for sequence modelling tasks.

This module provides a configurable Transformer encoder that supports optional
Mixture-of-Experts (MoE) feed-forward blocks with top-k gating and load
balancing. The module is designed to serve as a reusable building block for
federated learning experiments where auxiliary losses and lightweight metadata
are required alongside the primary logits output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


_ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATION_REGISTRY[name.lower()]()
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported activation '{name}'.") from exc


@dataclass
class MoEConfig:
    """Configuration options specific to Mixture-of-Experts layers."""

    num_experts: int = 4
    expert_hidden_size: int = 256
    top_k: int = 2
    load_balancing_weight: float = 1e-2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in the range [1, num_experts].")
        if not 0.0 <= self.load_balancing_weight:
            raise ValueError("load_balancing_weight must be non-negative.")


@dataclass
class MoETransformerConfig:
    """Configuration for the MoE Transformer backbone."""

    input_dim: int
    model_dim: int
    num_classes: int
    num_layers: int = 2
    num_heads: int = 4
    ff_hidden_dim: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 512
    use_moe: bool = True
    moe: Optional[MoEConfig] = None

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.use_moe:
            self.moe = self.moe or MoEConfig()
        else:
            self.moe = None
        _get_activation(self.activation)  # early validation


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                "Sequence length exceeds the configured maximum for positional encoding."
            )
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Standard Transformer feed-forward block."""

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.net(x))


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer with top-k gating."""

    def __init__(self, dim: int, activation: str, config: MoEConfig) -> None:
        super().__init__()
        self.dim = dim
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.load_balancing_weight = config.load_balancing_weight
        self.gate = nn.Linear(dim, self.num_experts)
        self.experts = nn.ModuleList(
            FeedForward(dim, config.expert_hidden_size, config.dropout, activation)
            for _ in range(self.num_experts)
        )
        self.post_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        batch_size, seq_len, _ = x.shape
        gate_logits = self.gate(x)
        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gating_probs = torch.softmax(topk_logits, dim=-1)
        dispatch_mask = torch.zeros_like(gate_logits)
        dispatch_mask.scatter_(-1, topk_indices, gating_probs)

        expert_outputs: List[Tensor] = []
        for idx, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_weight = dispatch_mask[..., idx].unsqueeze(-1)
            expert_outputs.append(expert_out * expert_weight)
        combined = torch.stack(expert_outputs, dim=-1).sum(dim=-1)
        combined = self.post_dropout(combined)

        expert_load = dispatch_mask.sum(dim=(0, 1)) / (batch_size * seq_len)
        uniform = torch.full_like(expert_load, 1.0 / self.num_experts)
        load_balancing_loss = self.load_balancing_weight * torch.sum(
            (expert_load - uniform) ** 2
        )

        metadata = {
            "expert_load": expert_load.detach(),
        }
        return combined, load_balancing_loss, metadata


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with optional MoE feed-forward block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
        use_moe: bool,
        moe_config: Optional[MoEConfig],
    ) -> None:
        super().__init__()
        self.use_moe = use_moe
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        if use_moe and moe_config is not None:
            self.feed_forward = MoELayer(dim, activation, moe_config)
        else:
            self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout, activation)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.attn_dropout(attn_output))

        metadata: Dict[str, Tensor] = {}
        if self.use_moe and isinstance(self.feed_forward, MoELayer):
            ff_output, aux_loss, moe_metadata = self.feed_forward(x)
            metadata.update(moe_metadata)
        else:
            ff_output = self.feed_forward(x)
            aux_loss = x.new_zeros(())

        x = self.norm2(x + ff_output)
        metadata["attention_mean"] = attn_weights.mean(dim=(-1, -2)).detach()
        return x, aux_loss, metadata


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers with metadata aggregation."""

    def __init__(self, config: MoETransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=config.model_dim,
                num_heads=config.num_heads,
                ff_hidden_dim=config.ff_hidden_dim,
                dropout=config.dropout,
                activation=config.activation,
                use_moe=config.use_moe,
                moe_config=config.moe,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self, x: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor], Dict[str, Tensor]]:
        aux_losses: List[Tensor] = []
        expert_loads: List[Tensor] = []
        attention_means: List[Tensor] = []

        for layer in self.layers:
            x, layer_aux_loss, layer_metadata = layer(x, padding_mask)
            aux_losses.append(layer_aux_loss)
            if "expert_load" in layer_metadata:
                expert_loads.append(layer_metadata["expert_load"])
            attention_means.append(layer_metadata["attention_mean"])

        x = self.final_norm(x)

        metadata: Dict[str, Tensor] = {}
        if expert_loads:
            metadata["expert_loads"] = torch.stack(expert_loads)
        metadata["attention_means"] = torch.stack(attention_means)
        return x, aux_losses, metadata


class MoETransformer(nn.Module):
    """Transformer encoder model with optional Mixture-of-Experts blocks."""

    def __init__(self, config: MoETransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.positional_encoding = PositionalEncoding(
            config.model_dim, config.max_seq_len, config.dropout
        )
        self.encoder = TransformerEncoder(config)
        self.output_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.model_dim, config.num_classes)

    def forward(
        self, inputs: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape (batch, sequence, features)")
        x = self.input_projection(inputs)
        x = self.positional_encoding(x)

        encoder_outputs, aux_losses, encoder_metadata = self.encoder(x, padding_mask)
        pooled = self._mean_pool(encoder_outputs, padding_mask)
        logits = self.classifier(self.output_dropout(pooled))

        auxiliary_losses: Dict[str, Tensor] = {}
        if aux_losses:
            stacked_aux = torch.stack(aux_losses)
            auxiliary_losses["load_balancing"] = stacked_aux.sum()
            for idx, loss in enumerate(aux_losses):
                auxiliary_losses[f"layer_{idx}_load_balancing"] = loss
        else:
            auxiliary_losses["load_balancing"] = logits.new_zeros(())

        metadata: Dict[str, Tensor] = dict(encoder_metadata)
        metadata["sequence_length"] = torch.tensor(
            inputs.size(1), device=inputs.device
        )
        if padding_mask is not None:
            valid_tokens = (~padding_mask).sum(dim=1)
            metadata["valid_tokens_per_batch"] = valid_tokens.detach()

        return {
            "logits": logits,
            "auxiliary_losses": auxiliary_losses,
            "metadata": metadata,
        }

    @staticmethod
    def _mean_pool(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        if padding_mask is None:
            return x.mean(dim=1)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(dtype=torch.bool)
        mask = (~padding_mask).unsqueeze(-1).float()
        masked_sum = (x * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / token_count

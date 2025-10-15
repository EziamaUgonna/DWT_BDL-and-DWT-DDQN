"""Configurable Transformer with optional Mixture-of-Experts backbone.

This module provides a flexible Transformer architecture with support for:
- Configurable positional encodings (sinusoidal, learned, or none)
- Multi-head self-attention layers
- Feed-forward networks
- Optional per-layer Mixture-of-Experts (MoE) blocks
- Sequence classification heads

The architecture is entirely driven by configuration values, supporting both
pure Transformer baselines and Transformer+MoE hybrid models. The MoE module
includes top-k routing with temperature scaling, load-balancing auxiliary loss,
and dynamic expert activation for federated learning scenarios.

Configuration interactions:
---------------------------
- `transformer.embedding_dim`: Dimensionality of token embeddings and all
  intermediate representations throughout the network.
- `transformer.num_layers`: Number of Transformer encoder blocks to stack.
- `transformer.num_heads`: Number of attention heads in multi-head attention.
- `transformer.ff_dim`: Dimensionality of feed-forward hidden layer. Defaults
  to 4 * embedding_dim if not specified.
- `transformer.dropout`: Dropout probability applied after attention and FFN.
- `transformer.positional_encoding`: Type of positional encoding - one of
  "sinusoidal", "learned", or "none".
- `transformer.max_seq_length`: Maximum sequence length for positional encoding.
- `transformer.use_moe`: Global flag to enable MoE blocks in all layers.
- `transformer.moe_layer_indices`: Optional list of layer indices (0-based) to
  insert MoE blocks. If None and use_moe=True, all layers use MoE.
- `moe.num_experts`: Number of expert networks in each MoE layer.
- `moe.top_k`: Number of experts to activate per token.
- `moe.use_gating_network`: Whether to use learned gating (vs. random routing).
- `moe.temperature`: Temperature for gating softmax (lower = more discrete).
- `moe.load_balance_weight`: Weight for auxiliary load-balancing loss.

Example usage:
--------------
>>> config = {
...     "data": {"input_dim": 768, "num_classes": 10},
...     "transformer": {
...         "embedding_dim": 256,
...         "num_layers": 6,
...         "num_heads": 8,
...         "dropout": 0.1,
...         "use_moe": True,
...         "moe_layer_indices": [2, 4],  # MoE only in layers 2 and 4
...     },
...     "moe": {"num_experts": 4, "top_k": 2, "temperature": 0.5},
... }
>>> model = TransformerMoE(config)
>>> output = model(input_ids)  # returns {"logits": ..., "auxiliary_loss": ...}
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'."""

    def __init__(self, embedding_dim: int, max_seq_length: int = 5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_seq_length, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, embedding_dim: int, max_seq_length: int = 5000):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_embedding[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with configurable dimensions."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MixtureOfExperts(nn.Module):
    """Mixture-of-Experts layer with top-k routing and load balancing.

    This module routes each token to a subset of expert networks based on a
    learned gating function. It implements:
    - Top-k expert selection per token
    - Temperature-scaled gating for controlling routing discreteness
    - Auxiliary load-balancing loss to encourage uniform expert usage
    - Dynamic expert activation tracking for federated scenarios

    The auxiliary loss is computed as the product of expert assignment
    frequencies and routing weights, encouraging balanced expert utilization.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        top_k: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_gating: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_gating = use_gating
        self.temperature = temperature

        self.experts = nn.ModuleList([
            FeedForward(embedding_dim, ff_dim, dropout) for _ in range(num_experts)
        ])

        if use_gating:
            self.gate = nn.Linear(embedding_dim, num_experts)
        else:
            self.gate = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embedding_dim = x.shape
        x_flat = x.view(-1, embedding_dim)

        if self.use_gating:
            gate_logits = self.gate(x_flat) / self.temperature
            gate_scores = F.softmax(gate_logits, dim=-1)
        else:
            gate_scores = torch.ones(
                x_flat.size(0), self.num_experts, device=x.device
            ) / self.num_experts

        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_scores[:, i].unsqueeze(-1)

            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_outputs[mask] += expert_weight[mask] * expert_output

        expert_outputs = expert_outputs.view(batch_size, seq_len, embedding_dim)

        auxiliary_loss = self._compute_load_balance_loss(gate_scores, top_k_indices)

        return expert_outputs, auxiliary_loss

    def _compute_load_balance_loss(
        self, gate_scores: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = gate_scores.size(0)
        expert_counts = torch.zeros(
            self.num_experts, device=gate_scores.device
        )
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_counts.scatter_add_(
                0, expert_idx, torch.ones_like(expert_idx, dtype=torch.float)
            )

        expert_fraction = expert_counts / (num_tokens * self.top_k)
        routing_weights = gate_scores.mean(dim=0)

        loss = self.num_experts * torch.sum(expert_fraction * routing_weights)
        return loss


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with optional MoE.

    If use_moe is True, replaces the standard feed-forward network with a
    Mixture-of-Experts module, enabling sparse computation and expert
    specialization.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.use_moe = use_moe

        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)

        if use_moe and moe_config:
            self.ffn = MixtureOfExperts(
                embedding_dim=embedding_dim,
                num_experts=moe_config.get("num_experts", 4),
                top_k=moe_config.get("top_k", 2),
                ff_dim=ff_dim,
                dropout=dropout,
                use_gating=moe_config.get("use_gating_network", True),
                temperature=moe_config.get("temperature", 1.0),
            )
        else:
            self.ffn = FeedForward(embedding_dim, ff_dim, dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        auxiliary_loss = None
        if self.use_moe:
            ffn_out, auxiliary_loss = self.ffn(x)
        else:
            ffn_out = self.ffn(x)

        x = self.norm2(x + self.dropout(ffn_out))

        return x, auxiliary_loss


class TransformerMoE(nn.Module):
    """Configurable Transformer encoder with optional Mixture-of-Experts.

    This model supports both pure Transformer architectures and hybrid
    Transformer+MoE designs. Configuration is entirely driven by the config
    dictionary, allowing easy experimentation with different architectural
    choices.

    The model expects input token indices and produces classification logits.
    When MoE is enabled, it also returns auxiliary losses for load balancing.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        data_config = config.get("data", {})
        transformer_config = config.get("transformer", {})
        moe_config = config.get("moe", {})

        self.input_dim = data_config.get("input_dim", 768)
        self.num_classes = data_config.get("num_classes", 2)
        self.embedding_dim = transformer_config.get("embedding_dim", 256)
        self.num_layers = transformer_config.get("num_layers", 6)
        self.num_heads = transformer_config.get("num_heads", 8)
        ff_dim_config = transformer_config.get("ff_dim")
        self.ff_dim = ff_dim_config if ff_dim_config is not None else self.embedding_dim * 4
        self.dropout = transformer_config.get("dropout", 0.1)
        self.max_seq_length = transformer_config.get("max_seq_length", 512)
        self.use_moe = transformer_config.get("use_moe", False)
        self.moe_layer_indices = transformer_config.get("moe_layer_indices", None)
        self.load_balance_weight = moe_config.get("load_balance_weight", 0.01)

        positional_encoding_type = transformer_config.get("positional_encoding", "sinusoidal")

        self.input_projection = nn.Linear(self.input_dim, self.embedding_dim)

        if positional_encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(
                self.embedding_dim, self.max_seq_length
            )
        elif positional_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(
                self.embedding_dim, self.max_seq_length
            )
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            use_moe_this_layer = self._should_use_moe_in_layer(layer_idx)
            layer = TransformerEncoderLayer(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                use_moe=use_moe_this_layer,
                moe_config=moe_config if use_moe_this_layer else None,
            )
            self.layers.append(layer)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def _should_use_moe_in_layer(self, layer_idx: int) -> bool:
        if not self.use_moe:
            return False
        if self.moe_layer_indices is None:
            return True
        return layer_idx in self.moe_layer_indices

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.input_projection(x)

        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        x = self.dropout_layer(x)

        total_auxiliary_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, auxiliary_loss = layer(x, mask)
            if auxiliary_loss is not None:
                total_auxiliary_loss = total_auxiliary_loss + auxiliary_loss

        x = x.mean(dim=1)

        logits = self.classifier(x)

        output = {
            "logits": logits,
            "auxiliary_loss": total_auxiliary_loss * self.load_balance_weight,
        }

        return output


__all__ = [
    "TransformerMoE",
    "MixtureOfExperts",
    "TransformerEncoderLayer",
    "MultiHeadAttention",
    "FeedForward",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
]

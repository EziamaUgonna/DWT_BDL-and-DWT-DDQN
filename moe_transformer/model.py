"""PyTorch implementation of a Mixture-of-Experts Transformer model.

This module provides a lightweight yet configurable Mixture-of-Experts (MoE)
Transformer encoder tailored for experimentation and federated learning
scenarios. The design emphasises modularity, allowing the MoE feed-forward
layer to be toggled for ablation studies, while exposing auxiliary statistics
that are helpful when analysing expert utilisation across clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MoETransformerConfig:
    """Configuration object used to build :class:`MoETransformer`.

    Attributes:
        input_dim: Dimensionality of the incoming feature vectors.
        model_dim: Dimensionality used internally by the Transformer encoder.
        num_classes: Output dimensionality (e.g. vocabulary size or number of
            classes). The model returns logits with shape ``(B, T, num_classes)``.
        num_layers: Number of encoder layers.
        num_heads: Number of self-attention heads.
        dim_feedforward: Width of the intermediate feed-forward layers.
        dropout: Dropout probability applied throughout the encoder.
        activation: Name of the activation function (``relu`` or ``gelu``).
        use_moe: If ``True`` the feed-forward block is replaced by a
            Mixture-of-Experts layer, otherwise a standard dense feed-forward
            network is used.
        num_experts: Number of experts in the MoE layer.
        top_k: Number of experts chosen per token during routing.
        expert_hidden_dim: Hidden dimensionality for each expert. Defaults to
            ``dim_feedforward`` when ``None``.
        max_seq_len: Maximum sequence length supported by the positional
            encoding table.
        layer_norm_eps: Epsilon used in LayerNorm layers for numerical
            stability.
        moe_load_balancing_weight: Weighting factor applied to the load
            balancing auxiliary loss returned by the model.
    """

    input_dim: int
    model_dim: int
    num_classes: int
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    use_moe: bool = True
    num_experts: int = 4
    top_k: int = 2
    expert_hidden_dim: Optional[int] = None
    max_seq_len: int = 512
    layer_norm_eps: float = 1e-5
    moe_load_balancing_weight: float = 0.01


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


def _get_activation_fn(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation function: {name}")


class FeedForwardExpert(nn.Module):
    """Single expert consisting of a two-layer feed-forward network."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TopKGate(nn.Module):
    """Gating network that selects the top-k experts for each token."""

    def __init__(self, model_dim: int, num_experts: int, k: int) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("top_k must be at least 1")
        if k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.linear = nn.Linear(model_dim, num_experts)
        self.num_experts = num_experts
        self.k = k

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute routing decisions.

        Args:
            inputs: Tensor of shape ``(N, D)`` where ``N`` is the number of
                tokens (batch * sequence length).

        Returns:
            topk_indices: LongTensor of shape ``(N, k)`` with the expert indices
                selected for each token.
            topk_gates: Tensor of shape ``(N, k)`` containing normalised gate
                values (sum to 1 across ``k`` for each token).
            all_probabilities: Tensor of shape ``(N, num_experts)`` with the
                full softmax distribution prior to truncation.
        """

        logits = self.linear(inputs)
        all_probabilities = F.softmax(logits, dim=-1)
        topk_logits, topk_indices = torch.topk(logits, k=self.k, dim=-1)
        # Renormalise over the top-k choices to form a valid mixture.
        topk_gates = F.softmax(topk_logits, dim=-1)
        return topk_indices, topk_gates, all_probabilities


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer with top-k gating."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [FeedForwardExpert(d_model, hidden_dim, dropout, activation) for _ in range(num_experts)]
        )
        self.gate = TopKGate(d_model, num_experts, top_k)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Route tokens through experts.

        Args:
            inputs: Tensor of shape ``(B, T, D)``.

        Returns:
            mixture: Tensor of shape ``(B, T, D)``.
            load_balance_loss: Scalar tensor encouraging even expert usage.
            metadata: Dictionary with routing diagnostics.
        """

        batch, seq_len, d_model = inputs.shape
        flat_inputs = inputs.reshape(-1, d_model)
        num_tokens = flat_inputs.size(0)

        if num_tokens == 0:
            zero = inputs.new_zeros(())
            metadata = {
                "tokens_per_expert": inputs.new_zeros(self.num_experts, dtype=torch.long),
                "prob_per_expert": inputs.new_zeros(self.num_experts),
                "assignment_fraction": inputs.new_zeros(self.num_experts),
                "load_balance_loss": zero.detach(),
            }
            return inputs, zero, metadata

        topk_indices, topk_gates, all_probs = self.gate(flat_inputs)

        # Build dense representations for routing statistics.
        dispatch_mask = torch.zeros(num_tokens, self.num_experts, device=inputs.device, dtype=flat_inputs.dtype)
        gate_mask = torch.zeros_like(dispatch_mask)
        one_hot = torch.ones_like(topk_gates)
        dispatch_mask.scatter_add_(1, topk_indices, one_hot)
        gate_mask.scatter_add_(1, topk_indices, topk_gates)

        expert_outputs = torch.zeros_like(flat_inputs)
        tokens_per_expert = (dispatch_mask > 0).sum(dim=0)
        assignment_fraction = gate_mask.sum(dim=0) / num_tokens
        expected_prob = all_probs.mean(dim=0)

        for expert_idx, expert in enumerate(self.experts):
            token_mask = dispatch_mask[:, expert_idx] > 0
            if token_mask.any():
                expert_input = flat_inputs[token_mask]
                gate_values = gate_mask[token_mask, expert_idx].unsqueeze(-1)
                expert_output = expert(expert_input)
                expert_outputs[token_mask] += expert_output * gate_values

        outputs = expert_outputs.reshape(batch, seq_len, d_model)
        load_balance_loss = (expected_prob * assignment_fraction).sum() * self.num_experts

        metadata = {
            "tokens_per_expert": tokens_per_expert,
            "prob_per_expert": expected_prob.detach(),
            "assignment_fraction": assignment_fraction.detach(),
            "load_balance_loss": load_balance_loss.detach(),
        }
        return outputs, load_balance_loss, metadata


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder block with optional MoE feed-forward layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        use_moe: bool,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoELayer(d_model, num_experts, top_k, dim_feedforward, dropout, activation)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = _get_activation_fn(activation)
            self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attn_output))

        aux_loss = src.new_zeros(())
        metadata: Dict[str, Tensor] = {}
        if self.use_moe:
            moe_output, moe_loss, moe_metadata = self.moe(src)
            src = self.norm2(src + self.dropout(moe_output))
            aux_loss = moe_loss
            metadata = moe_metadata
        else:
            ff_output = self.linear2(self.ff_dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout(ff_output))

        return src, aux_loss, metadata


class MoETransformer(nn.Module):
    """Configurable Transformer encoder with optional Mixture-of-Experts layers."""

    def __init__(self, config: MoETransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = (
            nn.Identity() if config.input_dim == config.model_dim else nn.Linear(config.input_dim, config.model_dim)
        )
        self.position_encoding = PositionalEncoding(config.model_dim, config.dropout, config.max_seq_len)

        layers: List[TransformerEncoderLayer] = []
        for _ in range(config.num_layers):
            layers.append(
                TransformerEncoderLayer(
                    d_model=config.model_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.expert_hidden_dim or config.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.activation,
                    layer_norm_eps=config.layer_norm_eps,
                    use_moe=config.use_moe,
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        self.output_head = nn.Linear(config.model_dim, config.num_classes)

    def forward(
        self,
        inputs: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Perform a forward pass.

        Args:
            inputs: Tensor of shape ``(B, T, input_dim)``.
            src_mask: Optional sequence mask broadcastable to ``(B, T, T)``.
            src_key_padding_mask: Optional mask shaped ``(B, T)`` where masked
                positions are filled with ``True``.

        Returns:
            logits: Tensor with shape ``(B, T, num_classes)``.
            aux_loss: Scalar tensor representing the weighted auxiliary loss
                (load balancing when ``use_moe`` is enabled).
            metadata: Dictionary containing diagnostics useful for federated
                training (e.g. per-layer expert utilisation statistics).
        """

        x = self.input_projection(inputs)
        x = self.position_encoding(x)

        aux_losses: List[Tensor] = []
        diagnostic: List[Dict[str, Tensor]] = []

        for layer in self.layers:
            x, layer_aux, layer_meta = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            if layer_aux is not None:
                aux_losses.append(layer_aux)
            if layer_meta:
                diagnostic.append(layer_meta)

        x = self.norm(x)
        logits = self.output_head(x)

        aux_loss = inputs.new_zeros(())
        if aux_losses:
            stacked = torch.stack(aux_losses)
            aux_loss = stacked.sum() * self.config.moe_load_balancing_weight

        metadata: Dict[str, Any] = {
            "layers": len(self.layers),
            "sequence_length": inputs.size(1),
            "moe_metrics": diagnostic,
        }
        return logits, aux_loss, metadata

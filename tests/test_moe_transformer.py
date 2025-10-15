"""Smoke tests for the Mixture-of-Experts Transformer implementation."""

import torch

from models import MoEConfig, MoETransformer, MoETransformerConfig


def test_moe_transformer_forward_with_moe():
    config = MoETransformerConfig(
        input_dim=16,
        model_dim=32,
        num_classes=4,
        num_layers=2,
        num_heads=4,
        ff_hidden_dim=64,
        max_seq_len=32,
        use_moe=True,
        moe=MoEConfig(
            num_experts=4,
            expert_hidden_size=64,
            top_k=2,
            load_balancing_weight=0.1,
            dropout=0.1,
        ),
    )
    model = MoETransformer(config)
    dummy_inputs = torch.randn(3, 10, 16)

    result = model(dummy_inputs)

    logits = result["logits"]
    assert logits.shape == (3, 4)

    aux_losses = result["auxiliary_losses"]
    assert "load_balancing" in aux_losses
    assert aux_losses["load_balancing"].shape == torch.Size([])

    metadata = result["metadata"]
    assert "expert_loads" in metadata
    assert metadata["expert_loads"].shape == (config.num_layers, config.moe.num_experts)
    assert metadata["attention_means"].shape == (
        config.num_layers,
        dummy_inputs.shape[0],
        config.num_heads,
    )


def test_moe_transformer_forward_without_moe():
    config = MoETransformerConfig(
        input_dim=8,
        model_dim=16,
        num_classes=3,
        num_layers=1,
        num_heads=2,
        ff_hidden_dim=32,
        max_seq_len=16,
        use_moe=False,
    )
    model = MoETransformer(config)
    dummy_inputs = torch.randn(2, 6, 8)

    padding_mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [False, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    result = model(dummy_inputs, padding_mask=padding_mask)

    logits = result["logits"]
    assert logits.shape == (2, 3)

    aux_losses = result["auxiliary_losses"]
    assert torch.allclose(aux_losses["load_balancing"], torch.tensor(0.0))

    metadata = result["metadata"]
    assert "expert_loads" not in metadata
    assert metadata["attention_means"].shape == (
        config.num_layers,
        dummy_inputs.shape[0],
        config.num_heads,
    )
    assert torch.equal(metadata["valid_tokens_per_batch"], torch.tensor([3, 2]))
    assert metadata["sequence_length"].item() == dummy_inputs.shape[1]

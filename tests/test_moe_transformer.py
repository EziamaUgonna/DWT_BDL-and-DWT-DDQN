import torch

from moe_transformer import MoETransformer, MoETransformerConfig


def build_config(**overrides):
    config = MoETransformerConfig(
        input_dim=16,
        model_dim=32,
        num_classes=8,
        num_layers=2,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.1,
        activation="gelu",
        use_moe=True,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=None,
        max_seq_len=64,
        layer_norm_eps=1e-5,
        moe_load_balancing_weight=0.05,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_moe_transformer_forward_with_moe():
    config = build_config()
    model = MoETransformer(config)
    dummy_input = torch.randn(3, 12, config.input_dim)
    logits, aux_loss, metadata = model(dummy_input)

    assert logits.shape == (3, 12, config.num_classes)
    assert aux_loss.shape == ()
    assert torch.isfinite(aux_loss)
    assert metadata["layers"] == config.num_layers
    assert metadata["sequence_length"] == dummy_input.size(1)
    assert len(metadata["moe_metrics"]) == config.num_layers
    first_layer_metrics = metadata["moe_metrics"][0]
    assert first_layer_metrics["tokens_per_expert"].shape[0] == config.num_experts
    assert torch.all(first_layer_metrics["prob_per_expert"] >= 0)


def test_moe_transformer_forward_without_moe():
    config = build_config(use_moe=False)
    model = MoETransformer(config)
    dummy_input = torch.randn(2, 5, config.input_dim)
    logits, aux_loss, metadata = model(dummy_input)

    assert logits.shape == (2, 5, config.num_classes)
    assert aux_loss.shape == ()
    assert aux_loss == 0
    assert metadata["moe_metrics"] == []

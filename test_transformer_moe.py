"""Simple test script to verify TransformerMoE implementation."""

import torch
from config.config import get_default_config
from models.transformer_moe import TransformerMoE


def test_pure_transformer():
    """Test pure Transformer without MoE."""
    print("Testing pure Transformer...")
    config = get_default_config()
    config["transformer"]["use_moe"] = False
    config["data"]["input_dim"] = 128
    config["data"]["num_classes"] = 10
    config["transformer"]["embedding_dim"] = 64
    config["transformer"]["num_layers"] = 2
    config["transformer"]["num_heads"] = 4

    model = TransformerMoE(config)
    batch_size = 8
    seq_len = 16
    input_dim = 128

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    assert "logits" in output
    assert "auxiliary_loss" in output
    assert output["logits"].shape == (batch_size, 10)
    assert output["auxiliary_loss"].item() == 0.0

    print(f"✓ Pure Transformer output shape: {output['logits'].shape}")
    print(f"✓ Auxiliary loss: {output['auxiliary_loss'].item()}")


def test_transformer_with_moe():
    """Test Transformer with MoE in all layers."""
    print("\nTesting Transformer + MoE (all layers)...")
    config = get_default_config()
    config["transformer"]["use_moe"] = True
    config["data"]["input_dim"] = 128
    config["data"]["num_classes"] = 10
    config["transformer"]["embedding_dim"] = 64
    config["transformer"]["num_layers"] = 2
    config["transformer"]["num_heads"] = 4
    config["moe"]["num_experts"] = 4
    config["moe"]["top_k"] = 2

    model = TransformerMoE(config)
    batch_size = 8
    seq_len = 16
    input_dim = 128

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    assert "logits" in output
    assert "auxiliary_loss" in output
    assert output["logits"].shape == (batch_size, 10)
    assert output["auxiliary_loss"].item() > 0.0

    print(f"✓ Transformer+MoE output shape: {output['logits'].shape}")
    print(f"✓ Auxiliary loss: {output['auxiliary_loss'].item()}")


def test_transformer_with_selective_moe():
    """Test Transformer with MoE in selected layers only."""
    print("\nTesting Transformer + MoE (selective layers)...")
    config = get_default_config()
    config["transformer"]["use_moe"] = True
    config["transformer"]["moe_layer_indices"] = [1]
    config["data"]["input_dim"] = 128
    config["data"]["num_classes"] = 10
    config["transformer"]["embedding_dim"] = 64
    config["transformer"]["num_layers"] = 3
    config["transformer"]["num_heads"] = 4
    config["moe"]["num_experts"] = 4
    config["moe"]["top_k"] = 2

    model = TransformerMoE(config)
    batch_size = 8
    seq_len = 16
    input_dim = 128

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    assert "logits" in output
    assert "auxiliary_loss" in output
    assert output["logits"].shape == (batch_size, 10)
    assert output["auxiliary_loss"].item() > 0.0

    print(f"✓ Selective MoE output shape: {output['logits'].shape}")
    print(f"✓ Auxiliary loss: {output['auxiliary_loss'].item()}")


def test_different_positional_encodings():
    """Test different positional encoding types."""
    print("\nTesting different positional encodings...")

    for encoding_type in ["sinusoidal", "learned", "none"]:
        print(f"  Testing {encoding_type} encoding...")
        config = get_default_config()
        config["transformer"]["positional_encoding"] = encoding_type
        config["data"]["input_dim"] = 128
        config["data"]["num_classes"] = 10
        config["transformer"]["embedding_dim"] = 64
        config["transformer"]["num_layers"] = 2
        config["transformer"]["num_heads"] = 4

        model = TransformerMoE(config)
        x = torch.randn(4, 8, 128)
        output = model(x)

        assert output["logits"].shape == (4, 10)
        print(f"  ✓ {encoding_type} encoding works correctly")


if __name__ == "__main__":
    test_pure_transformer()
    test_transformer_with_moe()
    test_transformer_with_selective_moe()
    test_different_positional_encodings()
    print("\n✅ All tests passed!")

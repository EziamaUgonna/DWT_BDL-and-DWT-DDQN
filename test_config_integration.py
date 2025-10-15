"""Test configuration integration with Transformer+MoE model."""

import torch
from config.config import load_config
from models import TransformerMoE


def test_default_config():
    """Test model instantiation with default config."""
    print("Testing default config...")
    config = load_config(cli_args=[])
    
    assert "transformer" in config
    assert config["transformer"]["embedding_dim"] == 256
    assert config["transformer"]["num_layers"] == 6
    assert config["transformer"]["num_heads"] == 8
    assert config["transformer"]["use_moe"] is False
    assert config["moe"]["num_experts"] == 4
    assert config["moe"]["top_k"] == 2
    assert config["moe"]["load_balance_weight"] == 0.01
    
    print("✓ Default config values are correct")


def test_cli_overrides():
    """Test CLI argument overrides for transformer config."""
    print("\nTesting CLI argument overrides...")
    
    cli_args = [
        "--transformer-embedding-dim", "512",
        "--transformer-num-layers", "8",
        "--transformer-num-heads", "16",
        "--transformer-use-moe", "true",
        "--moe-num-experts", "8",
        "--moe-top-k", "3",
    ]
    
    config = load_config(cli_args=cli_args)
    
    assert config["transformer"]["embedding_dim"] == 512
    assert config["transformer"]["num_layers"] == 8
    assert config["transformer"]["num_heads"] == 16
    assert config["transformer"]["use_moe"] is True
    assert config["moe"]["num_experts"] == 8
    assert config["moe"]["top_k"] == 3
    
    print("✓ CLI overrides work correctly")


def test_dict_overrides():
    """Test dictionary overrides."""
    print("\nTesting dictionary overrides...")
    
    overrides = {
        "transformer": {
            "embedding_dim": 128,
            "num_layers": 4,
            "use_moe": True,
            "moe_layer_indices": [1, 3],
        },
        "moe": {
            "num_experts": 6,
            "temperature": 0.5,
        },
    }
    
    config = load_config(cli_args=[], overrides=overrides)
    
    assert config["transformer"]["embedding_dim"] == 128
    assert config["transformer"]["num_layers"] == 4
    assert config["transformer"]["use_moe"] is True
    assert config["transformer"]["moe_layer_indices"] == [1, 3]
    assert config["moe"]["num_experts"] == 6
    assert config["moe"]["temperature"] == 0.5
    
    print("✓ Dictionary overrides work correctly")


def test_model_with_cli_config():
    """Test model instantiation with CLI-configured settings."""
    print("\nTesting model instantiation with CLI config...")
    
    cli_args = [
        "--data-input-dim", "256",
        "--data-num-classes", "5",
        "--transformer-embedding-dim", "128",
        "--transformer-num-layers", "3",
        "--transformer-num-heads", "4",
        "--transformer-dropout", "0.2",
        "--transformer-use-moe", "true",
        "--moe-num-experts", "4",
        "--moe-top-k", "2",
    ]
    
    config = load_config(cli_args=cli_args)
    model = TransformerMoE(config)
    
    x = torch.randn(8, 16, 256)
    output = model(x)
    
    assert output["logits"].shape == (8, 5)
    assert output["auxiliary_loss"].item() > 0.0
    
    print("✓ Model works with CLI config")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Auxiliary loss: {output['auxiliary_loss'].item():.6f}")


def test_all_positional_encoding_types():
    """Test all positional encoding types via config."""
    print("\nTesting positional encoding config...")
    
    for encoding in ["sinusoidal", "learned", "none"]:
        cli_args = [
            "--transformer-positional-encoding", encoding,
            "--data-input-dim", "128",
            "--data-num-classes", "3",
        ]
        
        config = load_config(cli_args=cli_args)
        assert config["transformer"]["positional_encoding"] == encoding
        
        model = TransformerMoE(config)
        x = torch.randn(4, 8, 128)
        output = model(x)
        
        assert output["logits"].shape == (4, 3)
        print(f"  ✓ {encoding} encoding works")


def test_moe_selective_layers():
    """Test selective MoE layer insertion via config."""
    print("\nTesting selective MoE layer configuration...")
    
    overrides = {
        "data": {"input_dim": 128, "num_classes": 5},
        "transformer": {
            "embedding_dim": 64,
            "num_layers": 6,
            "num_heads": 4,
            "use_moe": True,
            "moe_layer_indices": [2, 4, 5],
        },
        "moe": {
            "num_experts": 4,
            "top_k": 2,
        },
    }
    
    config = load_config(cli_args=[], overrides=overrides)
    model = TransformerMoE(config)
    
    x = torch.randn(4, 8, 128)
    output = model(x)
    
    assert output["logits"].shape == (4, 5)
    assert output["auxiliary_loss"].item() > 0.0
    
    print("✓ Selective MoE layers work")
    print(f"  Total layers: {config['transformer']['num_layers']}")
    print(f"  MoE layers: {config['transformer']['moe_layer_indices']}")
    print(f"  Regular FFN layers: [0, 1, 3]")


def test_ff_dim_override():
    """Test ff_dim override (None should default to 4 * embedding_dim)."""
    print("\nTesting ff_dim configuration...")
    
    config1 = load_config(cli_args=[], overrides={
        "data": {"input_dim": 128, "num_classes": 3},
        "transformer": {"embedding_dim": 128, "ff_dim": None},
    })
    model1 = TransformerMoE(config1)
    assert model1.ff_dim == 128 * 4
    print(f"✓ ff_dim=None defaults to 4 * embedding_dim = {model1.ff_dim}")
    
    config2 = load_config(cli_args=["--transformer-ff-dim", "256"], overrides={
        "data": {"input_dim": 128, "num_classes": 3},
        "transformer": {"embedding_dim": 128},
    })
    model2 = TransformerMoE(config2)
    assert model2.ff_dim == 256
    print(f"✓ ff_dim can be explicitly set = {model2.ff_dim}")


def test_load_balance_weight():
    """Test load balance weight configuration."""
    print("\nTesting load balance weight...")
    
    config = load_config(cli_args=["--moe-load-balance-weight", "0.05"], overrides={
        "data": {"input_dim": 128, "num_classes": 3},
        "transformer": {
            "embedding_dim": 64,
            "num_layers": 2,
            "use_moe": True,
        },
    })
    
    assert config["moe"]["load_balance_weight"] == 0.05
    
    model = TransformerMoE(config)
    x = torch.randn(4, 8, 128)
    output = model(x)
    
    print(f"✓ Load balance weight: {config['moe']['load_balance_weight']}")
    print(f"  Auxiliary loss: {output['auxiliary_loss'].item():.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Integration Tests")
    print("=" * 60)
    
    test_default_config()
    test_cli_overrides()
    test_dict_overrides()
    test_model_with_cli_config()
    test_all_positional_encoding_types()
    test_moe_selective_layers()
    test_ff_dim_override()
    test_load_balance_weight()
    
    print("\n" + "=" * 60)
    print("✅ All configuration integration tests passed!")
    print("=" * 60)

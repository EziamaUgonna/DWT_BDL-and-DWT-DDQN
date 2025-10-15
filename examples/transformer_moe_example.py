"""Example: Training Transformer+MoE in a federated setting.

This script demonstrates how to use the TransformerMoE model in different
configurations and simulates a simple federated learning scenario.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from config.config import load_config
from models import TransformerMoE


def example_pure_transformer():
    """Example 1: Pure Transformer without MoE."""
    print("=" * 60)
    print("Example 1: Pure Transformer")
    print("=" * 60)
    
    config = load_config(overrides={
        "data": {
            "input_dim": 768,
            "num_classes": 10,
        },
        "transformer": {
            "embedding_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "positional_encoding": "sinusoidal",
            "use_moe": False,
        },
    })
    
    model = TransformerMoE(config)
    
    batch_size = 32
    seq_len = 64
    input_dim = config["data"]["input_dim"]
    num_classes = config["data"]["num_classes"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    output = model(x)
    logits = output["logits"]
    auxiliary_loss = output["auxiliary_loss"]
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Auxiliary loss: {auxiliary_loss.item():.6f}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_transformer_with_moe():
    """Example 2: Transformer with MoE in all layers."""
    print("=" * 60)
    print("Example 2: Transformer + MoE (All Layers)")
    print("=" * 60)
    
    config = load_config(overrides={
        "data": {
            "input_dim": 768,
            "num_classes": 10,
        },
        "transformer": {
            "embedding_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "use_moe": True,
        },
        "moe": {
            "num_experts": 8,
            "top_k": 2,
            "use_gating_network": True,
            "temperature": 0.7,
            "load_balance_weight": 0.01,
        },
    })
    
    model = TransformerMoE(config)
    
    batch_size = 32
    seq_len = 64
    input_dim = config["data"]["input_dim"]
    num_classes = config["data"]["num_classes"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    output = model(x)
    logits = output["logits"]
    auxiliary_loss = output["auxiliary_loss"]
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Auxiliary loss: {auxiliary_loss.item():.6f} (encourages balanced expert usage)")
    print(f"Number of experts per layer: {config['moe']['num_experts']}")
    print(f"Active experts per token: {config['moe']['top_k']}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_selective_moe():
    """Example 3: Transformer with MoE in selected layers only."""
    print("=" * 60)
    print("Example 3: Transformer + Selective MoE")
    print("=" * 60)
    
    config = load_config(overrides={
        "data": {
            "input_dim": 768,
            "num_classes": 10,
        },
        "transformer": {
            "embedding_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "use_moe": True,
            "moe_layer_indices": [2, 4],  # Only layers 2 and 4 use MoE
        },
        "moe": {
            "num_experts": 4,
            "top_k": 2,
        },
    })
    
    model = TransformerMoE(config)
    
    batch_size = 32
    seq_len = 64
    input_dim = config["data"]["input_dim"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    print(f"Total layers: {config['transformer']['num_layers']}")
    print(f"MoE layers: {config['transformer']['moe_layer_indices']}")
    print(f"Regular FFN layers: [0, 1, 3, 5]")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Auxiliary loss: {output['auxiliary_loss'].item():.6f}")
    print()


def example_training_loop():
    """Example 4: Complete training loop with loss computation."""
    print("=" * 60)
    print("Example 4: Training Loop with Auxiliary Loss")
    print("=" * 60)
    
    config = load_config(overrides={
        "data": {"input_dim": 128, "num_classes": 5},
        "transformer": {
            "embedding_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "use_moe": True,
        },
        "moe": {
            "num_experts": 4,
            "top_k": 2,
            "load_balance_weight": 0.01,
        },
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 16,
        },
    })
    
    model = TransformerMoE(config)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    batch_size = config["training"]["batch_size"]
    seq_len = 32
    input_dim = config["data"]["input_dim"]
    num_classes = config["data"]["num_classes"]
    
    print("Running 3 training steps...")
    for step in range(3):
        x = torch.randn(batch_size, seq_len, input_dim)
        y = torch.randint(0, num_classes, (batch_size,))
        
        optimizer.zero_grad()
        
        output = model(x)
        
        task_loss = criterion(output["logits"], y)
        auxiliary_loss = output["auxiliary_loss"]
        
        total_loss = task_loss + auxiliary_loss
        
        total_loss.backward()
        optimizer.step()
        
        print(f"  Step {step + 1}: "
              f"Task Loss = {task_loss.item():.4f}, "
              f"Aux Loss = {auxiliary_loss.item():.4f}, "
              f"Total = {total_loss.item():.4f}")
    print()


def example_federated_simulation():
    """Example 5: Simulated federated learning with multiple clients."""
    print("=" * 60)
    print("Example 5: Federated Learning Simulation")
    print("=" * 60)
    
    config = load_config(overrides={
        "data": {"input_dim": 128, "num_classes": 3},
        "transformer": {
            "embedding_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "use_moe": True,
        },
        "moe": {
            "num_experts": 4,
            "top_k": 2,
        },
        "federated": {
            "num_clients": 3,
            "local_epochs": 2,
        },
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 8,
        },
    })
    
    num_clients = config["federated"]["num_clients"]
    local_epochs = config["federated"]["local_epochs"]
    
    global_model = TransformerMoE(config)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Simulating {num_clients} clients, each training for {local_epochs} epochs")
    print()
    
    for round_idx in range(2):
        print(f"Round {round_idx + 1}")
        print("-" * 40)
        
        client_weights = []
        
        for client_id in range(num_clients):
            client_model = TransformerMoE(config)
            client_model.load_state_dict(global_model.state_dict())
            
            optimizer = optim.SGD(client_model.parameters(), lr=config["training"]["learning_rate"])
            
            for epoch in range(local_epochs):
                x = torch.randn(8, 16, 128)
                y = torch.randint(0, 3, (8,))
                
                optimizer.zero_grad()
                output = client_model(x)
                loss = criterion(output["logits"], y) + output["auxiliary_loss"]
                loss.backward()
                optimizer.step()
            
            client_weights.append({k: v.clone() for k, v in client_model.state_dict().items()})
            print(f"  Client {client_id + 1} completed local training")
        
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.stack([w[key] for w in client_weights]).mean(0)
        
        global_model.load_state_dict(aggregated_weights)
        print(f"  ✓ Aggregated weights from {num_clients} clients")
        print()
    
    print("Federated training simulation complete!")
    print()


def example_positional_encodings():
    """Example 6: Different positional encoding strategies."""
    print("=" * 60)
    print("Example 6: Positional Encoding Comparison")
    print("=" * 60)
    
    for encoding_type in ["sinusoidal", "learned", "none"]:
        config = load_config(overrides={
            "data": {"input_dim": 128, "num_classes": 5},
            "transformer": {
                "embedding_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "positional_encoding": encoding_type,
            },
        })
        
        model = TransformerMoE(config)
        x = torch.randn(8, 32, 128)
        output = model(x)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"{encoding_type.capitalize():12} encoding: "
              f"{param_count:,} parameters, "
              f"output shape {output['logits'].shape}")
    
    print()
    print("Notes:")
    print("- Sinusoidal: Fixed encoding, no learned parameters")
    print("- Learned: Trainable positional embeddings")
    print("- None: No positional information (order-invariant)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer + MoE Configuration Examples")
    print("=" * 60 + "\n")
    
    example_pure_transformer()
    example_transformer_with_moe()
    example_selective_moe()
    example_training_loop()
    example_federated_simulation()
    example_positional_encodings()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

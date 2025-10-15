# Models

This directory contains model architectures for the federated Mixture of Experts project.

## Transformer + MoE Backbone

The `transformer_moe.py` module provides a configurable Transformer architecture with optional Mixture-of-Experts (MoE) integration, designed for federated learning scenarios.

### Key Features

- **Pure Transformer Mode**: Standard Transformer encoder with multi-head attention and feed-forward layers
- **Hybrid Transformer+MoE Mode**: Optionally replace feed-forward layers with MoE blocks for sparse computation
- **Flexible MoE Integration**: Insert MoE in all layers or selectively in specific layers
- **Multiple Positional Encodings**: Support for sinusoidal, learned, or no positional encoding
- **Load Balancing**: Auxiliary loss to encourage uniform expert utilization
- **Federated-Ready**: Returns auxiliary metadata for distributed training

### Architecture Components

#### Core Modules

- **`TransformerMoE`**: Main model class that orchestrates the entire architecture
- **`TransformerEncoderLayer`**: Single encoder layer with optional MoE
- **`MultiHeadAttention`**: Standard multi-head self-attention mechanism
- **`FeedForward`**: Position-wise feed-forward network
- **`MixtureOfExperts`**: MoE layer with top-k routing and load balancing
- **`SinusoidalPositionalEncoding`**: Fixed sine/cosine positional embeddings
- **`LearnedPositionalEncoding`**: Trainable positional embeddings

### Usage Examples

#### Pure Transformer

```python
from config.config import load_config
from models import TransformerMoE

config = load_config(overrides={
    "data": {"input_dim": 768, "num_classes": 10},
    "transformer": {
        "embedding_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "use_moe": False,
    }
})

model = TransformerMoE(config)
output = model(input_tensor)  # shape: (batch, seq_len, input_dim)
logits = output["logits"]  # shape: (batch, num_classes)
```

#### Transformer with Full MoE

```python
config = load_config(overrides={
    "transformer": {
        "embedding_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "use_moe": True,  # Enable MoE in all layers
    },
    "moe": {
        "num_experts": 8,
        "top_k": 2,
        "temperature": 0.5,
        "load_balance_weight": 0.01,
    }
})

model = TransformerMoE(config)
output = model(input_tensor)
logits = output["logits"]
auxiliary_loss = output["auxiliary_loss"]  # For load balancing

# Total training loss
total_loss = cross_entropy_loss + auxiliary_loss
```

#### Selective MoE Insertion

```python
config = load_config(overrides={
    "transformer": {
        "num_layers": 12,
        "use_moe": True,
        "moe_layer_indices": [4, 8, 11],  # Only layers 4, 8, 11 use MoE
    },
    "moe": {
        "num_experts": 4,
        "top_k": 2,
    }
})

model = TransformerMoE(config)
```

### Configuration Reference

All configuration is driven by the `config` dictionary. See `config/config.py` for full documentation.

**Transformer Parameters** (`config["transformer"]`):
- `embedding_dim`: Token embedding dimension (default: 256)
- `num_layers`: Number of encoder layers (default: 6)
- `num_heads`: Number of attention heads (default: 8)
- `ff_dim`: Feed-forward hidden dimension (default: 4 * embedding_dim)
- `dropout`: Dropout probability (default: 0.1)
- `positional_encoding`: "sinusoidal", "learned", or "none" (default: "sinusoidal")
- `max_seq_length`: Maximum sequence length (default: 512)
- `use_moe`: Enable MoE blocks (default: False)
- `moe_layer_indices`: List of layer indices for MoE, or None for all (default: None)

**MoE Parameters** (`config["moe"]`):
- `num_experts`: Number of expert networks (default: 4)
- `top_k`: Number of experts to activate per token (default: 2)
- `use_gating_network`: Use learned gating vs random routing (default: True)
- `temperature`: Temperature for gating softmax (default: 1.0)
- `load_balance_weight`: Weight for auxiliary load-balancing loss (default: 0.01)

### Federated Learning Integration

The model is designed for federated learning scenarios:

1. **Auxiliary Loss**: The load-balancing loss encourages uniform expert usage across federated clients
2. **Output Format**: Returns dict with `logits` and `auxiliary_loss` for easy integration
3. **Expert Specialization**: Different clients can develop specialized experts based on local data

Example federated training loop:

```python
# On each client
model = TransformerMoE(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

for batch in local_dataloader:
    optimizer.zero_grad()
    
    output = model(batch["input"])
    
    # Main task loss
    task_loss = criterion(output["logits"], batch["labels"])
    
    # Add auxiliary load-balancing loss
    total_loss = task_loss + output["auxiliary_loss"]
    
    total_loss.backward()
    optimizer.step()

# Aggregate model parameters across clients using FedAvg or other methods
```

### Performance Considerations

- **Memory**: MoE increases model capacity without proportionally increasing per-sample compute
- **Sparsity**: Only top-k experts are activated per token, enabling larger models
- **Load Balancing**: The auxiliary loss prevents expert collapse and encourages diversity
- **Positional Encoding**: Sinusoidal encoding uses no learnable parameters; learned encoding adds flexibility

### Testing

Run the test suite to verify the implementation:

```bash
python test_transformer_moe.py
```

This tests:
- Pure Transformer mode
- Full MoE mode
- Selective MoE insertion
- Different positional encoding types
- Output shapes and auxiliary loss computation
